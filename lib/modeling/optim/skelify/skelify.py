from lib.kits.basic import *

import cv2
import traceback
from tqdm import tqdm

from lib.body_models.common import make_SKEL
from lib.body_models.abstract_skeletons import Skeleton_OpenPose25
from lib.utils.vis import render_mesh_overlay_img
from lib.utils.data import to_tensor
from lib.utils.media import draw_kp2d_on_img, annotate_img, splice_img
from lib.utils.camera import perspective_projection

from .utils import (
    compute_rel_change,
    gmof,
)

from .closure import build_closure

class SKELify():

    def __init__(self, cfg, tb_logger=None, device='cuda:0', name='SKELify'):
        self.cfg = cfg
        self.name = name
        self.eq_thre = cfg.early_quit_thresholds

        self.tb_logger = tb_logger

        self.device = device
        # self.skel_model = make_SKEL(device=device)
        self.skel_model = instantiate(cfg.skel_model).to(device)

        # Shortcuts.
        self.n_samples = cfg.logger.samples_per_record


    def __call__(
        self,
        gt_kp2d    : Union[torch.Tensor, np.ndarray],
        init_poses : Union[torch.Tensor, np.ndarray],
        init_betas : Union[torch.Tensor, np.ndarray],
        init_cam_t : Union[torch.Tensor, np.ndarray],
        img_patch  : Optional[np.ndarray] = None,
        **kwargs
    ):
        '''
        Use optimization to fit the SKEL parameters to the 2D keypoints.

        ### Args:
        - gt_kp2d: torch.Tensor or np.ndarray, (B, J, 3)
               - The last three dim means [x, y, conf].
               - The 2D keypoints to fit, they are defined in [-0.5, 0.5], zero-centered space.
        - init_poses: torch.Tensor or np.ndarray, (B, 46)
        - init_betas: torch.Tensor or np.ndarray, (B, 10)
        - init_cam_t: torch.Tensor or np.ndarray, (B, 3)
        - img_patch: np.ndarray or None, (B, H, W, 3)
            - The image patch for visualization. H, W are defined in normalized bounding box space.
            - If it is None, the visualization will simply use a black background.

        ### Returns:
        - dict, containing the optimized parameters.
            - poses: torch.Tensor, (B, 46)
            - betas: torch.Tensor, (B, 10)
            - cam_t: torch.Tensor, (B, 3)
        '''
        self.init_v = None
        self.init_ct = None
        self.init_kp2d_err = None

        with PM.time_monitor('Input Preparation'):
            gt_kp2d = to_tensor(gt_kp2d, device=self.device).detach().float().clone()  # (B, J, 3)
            init_poses = to_tensor(init_poses, device=self.device).detach().float().clone()  # (B, 46)
            init_betas = to_tensor(init_betas, device=self.device).detach().float().clone()  # (B, 10)
            init_cam_t = to_tensor(init_cam_t, device=self.device).detach().float().clone()  # (B, 3)
            inputs = {
                    'poses_orient' : init_poses[:, :3],  # (B, 3)
                    'poses_body'   : init_poses[:, 3:],  # (B, 43)
                    'betas'        : init_betas,         # (B, 10)
                    'cam_t'        : init_cam_t,         # (B, 3)
                }
            focal_length = float(self.cfg.focal_length / self.cfg.img_patch_size)  # float

        # ⛩️ Optimization phases, controlled by config file.
        with PM.time_monitor('Optim') as tm:
            prev_steps = 0  # accumulate the steps are *supposed* to be done in the previous phases
            n_phases = len(self.cfg.phases)
            for phase_id, phase_name in enumerate(self.cfg.phases):
                phase_cfg = self.cfg.phases[phase_name]
                # 📦 Data preparation.
                optim_params = []
                for k in inputs.keys():
                    if k in phase_cfg.params_keys:
                        inputs[k].requires_grad = True
                        optim_params.append(inputs[k])  # (B, D)
                    else:
                        inputs[k].requires_grad = False
                log_data = {}
                tm.tick(f'Data preparation')

                # ⚙️ Optimization preparation.
                optimizer = instantiate(phase_cfg.optimizer, optim_params, _recursive_=True)
                closure = self._build_closure(
                        cfg=phase_cfg, optimizer=optimizer,  # basic
                        inputs=inputs, focal_length=focal_length, gt_kp2d=gt_kp2d,  # data reference
                        log_data=log_data,  # monitoring
                    )
                tm.tick(f'Optimizer * closure prepared.')

                # 🚀 Optimization loop.
                with tqdm(range(phase_cfg.max_loop)) as bar:
                    prev_loss = None
                    bar.set_description(f'[{phase_name}] Loss: ???')
                    for i in bar:
                        # 1. Main part of the optimization loop.
                        log_data.clear()
                        curr_loss = optimizer.step(closure)

                        # 2. Log.
                        if self.tb_logger is not None:
                            log_data.update({
                                'img_patch' : img_patch[:self.n_samples] if img_patch is not None else None,
                                'gt_kp2d'   : gt_kp2d[:self.n_samples].detach().clone(),
                            })
                            self._tb_log(prev_steps + i, phase_name, log_data)

                        # 3. The end of one optimization loop.
                        bar.set_description(f'[{phase_id+1}/{n_phases}] @ {phase_name} - Loss: {curr_loss:.4f}')
                        if self._can_early_quit(optim_params, prev_loss, curr_loss):
                            break
                        prev_loss = curr_loss

                    prev_steps += phase_cfg.max_loop
                    tm.tick(f'{phase_name} finished.')

        with PM.time_monitor('Last Inference'):
            poses = torch.cat([inputs['poses_orient'], inputs['poses_body']], dim=-1).detach().clone()  # (B, 46)
            betas = inputs['betas'].detach().clone()  # (B, 10)
            cam_t = inputs['cam_t'].detach().clone()  # (B, 3)
            skel_outputs = self.skel_model(poses=poses, betas=betas, skelmesh=False)  # (B, 44, 3)
            optim_kp3d = skel_outputs.joints  # (B, 44, 3)
            # Evaluate the confidence of the results.
            focal_length_xy = np.ones((len(poses), 2)) * focal_length  # (B, 2)
            optim_kp2d = perspective_projection(
                    points       = optim_kp3d,
                    translation  = cam_t,
                    focal_length = to_tensor(focal_length_xy, device=self.device),
                )
            kp2d_err = SKELify.eval_kp2d_err(gt_kp2d, optim_kp2d)  # (B,)

        # ⛩️ Prepare the output data.
        outputs = {
                'poses'    : poses,     # (B, 46)
                'betas'    : betas,     # (B, 10)
                'cam_t'    : cam_t,     # (B, 3)
                'kp2d_err' : kp2d_err,  # (B,)
            }
        return outputs


    def _can_early_quit(self, opt_params, prev_loss, curr_loss):
        ''' Judge whether to early quit the optimization process. If yes, return True, otherwise False.'''
        if self.cfg.early_quit_thresholds is None:
            # Never early quit.
            return False

        # Relative change test.
        if prev_loss is not None:
            loss_rel_change = compute_rel_change(prev_loss, curr_loss)
            if loss_rel_change < self.cfg.early_quit_thresholds.rel:
                get_logger().info(f'Early quit due to relative change: {loss_rel_change} = rel({prev_loss}, {curr_loss})')
                return True

        # Absolute change test.
        if all([
            torch.abs(param.grad.max()).item() < self.cfg.early_quit_thresholds.abs
            for param in opt_params if param.grad is not None
        ]):
            get_logger().info(f'Early quit due to absolute change.')
            return True

        return False


    def _build_closure(self, *args, **kwargs):
        # Using this way to hide the very details and simplify the code.
        return build_closure(self, *args, **kwargs)


    @staticmethod
    def eval_kp2d_err(gt_kp2d_with_conf:torch.Tensor, pd_kp2d:torch.Tensor):
        ''' Evaluate the mean 2D keypoints L2 error. The formula is: ∑(gt - pd)^2 * conf / ∑conf. '''
        assert len(gt_kp2d_with_conf.shape) == len(gt_kp2d_with_conf.shape), f'gt_kp2d_with_conf.shape={gt_kp2d_with_conf.shape}, pd_kp2d.shape={pd_kp2d.shape} but they should both be ((B,) J, D).'
        if len(gt_kp2d_with_conf.shape) == 2:
            gt_kp2d_with_conf, pd_kp2d = gt_kp2d_with_conf[None], pd_kp2d[None]
        assert len(gt_kp2d_with_conf.shape) == 3, f'gt_kp2d_with_conf.shape={gt_kp2d_with_conf.shape}, pd_kp2d.shape={pd_kp2d.shape} but they should both be ((B,) J, D).'
        B, J, _ = gt_kp2d_with_conf.shape
        assert gt_kp2d_with_conf.shape == (B, J, 3), f'gt_kp2d_with_conf.shape={gt_kp2d_with_conf.shape} but it should be ((B,) J, 3).'
        assert pd_kp2d.shape == (B, J, 2), f'pd_kp2d.shape={pd_kp2d.shape} but it should be ((B,) J, 2).'

        conf = gt_kp2d_with_conf[..., 2]  # (B, J)
        gt_kp2d = gt_kp2d_with_conf[..., :2]  # (B, J, 2)
        kp2d_err = torch.sum((gt_kp2d - pd_kp2d) ** 2, dim=-1) * conf  # (B, J)
        kp2d_err = kp2d_err.sum(dim=-1) / (torch.sum(conf, dim=-1) + 1e-6)  # (B,)
        return kp2d_err


    @rank_zero_only
    def _tb_log(self, step_cnt:int, phase_name:str, log_data:Dict, *args, **kwargs):
        ''' Write the logging information to the TensorBoard. '''
        if step_cnt != 0 and (step_cnt + 1) % self.cfg.logger.interval_skelify != 0:
            return

        summary_writer = self.tb_logger.experiment

        # Save losses.
        for loss_name, loss_val in log_data['losses'].items():
            summary_writer.add_scalar(f'skelify/{loss_name}', loss_val, step_cnt)

        # Visualization of the optimization process.  TODO: Maybe we can make this more elegant.
        if log_data['img_patch'] is None:
            log_data['img_patch'] = [np.zeros((self.cfg.img_patch_size, self.cfg.img_patch_size, 3), dtype=np.uint8)] \
                                  * len(log_data['gt_kp2d'])

        if self.init_v is None:
            self.init_v = log_data['pd_verts']
            self.init_ct = log_data['cam_t']
            self.init_kp2d_err = log_data['kp2d_err']

        # Overlay the skin mesh of the results on the original image.
        try:
            imgs_spliced = []
            for i, img_patch in enumerate(log_data['img_patch']):
                kp2d_err = log_data['kp2d_err'][i].item()

                img_with_init = render_mesh_overlay_img(
                        faces      = self.skel_model.skin_f,
                        verts      = self.init_v[i],
                        K4         = [self.cfg.focal_length, self.cfg.focal_length, 128, 128],
                        img        = img_patch,
                        Rt         = [torch.eye(3), self.init_ct[i]],
                        mesh_color = 'pink',
                    )
                img_with_init = annotate_img(img_with_init, 'init')
                img_with_init = annotate_img(img_with_init, f'Quality: {self.init_kp2d_err[i].item()*1000:.3f}/1e3', pos='tl')

                img_with_mesh = render_mesh_overlay_img(
                        faces      = self.skel_model.skin_f,
                        verts      = log_data['pd_verts'][i],
                        K4         = [self.cfg.focal_length, self.cfg.focal_length, 128, 128],
                        img        = img_patch,
                        Rt         = [torch.eye(3), log_data['cam_t'][i]],
                        mesh_color = 'pink',
                    )
                betas_max = log_data['optim_betas'][i].abs().max().item()
                img_patch_raw = annotate_img(img_patch, 'raw')

                log_data['gt_kp2d'][i][..., :2] = (log_data['gt_kp2d'][i][..., :2] + 0.5) * self.cfg.img_patch_size
                img_with_gt = annotate_img(img_patch, 'gt_kp2d')
                img_with_gt = draw_kp2d_on_img(
                        img_with_gt,
                        log_data['gt_kp2d'][i],
                        Skeleton_OpenPose25.bones,
                        Skeleton_OpenPose25.bone_colors,
                    )

                log_data['pd_kp2d'][i] = (log_data['pd_kp2d'][i] + 0.5) * self.cfg.img_patch_size
                img_with_pd = cv2.addWeighted(img_with_mesh, 0.7, img_patch, 0.3, 0)
                img_with_pd = draw_kp2d_on_img(
                        img_with_pd,
                        log_data['pd_kp2d'][i],
                        Skeleton_OpenPose25.bones,
                        Skeleton_OpenPose25.bone_colors,
                    )

                img_with_pd = annotate_img(img_with_pd, 'pd')
                img_with_pd = annotate_img(img_with_pd, f'Quality: {kp2d_err*1000:.3f}/1e3\nbetas_max: {betas_max:.3f}', pos='tl')
                img_with_mesh = annotate_img(img_with_mesh, f'Quality: {kp2d_err*1000:.3f}/1e3\nbetas_max: {betas_max:.3f}', pos='tl')
                img_with_mesh = annotate_img(img_with_mesh, 'pd_mesh')

                img_spliced = splice_img(
                        img_grids = [img_patch_raw, img_with_gt, img_with_pd, img_with_mesh, img_with_init],
                        grid_ids  = [[1, 2, 3, 4]],
                    )
                img_spliced = annotate_img(img_spliced, f'{phase_name}/{step_cnt}', pos=(32, 224))
                imgs_spliced.append(img_spliced)

            img_final = splice_img(imgs_spliced, grid_ids=[[i] for i in range(len(log_data['img_patch']))])

            img_final = to_tensor(img_final, device=None).permute(2, 0, 1)  # (3, H, W)
            summary_writer.add_image('skelify/visualization', img_final, step_cnt)
        except Exception as e:
            get_logger().error(f'Failed to visualize the optimization process: {e}')
            traceback.print_exc()