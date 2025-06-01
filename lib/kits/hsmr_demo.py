from lib.kits.basic import *

import cv2
import argparse
from tqdm import tqdm

from lib.version import DEFAULT_HSMR_ROOT
from lib.utils.vis.py_renderer import render_meshes_overlay_img, render_mesh_overlay_img
from lib.utils.bbox import crop_with_lurb, fit_bbox_to_aspect_ratio, lurb_to_cs, cs_to_lurb
from lib.utils.media import *
from lib.platform.monitor import TimeMonitor
from lib.platform.sliding_batches import asb
from lib.modeling.pipelines.hsmr import build_inference_pipeline
from lib.modeling.pipelines.vitdet import build_detector


IMG_MEAN_255 = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.
IMG_STD_255  = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.


# ================== Command Line Supports ==================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input_type', type=str, default='auto', help='Specify the input type. auto: file~video, folder~imgs', choices=['auto', 'video', 'imgs'])
    parser.add_argument('-i', '--input_path', type=str, required=True, help='The input images root or video file path.')
    parser.add_argument('-o', '--output_path', type=str, default=PM.outputs/'demos', help='The output root.')
    parser.add_argument('-m', '--model_root', type=str, default=DEFAULT_HSMR_ROOT, help='The model root which contains `.hydra/config.yaml`.')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='The device.')
    parser.add_argument('--det_bs', type=int, default=10, help='The max batch size for detector.')
    parser.add_argument('--det_mis', type=int, default=512, help='The max image size for detector.')
    parser.add_argument('--rec_bs', type=int, default=300, help='The batch size for recovery.')
    parser.add_argument('--max_instances', type=int, default=5, help='Max instances activated in one image.')
    parser.add_argument('--ignore_skel', action='store_true', help='Do not render skeleton to boost the rendering.')
    parser.add_argument('--have_caption', action='store_true', help='Add caption to the rendered images.')
    args = parser.parse_args()
    return args


# ================== Data Process Tools ==================

def load_inputs(args, MAX_IMG_W=1920, MAX_IMG_H=1080):
    # 1. Inference inputs type.
    inputs_path = Path(args.input_path)
    if args.input_type != 'auto': inputs_type = args.input_type
    else: inputs_type = 'video' if Path(args.input_path).is_file() else 'imgs'
    get_logger(brief=True).info(f'🚚 Loading inputs from: {inputs_path}, regarded as <{inputs_type}>.')

    # 2. Load inputs.
    inputs_meta = {'type': inputs_type}
    if inputs_type == 'video':
        inputs_meta['seq_name'] = inputs_path.stem
        frames, _ = load_video(inputs_path)
        if frames.shape[1] > MAX_IMG_H:
            frames = flex_resize_video(frames, (MAX_IMG_H, -1), kp_mod=4)
        if frames.shape[2] > MAX_IMG_W:
            frames = flex_resize_video(frames, (-1, MAX_IMG_W), kp_mod=4)
        raw_imgs = [frame for frame in frames]
    elif inputs_type == 'imgs':
        img_fns = list(inputs_path.glob('*.*'))
        img_fns = [fn for fn in img_fns if fn.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        inputs_meta['seq_name'] = f'{inputs_path.stem}-img_cnt={len(img_fns)}'
        raw_imgs = []
        for fn in img_fns:
            img, _ = load_img(fn)
            if img.shape[0] > MAX_IMG_H:
                img = flex_resize_img(img, (MAX_IMG_H, -1), kp_mod=4)
            if img.shape[1] > MAX_IMG_W:
                img = flex_resize_img(img, (-1, MAX_IMG_W), kp_mod=4)
            raw_imgs.append(img)
        inputs_meta['img_fns'] = img_fns
    else:
        raise ValueError(f'Unsupported inputs type: {inputs_type}.')
    get_logger(brief=True).info(f'📦 Totally {len(raw_imgs)} images are loaded.')

    return raw_imgs, inputs_meta


def imgs_det2patches(imgs, dets, downsample_ratios, max_instances_per_img):
    ''' Given the raw images and the detection results, return the image patches of human instances. '''
    assert len(imgs) == len(dets), f'L_img = {len(imgs)}, L_det = {len(dets)}'
    patches_per_img, n_patch_per_img, bbx_cs_per_img = [], [], []
    for i in tqdm(range(len(imgs))):
        patches_i, bbx_cs_i = _img_det2patches(imgs[i], dets[i], downsample_ratios[i], max_instances_per_img)
        n_patch_per_img.append(len(patches_i))
        if len(patches_i) > 0:
            patches_per_img.append(patches_i.astype(np.float32))
            bbx_cs_per_img.append(bbx_cs_i)
        else:
            patches_per_img.append(None)
            bbx_cs_per_img.append(None)
            get_logger(brief=True).warning(f'No human detection results on image No.{i}.')

    try:
        bbx_cs = np.concatenate([x for x in bbx_cs_per_img if x is not None], axis=0)  # (N, 3), center-scale bounding boxes
        patches = np.concatenate([x for x in patches_per_img if x is not None], axis=0)  # (N, 256, 256, 3)
    except:
        get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
        exit(-1)

    det_meta = {
            'n_patch_per_img' : n_patch_per_img,
            'bbx_cs_per_img'  : bbx_cs_per_img,
            'bbx_cs'          : bbx_cs
        }

    return patches, det_meta


def _img_det2patches(img, det_instances, downsample_ratio:float, max_instances:int=5):
    '''
    1. Filter out the trusted human detections.
    2. Enlarge the bounding boxes to aspect ratio (ViT backbone only use 192*256 pixels, make sure these 
       pixels can capture main contents) and then to squares (to adapt the data module).
    3. Crop the image with the bounding boxes and resize them to 256x256.
    4. Normalize the cropped images.
    '''
    if det_instances is None:  # no human detected
        return to_numpy([]), to_numpy([])
    CLASS_HUMAN_ID, DET_THRESHOLD_SCORE = 0, 0.5

    # Filter out the trusted human detections.
    is_human_mask = det_instances['pred_classes'] == CLASS_HUMAN_ID
    reliable_mask = det_instances['scores'] > DET_THRESHOLD_SCORE
    active_mask = is_human_mask & reliable_mask

    # Filter out the top-k human instances.
    if active_mask.sum().item() > max_instances:
        humans_scores = det_instances['scores'] * is_human_mask.float()
        _, top_idx = humans_scores.topk(max_instances)
        valid_mask = torch.zeros_like(active_mask).bool()
        valid_mask[top_idx] = True
    else:
        valid_mask = active_mask

    # Process the bounding boxes and crop the images.
    lurb_all = det_instances['pred_boxes'][valid_mask].numpy() / downsample_ratio  # (N, 4)
    lurb_all = [fit_bbox_to_aspect_ratio(bbox=lurb, tgt_ratio=(192, 256)) for lurb in lurb_all]  # regularize the bbox size
    cs_all   = [lurb_to_cs(lurb) for lurb in lurb_all]  # convert rectangle left-up-right-bottom bbx to square center-scale bbx
    lurb_all = [cs_to_lurb(cs) for cs in cs_all]  # convert square center-scale bbx to rectangle left-up-right-bottom bbx
    cropped_imgs = [crop_with_lurb(img, lurb) for lurb in lurb_all]
    patches = to_numpy([flex_resize_img(cropped_img, (256, 256)) for cropped_img in cropped_imgs])  # (N, 256, 256, 3)
    return patches, cs_all


# ================== Secondary Outputs Tools ==================

def prepare_mesh(pipeline, pd_params) -> Tuple[Dict, Dict]:
    B = 720  # full SKEL inference is memory consuming
    v_skin_all, v_skel_all = [], []
    for bw in asb(total=len(pd_params['poses']), bs_scope=B, enable_tqdm=True):
        skel_outputs = pipeline.skel_model(
                poses = pd_params['poses'][bw.sid:bw.eid].to(pipeline.device),
                betas = pd_params['betas'][bw.sid:bw.eid].to(pipeline.device),
            )
        v_skin = skel_outputs.skin_verts.detach().cpu()  # (B, Vi, 3)
        v_skel = skel_outputs.skel_verts.detach().cpu()  # (B, Ve, 3)
        v_skin_all.append(v_skin)
        v_skel_all.append(v_skel)
    v_skel_all = torch.cat(v_skel_all, dim=0)
    v_skin_all = torch.cat(v_skin_all, dim=0)
    m_skin = {'v': v_skin_all, 'f': pipeline.skel_model.skin_f}
    m_skel = {'v': v_skel_all, 'f': pipeline.skel_model.skel_f}
    return m_skin, m_skel


# ================== Visualization Tools ==================

def visualize_patches(pd_cam_t, patches, m_skin, m_skel) -> List[np.ndarray]:
    ''' Render the results to the patches. '''
    results = []
    for i in tqdm(range(len(patches)), desc='Rendering'):
        render_settings = {
                'img': patches[i].copy(),
                'K4' : [5000, 5000, 128, 128],
                'Rt' : [torch.eye(3).float(), pd_cam_t[i].float()],
            }
        overlay_smpl = render_mesh_overlay_img(
                faces      = m_skin['f'],
                verts      = m_skin['v'][i].float(),
                mesh_color = 'blue',
                **render_settings,
            )
        if m_skel is not None:
            overlay_skel = render_mesh_overlay_img(
                    faces      = m_skel['f'],
                    verts      = m_skel['v'][i].float(),
                    mesh_color = 'human_yellow',
                    **render_settings,
                )
            result = splice_img([patches[i], overlay_smpl, overlay_skel], grid_ids=[[0, 1, 2]])
        else:
            result = splice_img([patches[i], overlay_smpl], grid_ids=[[0, 1, -1]])
        results.append(result)
    return results


def visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, have_caption:bool=False) -> Tuple[List[np.ndarray], np.ndarray]:
    ''' Render the results to the patches. '''
    bbx_cs, n_patches_per_img = det_meta['bbx_cs'], det_meta['n_patch_per_img']

    results = []
    pp = 0  # patch pointer
    raw_cam_t_list = []
    for i in tqdm(range(len(raw_imgs)), desc='Rendering'):
        raw_h, raw_w = raw_imgs[i].shape[:2]
        raw_cx, raw_cy = raw_w/2, raw_h/2
        spp, epp = pp, pp + n_patches_per_img[i]

        # Rescale the camera translation.
        raw_cam_t_i = pd_cam_t[spp:epp].clone().float()
        bbx_s = to_tensor(bbx_cs[spp:epp, 2], device=raw_cam_t_i.device)
        bbx_cx = to_tensor(bbx_cs[spp:epp, 0], device=raw_cam_t_i.device)
        bbx_cy = to_tensor(bbx_cs[spp:epp, 1], device=raw_cam_t_i.device)

        raw_cam_t_i[:, 2] = pd_cam_t[spp:epp, 2] * 256 / bbx_s
        raw_cam_t_i[:, 1] += (bbx_cy - raw_cy) / 5000 * raw_cam_t_i[:, 2]
        raw_cam_t_i[:, 0] += (bbx_cx - raw_cx) / 5000 * raw_cam_t_i[:, 2]

        raw_cam_t_list.append(raw_cam_t_i)

        # Render overlays on the full image.
        full_img_bg = raw_imgs[i].copy()
        render_results = {}
        for view in ['front']:  # ['front', 'side60d', 'side90d', 'top90d']
            full_img_skin = render_meshes_overlay_img(
                        faces_all  = m_skin['f'],
                        verts_all  = m_skin['v'][spp:epp].float(),
                        cam_t_all  = raw_cam_t_i,
                        mesh_color = 'blue',
                        img        = full_img_bg,
                        K4         = [5000, 5000, raw_cx, raw_cy],
                        view       = view,
                    )

            if m_skel is not None:
                full_img_skel = render_meshes_overlay_img(
                        faces_all  = m_skel['f'],
                        verts_all  = m_skel['v'][spp:epp].float(),
                        cam_t_all  = raw_cam_t_i,
                        mesh_color = 'human_yellow',
                        img        = full_img_bg,
                        K4         = [5000, 5000, raw_cx, raw_cy],
                        view       = view,
                    )

            if m_skel is not None:
                full_img_blend = cv2.addWeighted(full_img_skin, 0.7, full_img_skel, 0.3, 0)
                render_results[f'{view}_blend'] = full_img_blend
                if view == 'front':
                    render_results[f'{view}_skel'] = full_img_skel
            else:
                render_results[f'{view}_skin'] = full_img_skin

        for view, img in render_results.items():
            desc = ' '.join(view.split('_'))
            if have_caption:
                render_results[view] = annotate_img(img, desc)

        # Merge views.
        if m_skel is not None:
            row = splice_img([
                    raw_imgs[i],
                    render_results['front_skel'],
                    render_results['front_blend'],
                ], grid_ids=[[0, 1, 2]])
            results.append(row)
        else:
            row = splice_img([
                    raw_imgs[i],
                    render_results['front_skin'],
                ], grid_ids=[[0, 1]])
            results.append(row)

        pp = epp

    raw_cam_t = to_numpy(torch.cat(raw_cam_t_list, dim=0))  # (N, 3)
    return results, raw_cam_t