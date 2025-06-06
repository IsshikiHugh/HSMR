from lib.kits.basic import *
from lib.kits.debug import *
from lib.platform import entrypoint_with_args
from lib.utils.media import (
    load_img,
    save_img,
    crop_img,
    draw_bbx_on_img,
    draw_kp2d_on_img,
    flex_resize_img
)
from pytorch_lightning.loggers import TensorBoardLogger

def prepare_data_simple_example(idx:int=0):
    # Load labels for kp2d.
    npz_fn = PM.inputs / 'hmr2_evaluation_data' / 'h36m_val_p2.npz'
    data = np.load(npz_fn)
    img_name = data['imgname'][idx].decode('ascii')
    img_fn = PM.inputs / 'datasets' / 'h36m' / 'images' / img_name
    bbox = data['bbox'][idx]  # (4,) np.ndarray
    bd_kp2d = data['body_keypoints_2d'][idx]  # (25, 3) np.ndarray
    ex_kp2d = data['extra_keypoints_2d'][idx]  # (19, 3) np.ndarray
    kp2d = np.concatenate([bd_kp2d, ex_kp2d], axis=0)  # (44, 3) np.ndarray

    # For visualization.
    img, _ = load_img(img_fn)
    img_patch = crop_img(img, bbox)
    img_patch = flex_resize_img(img_patch, (256, 256))

    # For optimization reference.
    kp2d[:, :2] = (kp2d[:, :2] - bbox[:2]) / (bbox[2:] - bbox[:2]) - 0.5

    # Load mean pose for initialization.
    mean_fn = PM.root / 'lib' / 'modeling' / 'networks' / 'heads' / 'SKEL_mean.npz'
    mean_data = np.load(mean_fn)
    init_cam_t = np.stack([
                    mean_data['cam'][1],
                    mean_data['cam'][2],
                    2 * 5000 / (256 * mean_data['cam'][0] + 1e-9)
                ], axis=-1)  # (3,)
    init_poses = mean_data['poses']  # (46,) np.ndarray
    init_betas = mean_data['betas']  # (10,) np.ndarray

    # This is an example of the input data format for the skelify pipeline.
    return {
        'gt_kp2d'    : kp2d[None],        # (1, 44, 3), in [-0.5, 0.5]
        'init_poses' : init_poses[None],  # (1, 46)
        'init_betas' : init_betas[None],  # (1, 10)
        'init_cam_t' : init_cam_t[None],  # (1, 3)
        'img_patch'  : img_patch[None],   # (1, H, W, 3)
    }


@entrypoint_with_args(exp='skelify', exp_tag='demo')
def main(cfg:DictConfig):
    # Change `idx` to switch to different examples.
    data = prepare_data_simple_example(idx=0)

    logger = TensorBoardLogger(
            save_dir = Path(cfg.output_dir) / 'tb_logs',
            name     = cfg.exp_name,
            version  = '',
        )

    skelify = instantiate(cfg.pipeline, tb_logger=logger, _recursive_=False)
    skelify(**data)

if __name__ == '__main__':
    main()