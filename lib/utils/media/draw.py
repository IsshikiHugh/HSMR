from lib.kits.basic import *

import cv2
import imageio

from lib.utils.data import to_numpy
from lib.utils.vis import ColorPalette


def annotate_img(
    img  : np.ndarray,
    text : str,
    pos  : Union[str, Tuple[int, int]] = 'bl',
):
    '''
    Annotate the image with the given text.

    ### Args
    - img: np.ndarray, (H, W, 3)
    - text: str
    - pos: str or tuple(int, int), default 'bl'
        - If str, one of ['tl', 'bl'].
        - If tuple, (x, y), the position of the text.

    ### Returns
    - np.ndarray, (H, W, 3)
        - The annotated image.
    '''
    assert len(img.shape) == 3, 'img must have 3 dimensions.'
    return annotate_video(frames=img[None], text=text, pos=pos)[0]


def annotate_video(
    frames : np.ndarray,
    text   : str,
    pos    : Union[str, Tuple[int, int]] = 'bl',
    alpha  : float = 0.75,
):
    '''
    Annotate the video frames with the given text.

    ### Args
    - frames: np.ndarray, (L, H, W, 3)
    - text: str
    - pos: str or tuple(int, int), default 'bl'
        - If str, one of ['tl', 'bl'].
        - If tuple, (x, y), the position of the text.
    - alpha: float, default 0.5
        - The transparency of the text.

    ### Returns
    - np.ndarray, (L, H, W, 3)
        - The annotated video.
    '''
    assert len(frames.shape) == 4, 'frames must have 4 dimensions.'
    frames = frames.copy()
    L, H, W = frames.shape[:3]

    if isinstance(pos, str):
        if pos == 'tl':
            offset = (int(0.1 * W), int(0.1 * H))
        elif pos == 'bl':
            offset = (int(0.1 * W), int(0.9 * H))
        else:
            raise ValueError(f'Invalid position: {pos}')
    else:
        offset = pos

    for i, frame in enumerate(frames):
        overlay = frame.copy()
        _put_text(overlay, text, offset)
        frames[i] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frames


def draw_bbx_on_img(
    img   : np.ndarray,
    lurb  : np.ndarray,
    color : str = 'red',
):
    '''
    Draw the bounding box on the image.

    ### Args
    - img: np.ndarray, (H, W, 3)
    - lurb: np.ndarray, (4,)
        - The bounding box in the format of left, up, right, bottom.
    - color: str, default 'red'

    ### Returns
    - np.ndarray, (H, W, 3)
        - The image with the bounding box.
    '''
    assert len(img.shape) == 3, 'img must have 3 dimensions.'

    img = img.copy()
    l, u, r, b = lurb.astype(int)
    color_rgb_int8 = ColorPalette.presets_int8[color]
    cv2.rectangle(img, (l, u), (r, b), color_rgb_int8, 3)

    return img


def draw_bbx_on_video(
    frames : np.ndarray,
    lurbs  : np.ndarray,
    color  : str = 'red',
):
    '''
    Draw the bounding box on the video frames.

    ### Args
    - frames: np.ndarray, (L, H, W, 3)
    - lurbs: np.ndarray, (L, 4,)
        - The bounding box in the format of left, up, right, bottom.
    - color: str, default 'red'

    ### Returns
    - np.ndarray, (L, H, W, 3)
        - The video with the bounding box.
    '''
    assert len(frames.shape) == 4, 'frames must have 4 dimensions.'
    frames = frames.copy()

    for i, frame in enumerate(frames):
        frames[i] = draw_bbx_on_img(frame, lurbs[i], color)

    return frames


def draw_kp2d_on_img(
    img         : np.ndarray,
    kp2d        : Union[np.ndarray, torch.Tensor],
    links       : list = [],
    link_colors : list = [],
    show_conf   : bool = False,
    show_idx    : bool = False,
):
    '''
    Draw the 2d keypoints (and connection lines if exists) on the image.

    ### Args
    - img: np.ndarray, (H, W, 3)
        - The image.
    - kp2d: np.ndarray or torch.Tensor, (N, 2) or (N, 3)
        - The 2d keypoints without/with confidence.
    - links: list of [int, int] or (int, int), default []
        - The connections between keypoints. Each element is a tuple of two indices.
        - If empty, only keypoints will be drawn.
    - link_colors: list of [int, int, int] or (int, int, int), default []
        - The colors of the connections.
        - If empty, the connections will be drawn in white.
    - show_conf: bool, default False
        - Whether to show the confidence of keypoints.
    - show_idx: bool, default False
        - Whether to show the index of keypoints.

    ### Returns
    - img: np.ndarray, (H, W, 3)
        - The image with skeleton.
    '''
    img = img.copy()
    kp2d = to_numpy(kp2d)  # (N, 2) or (N, 3)
    assert len(img.shape) == 3, f'`img`\'s shape should be (H, W, 3) but got {img.shape}'
    assert len(kp2d.shape) == 2, f'`kp2d`\'s shape should be (N, 2) or (N, 3) but got {kp2d.shape}'

    if kp2d.shape[1] == 2:
        kp2d = np.concatenate([kp2d, np.ones((kp2d.shape[0], 1))], axis=-1)  # (N, 3)

    kp_has_drawn = [False] * kp2d.shape[0]
    # Draw connections.
    for lid, link in enumerate(links):
        # Skip links related to impossible keypoints.
        if kp2d[link[0], 2] < 0.5 or kp2d[link[1], 2] < 0.5:
            continue

        pt1 = tuple(kp2d[link[0], :2].astype(int))
        pt2 = tuple(kp2d[link[1], :2].astype(int))
        color = (255, 255, 255) if len(link_colors) == 0 else tuple(link_colors[lid])
        cv2.line(img, pt1, pt2, color, 2)
        if not kp_has_drawn[link[0]]:
            cv2.circle(img, pt1, 3, color, -1)
        if not kp_has_drawn[link[1]]:
            cv2.circle(img, pt2, 3, color, -1)
        kp_has_drawn[link[0]] = kp_has_drawn[link[1]] = True

    # Draw keypoints and annotate the confidence.
    for i, kp in enumerate(kp2d):
        conf = kp[2]
        pos = tuple(kp[:2].astype(int))

        if not kp_has_drawn[i]:
            cv2.circle(img, pos, 4, (255, 255, 255), -1)
            cv2.circle(img, pos, 2, (  0, 255,   0), -1)
        kp_has_drawn[i] = True

        if show_conf:
            _put_text(img, f'{conf:.2f}', pos)
        if show_idx:
            if i >= 40:
                continue
            _put_text(img, f'{i}', pos, scale=0.03)

    return img


# ====== Internal Utils ======

def _put_text(
    img          : np.ndarray,
    text         : str,
    pos          : Tuple[int, int],
    scale        : float = 0.05,
    color_inside : Tuple[int, int, int] = ColorPalette.presets_int8['black'],
    color_stroke : Tuple[int, int, int] = ColorPalette.presets_int8['white'],
    **kwargs
):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    if 'fontFace' in kwargs:
        fontFace = kwargs['fontFace']
        kwargs.pop('fontFace')

    H, W = img.shape[:2]
    # https://stackoverflow.com/a/55772676/22331129
    font_scale = scale * min(H, W) / 25 * 1.5
    thickness_inside = max(int(font_scale), 1)
    thickness_stroke = max(int(font_scale * 6), 6)

    # Deal with the multi-line text.

    ((fw, fh), baseline) = cv2.getTextSize(
            text      = text,
            fontFace  = fontFace,
            fontScale = font_scale,
            thickness = thickness_stroke,
        )  # https://stackoverflow.com/questions/73664883/opencv-python-draw-text-with-fontsize-in-pixels

    lines = text.split('\n')
    line_height = baseline + fh

    for i, line in enumerate(lines):
        pos_ = (pos[0], pos[1] + line_height * i)
        cv2.putText(img, line, pos_, fontFace, font_scale, color_stroke, thickness_stroke)
        cv2.putText(img, line, pos_, fontFace, font_scale, color_inside, thickness_inside)