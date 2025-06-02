import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import trimesh
import pyrender

from typing import List, Optional, Union, Tuple
from pathlib import Path

from lib.utils.vis import ColorPalette
from lib.utils.data import to_numpy
from lib.utils.media import save_img

from .utils import *


def render_mesh_overlay_img(
    faces       : Union[torch.Tensor, np.ndarray],
    verts       : torch.Tensor,
    K4          : List,
    img         : np.ndarray,
    output_fn   : Optional[Union[str, Path]] = None,
    device      : str = 'cuda',
    resize      : float = 1.0,
    Rt          : Optional[Tuple[torch.Tensor]] = None,
    mesh_color : Optional[Union[List[float], str]] = 'green',
):
    '''
    Render the mesh overlay on the input video frames.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: torch.Tensor, (V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - img: np.ndarray, (H, W, 3)
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered img.
    - fps: int, default 30
    - device: str, default 'cuda'
    - resize: float, default 1.0
        - The resize factor of the output video.
    - Rt: Tuple of Tensor, default None
        - The extrinsic camera matrix, in the form of (R, t).
    '''
    frame = render_mesh_overlay_video(
            faces      = faces,
            verts      = verts[None],
            K4         = K4,
            frames     = img[None],
            device     = device,
            resize     = resize,
            Rt         = Rt,
            mesh_color = mesh_color,
        )[0]

    if output_fn is None:
        return frame
    else:
        save_img(frame, output_fn)


def render_mesh_overlay_video(
    faces      : Union[torch.Tensor, np.ndarray],
    verts      : Union[torch.Tensor, np.ndarray],
    K4         : List,
    frames     : np.ndarray,
    output_fn  : Optional[Union[str, Path]] = None,
    fps        : int = 30,
    device     : str = 'cuda',
    resize     : float = 1.0,
    Rt         : Tuple = None,
    mesh_color : Optional[Union[List[float], str]] = 'green',
):
    '''
    Render the mesh overlay on the input video frames.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: Union[torch.Tensor, np.ndarray], (L, V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - frames: np.ndarray, (L, H, W, 3)
    - output_fn: useless, only for compatibility.
    - fps: useless, only for compatibility.
    - device: useless, only for compatibility.
    - resize: useless, only for compatibility.
    - Rt: Tuple, default None
        - The extrinsic camera matrix, in the form of (R, t).
    '''
    faces, verts = to_numpy(faces), to_numpy(verts)
    assert len(K4) == 4, 'K4 must be a list of 4 elements.'
    assert frames.shape[0] == verts.shape[0], 'The length of frames and verts must be the same.'
    assert frames.shape[-1] == 3, 'The last dimension of frames must be 3.'
    if isinstance(mesh_color, str):
        mesh_color = ColorPalette.presets_float[mesh_color]

    # Prepare the data.
    L = len(frames)
    frame_w, frame_h = frames.shape[-2], frames.shape[-3]

    renderer = pyrender.OffscreenRenderer(
            viewport_width  = frame_w,
            viewport_height = frame_h,
            point_size      = 1.0
        )

    # Camera
    camera, cam_pose = create_camera(K4, Rt)

    # Scene.
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor  = 0.0,
            alphaMode       = 'OPAQUE',
            baseColorFactor = (*mesh_color, 1.0)
        )

    # Light.
    light_nodes = create_raymond_lights()

    results = []
    for i in range(L):
        mesh = trimesh.Trimesh(verts[i].copy(), faces.copy())
        # if side_view:
        #     rot = trimesh.transformations.rotation_matrix(
        #         np.radians(rot_angle), [0, 1, 0])
        #     mesh.apply_transform(rot)
        # elif top_view:
        #     rot = trimesh.transformations.rotation_matrix(
        #         np.radians(rot_angle), [1, 0, 0])
        #     mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(
                bg_color      = [0.0, 0.0, 0.0, 0.0],
                ambient_light = (0.3, 0.3, 0.3),
            )

        scene.add(mesh, 'mesh')
        scene.add(camera, pose=cam_pose)

        # Light.
        for node in light_nodes:
            scene.add_node(node)

        # Render.
        result_rgba, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        valid_mask = result_rgba.astype(np.float32)[:, :, [-1]] / 255.0  # (H, W, 1)
        bg = frames[i]  # (H, W, 3)
        final = result_rgba[:, :, :3] * valid_mask + bg * (1 - valid_mask)
        final = final.astype(np.uint8)  # (H, W, 3)
        results.append(final)
    results = np.stack(results, axis=0)  # (L, H, W, 3)

    renderer.delete()
    return results



def render_meshes_overlay_img(
    faces_all   : Union[torch.Tensor, np.ndarray],
    verts_all   : Union[torch.Tensor, np.ndarray],
    cam_t_all   : Union[torch.Tensor, np.ndarray],
    K4          : List,
    img         : np.ndarray,
    output_fn   : Optional[Union[str, Path]] = None,
    device      : str = 'cuda',
    resize      : float = 1.0,
    Rt          : Optional[Tuple[torch.Tensor]] = None,
    mesh_color : Optional[Union[List[float], str]] = 'green',
    view       : str = 'front',
    ret_rgba   : bool = False,
):
    '''
    Render the mesh overlay on the input video frames.

    ### Args
    - faces_all: Union[torch.Tensor, np.ndarray], ((Nm,) V, 3)
    - verts_all: Union[torch.Tensor, np.ndarray], ((Nm,) V, 3)
    - cam_t_all: Union[torch.Tensor, np.ndarray], ((Nm,) 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - img: np.ndarray, (H, W, 3)
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered img.
    - fps: int, default 30
    - device: str, default 'cuda'
    - resize: float, default 1.0
        - The resize factor of the output video.
    - Rt: Tuple of Tensor, default None
        - The extrinsic camera matrix, in the form of (R, t).
    - view: str, default 'front', {'front', 'side90d', 'side60d', 'top90d'}
    - ret_rgba: bool, default False
        - If True, return rgba images, otherwise return rgb images.
        - For view is not 'front', the background will become transparent.
    '''
    if len(verts_all.shape) == 2:
        verts_all = verts_all[None] # (1, V, 3)
    elif len(verts_all.shape) == 3:
        verts_all = verts_all[:, None]  # ((Nm,) 1, V, 3)
    else:
        raise ValueError('The shape of verts_all is not correct.')
    if len(cam_t_all.shape) == 1:
        cam_t_all = cam_t_all[None] # (1, 3)
    elif len(cam_t_all.shape) == 2:
        cam_t_all = cam_t_all[:, None]  # ((Nm,) 1, 3)
    else:
        raise ValueError('The shape of verts_all is not correct.')
    frame = render_meshes_overlay_video(
            faces_all  = faces_all,
            verts_all  = verts_all,
            cam_t_all  = cam_t_all,
            K4         = K4,
            frames     = img[None],
            device     = device,
            resize     = resize,
            Rt         = Rt,
            mesh_color = mesh_color,
            view       = view,
            ret_rgba   = ret_rgba,
        )[0]

    if output_fn is None:
        return frame
    else:
        save_img(frame, output_fn)


def render_meshes_overlay_video(
    faces_all  : Union[torch.Tensor, np.ndarray],
    verts_all  : Union[torch.Tensor, np.ndarray],
    cam_t_all  : Union[torch.Tensor, np.ndarray],
    K4         : List,
    frames     : np.ndarray,
    output_fn  : Optional[Union[str, Path]] = None,
    fps        : int = 30,
    device     : str = 'cuda',
    resize     : float = 1.0,
    Rt         : Tuple = None,
    mesh_color : Optional[Union[List[float], str]] = 'green',
    view       : str = 'front',
    ret_rgba   : bool = False,
):
    '''
    Render the mesh overlay on the input video frames.

    ### Args
    - faces_all: Union[torch.Tensor, np.ndarray], ((Nm,) V, 3)
    - verts_all: Union[torch.Tensor, np.ndarray], ((Nm,) L, V, 3)
    - cam_t_all: Union[torch.Tensor, np.ndarray], ((Nm,) L, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - frames: np.ndarray, (L, H, W, 3)
    - output_fn: useless, only for compatibility.
    - fps: useless, only for compatibility.
    - device: useless, only for compatibility.
    - resize: useless, only for compatibility.
    - Rt: Tuple, default None
        - The extrinsic camera matrix, in the form of (R, t).
    - view: str, default 'front', {'front', 'side90d', 'side60d', 'top90d'}
    - ret_rgba: bool, default False
        - If True, return rgba images, otherwise return rgb images.
        - For view is not 'front', the background will become transparent.
    '''
    faces_all, verts_all = to_numpy(faces_all), to_numpy(verts_all)
    if len(verts_all.shape) == 3:
        verts_all = verts_all[None]  # (1, L, V, 3)
    if len(cam_t_all.shape) == 2:
        cam_t_all = cam_t_all[None]  # (1, L, 3)
    Nm, L, _, _ = verts_all.shape
    if len(faces_all.shape) == 2:
        faces_all = faces_all[None].repeat(Nm, axis=0)  # (Nm, V, 3)

    assert len(K4) == 4, 'K4 must be a list of 4 elements.'
    assert frames.shape[0] == L, 'The length of frames and verts must be the same.'
    assert frames.shape[-1] == 3, 'The last dimension of frames must be 3.'
    assert len(verts_all.shape) == 4, 'The shape of verts_all must be (Nm, L, V, 3).'
    assert len(faces_all.shape) == 3, 'The shape of faces_all must be (Nm, V, 3).'
    if isinstance(mesh_color, str):
        mesh_color = ColorPalette.presets_float[mesh_color]

    # Prepare the data.
    frame_w, frame_h = frames.shape[-2], frames.shape[-3]

    renderer = pyrender.OffscreenRenderer(
            viewport_width  = frame_w,
            viewport_height = frame_h,
            point_size      = 1.0
        )

    # Camera
    camera, cam_pose = create_camera(K4, Rt)

    # Scene.
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor  = 0.0,
            alphaMode       = 'OPAQUE',
            baseColorFactor = (*mesh_color, 1.0)
        )

    # Light.
    light_nodes = create_raymond_lights()

    results = []
    for i in range(L):
        scene = pyrender.Scene(
                bg_color      = [0.0, 0.0, 0.0, 0.0],
                ambient_light = (0.3, 0.3, 0.3),
            )

        for mid in range(Nm):
            mesh = trimesh.Trimesh(verts_all[mid][i].copy(), faces_all[mid].copy())
            if view == 'front':
                pass
            elif view == 'side90d':
                rot = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
                mesh.apply_transform(rot)
            elif view == 'side60d':
                rot = trimesh.transformations.rotation_matrix(np.radians(-60), [0, 1, 0])
                mesh.apply_transform(rot)
            elif view == 'top90d':
                rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
                mesh.apply_transform(rot)
            else:
                raise ValueError('The view is not supported.')
            trans = trimesh.transformations.translation_matrix(to_numpy(cam_t_all[mid][i]))
            mesh.apply_transform(trans)
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

            scene.add(mesh, f'mesh_{mid}')
        scene.add(camera, pose=cam_pose)

        # Light.
        for node in light_nodes:
            scene.add_node(node)

        # Render.
        result_rgba, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        valid_mask = result_rgba.astype(np.float32)[:, :, [-1]] / 255.0  # (H, W, 1)
        if view == 'front':
            bg = frames[i]  # (H, W, 3)
        else:
            bg = np.ones_like(frames[i]) * 255  # (H, W, 3)
        if ret_rgba:
            if view == 'front':
                bg_alpha = np.ones_like(bg[..., [0]]) * 255  # (H, W, 1)
            else:
                bg_alpha = np.zeros_like(bg[..., [0]]) * 255  # (H, W, 1)
            bg = np.concatenate([bg, bg_alpha], axis=-1)  # (H, W, 4)
            final = result_rgba * valid_mask + bg * (1 - valid_mask)  # (H, W, 4)
        else:
            final = result_rgba[:, :, :3] * valid_mask + bg * (1 - valid_mask)
        final = final.astype(np.uint8)  # (H, W, 3)
        results.append(final)
    results = np.stack(results, axis=0)  # (L, H, W, 3)

    renderer.delete()
    return results
