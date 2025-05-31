from lib.kits.basic import *

import torch
import numpy as np
import torch.nn.functional as F

from .definition import JOINTS_DEF, N_JOINTS, JID2DOF, JID2QIDS, DoF1_JIDS, DoF2_JIDS, DoF3_JIDS, DoF1_QIDS, DoF2_QIDS, DoF3_QIDS

from lib.utils.data import to_tensor
from lib.utils.geometry.rotation import (
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
)


# ====== Internal Utils ======


def axis2convention(axis:List):
    ''' [1,0,0] -> 'X', [0,1,0] -> 'Y', [0,0,1] -> 'Z' '''
    if axis == [1, 0, 0]:
        return 'X'
    elif axis == [0, 1, 0]:
        return 'Y'
    elif axis == [0, 0, 1]:
        return 'Z'
    else:
        raise ValueError(f'Unsupported axis: {axis}.')


def rotation_2d_to_angle(r2d:torch.Tensor):
    '''
    Extract single angle from a 2D rotation vector, which is the first column of a 2x2 rotation matrix.

    ### Args
    - r2d: torch.Tensor
        - shape = (...B, 2)

    ### Returns
    - shape = (...B,)
    '''
    cos, sin = r2d[..., [0]], -r2d[..., [1]]
    return torch.atan2(sin, cos)

# ====== Tools ======


OS2S_FLIP = [-1, 1, 1]
OS2S_CONV = 'YZX'
def real_orient_mat2q(orient_mat:torch.Tensor) -> torch.Tensor:
    '''
    The rotation matrix that SKEL uses is different from the SMPL's orientation matrix.
    The rotation to representation functions below can not be used to transform the rotaiton matrix.
    This function is used to convert the SMPL's orientation matrix to the SKEL's orientation q.
    BUT, is that really important? Maybe we shouldn't align SMPL's orientation with SKEL's, can they be different? 

    ### Args
    - orient_mat: torch.Tensor, shape = (..., 3, 3)

    ### Returns
    - orient_q: torch.Tensor, shape = (..., 3)
    '''
    device = orient_mat.device
    flip = to_tensor(OS2S_FLIP, device=device)  # (3,)
    orient_ea = matrix_to_euler_angles(orient_mat.clone(), OS2S_CONV)  # (..., 3)
    orient_ea = orient_ea[..., [2, 1, 0]]  # Re-permuting the order.
    orient_q = orient_ea * flip[None]
    return orient_q


def real_orient_q2mat(orient_q:torch.Tensor) -> torch.Tensor:
    '''
    The rotation matrix that SKEL uses is different from the SMPL's orientation matrix.
    The rotation to representation functions below can not be used to transform the rotation matrix.
    This function is used to convert the SKEL's orientation q to the SMPL's orientation matrix.
    BUT, is that really important? Maybe we shouldn't align SMPL's orientation with SKEL's, can they be different? 

    ### Args
    - orient_q: torch.Tensor, shape = (..., 3)

    ### Returns
    - orient_mat: torch.Tensor, shape = (..., 3, 3)
    '''
    device = orient_q.device
    flip = to_tensor(OS2S_FLIP, device=device)  # (3,)
    orient_ea = orient_q * flip[None]
    orient_ea = orient_ea[..., [2, 1, 0]]  # Re-permuting the order.
    orient_mat = euler_angles_to_matrix(orient_ea, OS2S_CONV)
    return orient_mat


def flip_params_lr(poses:torch.Tensor) -> torch.Tensor:
    '''
    It flips the skel through exchanging the params of left part and right part of the body. It's useful for
    data augmentation. Note that the 'left & right' defined when the body is facing z+ direction, this is
    only important for the orientation.

    ### Args
    - poses: torch.Tensor, shape = (B, L, 46) or (L, 46)

    ### Returns
    - flipped_poses: torch.Tensor, shape = (B, L, 46) or (L, 46)
    '''
    assert len(poses.shape) in [2, 3] and poses.shape[-1] == 46, f'Shape of poses should be (B, L, 46) or (L, 46) but get {poses.shape}.'

    # 1. Switch the value of each pair through re-permuting.
    flipped_perm = [
             0,  1,  2,  # pelvis
            10, 11, 12,  # femur-r -> femur-l
            13,          # tibia-r -> tibia-l
            14,          # talus-r -> talus-l
            15,          # calcn-r -> calcn-l
            16,          # toes-r -> toes-l
             3,  4,  5,  # femur-l -> femur-r
             6,          # tibia-l -> tibia-r
             7,          # talus-l -> talus-r
             8,          # calcn-l -> calcn-r
             9,          # toes-l -> toes-r
            17, 18, 19,  # lumbar
            20, 21, 22,  # thorax
            23, 24, 25,  # head
            36, 37, 38,  # scapula-r -> scapula-l
            39, 40, 41,  # humerus-r -> humerus-l
            42,          # ulna-r -> ulna-l
            43,          # radius-r -> radius-l
            44, 45,      # hand-r -> hand-l
            26, 27, 28,  # scapula-l -> scapula-r
            29, 30, 31,  # humerus-l -> humerus-r
            32,          # ulna-l -> ulna-r
            33,          # radius-l -> radius-r
            34, 35       # hand-l -> hand-r
        ]

    flipped_poses = poses[..., flipped_perm]

    # 2. Mirror the rotation direction through apply -1.
    flipped_sign = [
             1, -1, -1,  # pelvis
             1,  1,  1,  # femur-r'
             1,          # tibia-r'
             1,          # talus-r'
             1,          # calcn-r'
             1,          # toes-r'
             1,  1,  1,  # femur-l'
             1,          # tibia-l'
             1,          # talus-l'
             1,          # calcn-l'
             1,          # toes-l'
            -1,  1, -1,  # lumbar
            -1,  1, -1,  # thorax
            -1,  1, -1,  # head
            -1, -1,  1,  # scapula-r'
            -1, -1,  1,  # humerus-r'
             1,          # ulna-r'
             1,          # radius-r'
             1,  1,      # hand-r'
            -1, -1,  1,  # scapula-l'
            -1, -1,  1,  # humerus-l'
             1,          # ulna-l'
             1,          # radius-l'
             1,  1       # hand-l'
        ]
    flipped_sign = torch.tensor(flipped_sign, dtype=poses.dtype, device=poses.device)  # (46,)
    flipped_poses = flipped_sign * flipped_poses

    return flipped_poses



# def rotate_orient_around_z(q, rot):
#     """
#     Rotate SKEL orientation.
#     Args:
#         q (np.ndarray): SKEL style rotation representation (3,).
#         rot (np.ndarray): Rotation angle in degrees.
#     Returns:
#         np.ndarray: Rotated axis-angle vector.
#     """
#     import torch
#     from lib.body_models.skel.osim_rot import CustomJoint
#     # q to mat
#     root = CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1])  # pelvis
#     q = torch.from_numpy(q).unsqueeze(0)
#     q = q[:, [2, 1, 0]]
#     Rp = euler_angles_to_matrix(q, convention="YXZ")
#     # rotate around z
#     R = torch.Tensor([[np.deg2rad(-rot), 0, 0]])
#     R = axis_angle_to_matrix(R)
#     R = torch.matmul(R, Rp)
#     # mat to q
#     q = matrix_to_euler_angles(R, convention="YXZ")
#     q = q[:, [2, 1, 0]]
#     q = q.numpy().squeeze()

#     return q.astype(np.float32)


def params_q2rot(params_q:Union[torch.Tensor, np.ndarray]):
    '''
    Transform parts of the euler-like SKEL parameters representation all to rotation matrix.

    ### Args
    - params_q: Union[torch.Tensor, np.ndarray], shape = (...B, 46) or (...B, 46)

    ### Returns
    - shape = (...B, 24, 9)  # 24 joints, each joint has a 3x3 matrix, but for some joints, the matrix is not all used.
    '''
    # Check the type and unify to torch.
    is_np = isinstance(params_q, np.ndarray)
    if is_np:
        params_q = torch.from_numpy(params_q)

    # Prepare for necessary variables.
    Bs = params_q.shape[:-1]
    ident = torch.eye(3, dtype=params_q.dtype).to(params_q.device)  # (3, 3)
    params_rot = ident.repeat(*Bs, N_JOINTS, 1, 1)  # (...B, 24, 3, 3)

    # Deal with each joints separately. Modified from the `skel_model.py` but a static version.
    sid = 0
    for jid in range(N_JOINTS):
        joint_obj = JOINTS_DEF[jid]
        eid = sid + joint_obj.nb_dof.item()
        params_rot[..., jid, :, :] = joint_obj.q_to_rot(params_q[..., sid:eid])
        sid = eid

    if is_np:
        params_rot = params_rot.detach().cpu().numpy()
    return params_rot


def params_q2rep(params_q:Union[torch.Tensor, np.ndarray]):
    '''
    Transform the euler-like SKEL parameters representation to the continuous representation.
    This function is not supposed to be used in the training process, but only for debugging.
    The function that matters actually is the inverse of this function.

    ### Args
    - params_q: Union[torch.Tensor, np.ndarray], shape = (...B, 46) or (...B, 46)

    ### Returns
    - shape = (...B, 24, 6)
        - Among 24 joints, for 3 DoF ones, all 6 values are used to represent the rotation;
          but for 1 DoF joints, only the first 2 are used. The rest will be represented as zeros.
    '''
    # Check the type and unify to torch.
    is_np = isinstance(params_q, np.ndarray)
    if is_np:
        params_q = torch.from_numpy(params_q)

    # Prepare for necessary variables.
    Bs = params_q.shape[:-1]
    params_rep = params_q.new_zeros(*Bs, N_JOINTS, 6)  # (...B, 24, 6)

    # Deal with each joints separately. Modified from the `skel_model.py` but a static version.
    sid = 0
    for jid in range(N_JOINTS):
        joint_obj = JOINTS_DEF[jid]
        dof = joint_obj.nb_dof.item()
        eid = sid + dof
        if dof == 3:
            mat = joint_obj.q_to_rot(params_q[..., sid:eid])  # (...B, 3, 3)
            params_rep[..., jid, :] = matrix_to_rotation_6d(mat)  # (...B, 6)
        elif dof == 2:
            # mat = joint_obj.q_to_rot(params_q[..., sid:eid])  # (...B, 3, 3)
            # params_rep[..., jid, :] = matrix_to_rotation_6d(mat)  # (...B, 6)
            params_rep[..., jid, :2] = params_q[..., sid:eid]
        elif dof == 1:
            cos = torch.cos(params_q[..., sid])
            sin = torch.sin(params_q[..., sid])
            params_rep[..., jid, 0] = cos
            params_rep[..., jid, 1] = -sin

        sid = eid

    if is_np:
        params_rep = params_rep.detach().cpu().numpy()
    return params_rep


# Deprecated.
def dof3_to_q(rot, axises:List, flip:List):
    '''
    Convert a rotation matrix to SKEL style rotation representation.

    ### Args
    - rot: torch.Tensor, shape (...B, 3, 3)
        - The rotation matrix.
    - axises: list
        - [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]
        - The axis defined in the SKEL's joint_dict. Only one of xi, yi, zi is 1, the others are 0.
    - flip: list
        - [f0, f1, f2]
        - The flip value defined in the SKEL's joint_dict. fi is 1 or -1.

    ### Returns
    - shape = (...B, 3)
    '''
    convention = [axis2convention(axis) for axis in reversed(axises)]  # SKEL use euler angle in reverse order
    convention = ''.join(convention)
    q = matrix_to_euler_angles(rot[..., :, :], convention=convention)  # (...B, 3)
    q = q[..., [2, 1, 0]]  # SKEL use euler angle in reverse order
    flip = rot.new_tensor(flip)  # (3,)
    q = flip * q
    return q


### Slow version, deprecated. ###
# def params_rep2q(params_rot:Union[torch.Tensor, np.ndarray]):
#     '''
#     Transform the continuous representation back to the SKEL style euler-like representation.
#
#     ### Args
#     - params_rot: Union[torch.Tensor, np.ndarray]
#         - shape = (...B, 24, 6)
#
#     ### Returns
#     - shape = (...B, 46)
#     '''
#
#     # Check the type and unify to torch.
#     is_np = isinstance(params_rot, np.ndarray)
#     if is_np:
#         params_rot = torch.from_numpy(params_rot)
#
#     # Prepare for necessary variables.
#     Bs = params_rot.shape[:-2]
#     params_q = params_rot.new_zeros((*Bs, 46))  # (...B, 46)
#
#     for jid in range(N_JOINTS):
#         joint_obj = JOINTS_DEF[jid]
#         dof = joint_obj.nb_dof.item()
#         sid, eid = JID2QIDS[jid][0], JID2QIDS[jid][-1] + 1
#
#         if dof == 3:
#             mat = rotation_6d_to_matrix(params_rot[..., jid, :])  # (...B, 3, 3)
#             params_q[..., sid:eid] = dof3_to_q(
#                     mat,
#                     joint_obj.axis.tolist(),
#                     joint_obj.axis_flip.detach().cpu().tolist(),
#                 )
#         elif dof == 2:
#             params_q[..., sid:eid] = params_rot[..., jid, :2]
#         else:
#             params_q[..., sid:eid] = rotation_2d_to_angle(params_rot[..., jid, :2])
#
#     if is_np:
#         params_q = params_q.detach().cpu().numpy()
#     return params_q

def orient_mat2q(orient_mat:torch.Tensor):
    ''' This is a tool function for inspecting only. orient_mat ~ (...B, 3, 3)'''
    poses_rep = orient_mat.new_zeros(orient_mat.shape[:-2] + (24, 6))  # (...B, 24, 6)
    orient_rep = matrix_to_rotation_6d(orient_mat)  # (...B, 6)
    poses_rep[..., 0, :] = orient_rep
    poses_q = params_rep2q(poses_rep)  # (...B, 46)
    return poses_q[..., :3]


# Pre-grouped the joints for different conventions
CON_GROUP2JIDS  = {'YXZ': [0, 1, 6], 'YZX': [11, 12, 13], 'XZY': [14, 19], 'ZYX': [15, 20]}
CON_GROUP2FLIPS = {'YXZ': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, -1.0]], 'YZX': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 'XZY': [[1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], 'ZYX': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]}
# Faster version.
def params_rep2q(params_rot:Union[torch.Tensor, np.ndarray]):
    '''
    Transform the continuous representation back to the SKEL style euler-like representation.

    ### Args
    - params_rot: Union[torch.Tensor, np.ndarray], shape = (...B, 24, 6)

    ### Returns
    - shape = (...B, 46)
    '''

    with PM.time_monitor('params_rep2q'):
        with PM.time_monitor('preprocess'):
            params_rot, recover_type_back = to_tensor(params_rot, device=None, temporary=True)

            # Prepare for necessary variables.
            Bs = params_rot.shape[:-2]
            params_q = params_rot.new_zeros((*Bs, 46))  # (...B, 46)

        with PM.time_monitor(f'dof1&dof2'):
            params_q[..., DoF1_QIDS] = rotation_2d_to_angle(params_rot[..., DoF1_JIDS, :2]).squeeze(-1)
            params_q[..., DoF2_QIDS] = params_rot[..., DoF2_JIDS, :2].reshape(*Bs, -1)  # (...B, J2=2 * 2)

        with PM.time_monitor(f'dof3'):
            dof3_6ds = params_rot[..., DoF3_JIDS, :].reshape(*Bs, len(DoF3_JIDS), 6)  # (...B, J3=10, 3, 6)
            dof3_mats = rotation_6d_to_matrix(dof3_6ds)  # (...B, J3=10, 3, 3)

            for convention, jids in CON_GROUP2JIDS.items():
                idxs = [DoF3_JIDS.index(jid) for jid in jids]
                mats = dof3_mats[..., idxs, :, :]  # (...B, J', 3, 3)
                qs = matrix_to_euler_angles(mats, convention=convention)  # (...B, J', 3)
                qs = qs[..., [2, 1, 0]]  # SKEL use euler angle in reverse order
                flips = qs.new_tensor(CON_GROUP2FLIPS[convention])  # (J', 3)
                qs = qs * flips  # (...B, J', 3)
                qids = [qid for jid in jids for qid in JID2QIDS[jid]]
                params_q[..., qids] = qs.reshape(*Bs, -1)

        with PM.time_monitor('post_process'):
            params_q = recover_type_back(params_q)
    return params_q