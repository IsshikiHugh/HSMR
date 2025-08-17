import math
import torch

from skel.kin_skel import pose_param_names

# Different from the original one, has some modifications.
pose_limits = {
        'scapula_abduction_r' :  [-0.628, 0.628],  # 26
        'scapula_elevation_r' :  [-0.4, -0.1],  # 27
        'scapula_upward_rot_r' : [-0.190, 0.319],  # 28

        'scapula_abduction_l' :  [-0.628, 0.628],  # 36
        'scapula_elevation_l' :  [-0.4, -0.1],  # 37
        'scapula_upward_rot_l' : [-0.210, 0.219],  # 38

        'elbow_flexion_r' : [0, (3/4)*math.pi],  # 32
        'pro_sup_r'       : [-3/4*math.pi/2, 3/4*math.pi/2],  # 33
        'wrist_flexion_r' : [-math.pi/2, math.pi/2],  # 34
        'wrist_deviation_r' :[-math.pi/4, math.pi/4],  # 35

        'elbow_flexion_l' : [0, (3/4)*math.pi],  # 42
        'pro_sup_l'       : [-math.pi/2, math.pi/2],  # 43
        'wrist_flexion_l' : [-math.pi/2, math.pi/2],  # 44
        'wrist_deviation_l' :[-math.pi/4, math.pi/4],  # 45

        'lumbar_bending' : [-2/3*math.pi/4, 2/3*math.pi/4],  # 17
        'lumbar_extension' : [-math.pi/4, math.pi/4],  # 18
        'lumbar_twist' :  [-math.pi/4, math.pi/4],  # 19

        'thorax_bending' :[-math.pi/4, math.pi/4],  # 20
        'thorax_extension' :[-math.pi/4, math.pi/4],  # 21
        'thorax_twist' :[-math.pi/4, math.pi/4],  # 22

        'head_bending' :[-math.pi/4, math.pi/4],  # 23
        'head_extension' :[-math.pi/4, math.pi/4],  # 24
        'head_twist' :[-math.pi/4, math.pi/4],  # 25

        'ankle_angle_r' : [-math.pi/4, math.pi/4],  # 7
        'subtalar_angle_r' : [-math.pi/4, math.pi/4],  # 8
        'mtp_angle_r' : [-math.pi/4, math.pi/4],  # 9

        'ankle_angle_l' : [-math.pi/4, math.pi/4],  # 14
        'subtalar_angle_l' : [-math.pi/4, math.pi/4],  # 15
        'mtp_angle_l' : [-math.pi/4, math.pi/4],  # 16

        'knee_angle_r' : [0, 3/4*math.pi],  # 6
        'knee_angle_l' : [0, 3/4*math.pi],  # 13

        # Added by HSMR to make optimization more stable.
        'hip_flexion_r' : [-math.pi/4, 3/4*math.pi],  # 3
        'hip_adduction_r' : [-math.pi/4, 2/3*math.pi/4],  # 4
        'hip_rotation_r' : [-math.pi/4, math.pi/4],  # 5
        'hip_flexion_l' : [-math.pi/4, 3/4*math.pi],  # 10
        'hip_adduction_l' : [-math.pi/4, 2/3*math.pi/4],  # 11
        'hip_rotation_l' : [-math.pi/4, math.pi/4],  # 12

        'shoulder_r_x' : [-math.pi/2, math.pi/2+1.5],  # 29, from bsm.osim
        'shoulder_r_y' : [-math.pi/2, math.pi/2],  # 30
        'shoulder_r_z' : [-math.pi/2, math.pi/2],  # 31, from bsm.osim

        'shoulder_l_x' : [-math.pi/2-1.5, math.pi/2],  # 39, from bsm.osim
        'shoulder_l_y' : [-math.pi/2, math.pi/2],  # 40
        'shoulder_l_z' : [-math.pi/2, math.pi/2],  # 41, from bsm.osim
    }

pose_param_name2qid = {name: qid for qid, name in enumerate(pose_param_names)}
qid2pose_param_name = {qid: name for qid, name in enumerate(pose_param_names)}

SKEL_LIM_QIDS = []
SKEL_LIM_BOUNDS = []
for name, (low, up) in pose_limits.items():
    if low > up:
        low, up = up, low
    SKEL_LIM_QIDS.append(pose_param_name2qid[name])
    SKEL_LIM_BOUNDS.append([low, up])

SKEL_LIM_BOUNDS = torch.Tensor(SKEL_LIM_BOUNDS).float()
SKEL_LIM_QID2IDX = {qid: i for i, qid in enumerate(SKEL_LIM_QIDS)}  # inverse mapping