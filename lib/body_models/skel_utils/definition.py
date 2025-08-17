from skel.osim_rot import ConstantCurvatureJoint, CustomJoint, EllipsoidJoint, PinJoint, WalkerKnee

Q_COMPONENTS = [
    {'qid':  0, 'name': 'pelvis',    'jid':  0},
    {'qid':  1, 'name': 'pelvis',    'jid':  0},
    {'qid':  2, 'name': 'pelvis',    'jid':  0},
    {'qid':  3, 'name': 'femur-r',   'jid':  1},
    {'qid':  4, 'name': 'femur-r',   'jid':  1},
    {'qid':  5, 'name': 'femur-r',   'jid':  1},
    {'qid':  6, 'name': 'tibia-r',   'jid':  2},
    {'qid':  7, 'name': 'talus-r',   'jid':  3},
    {'qid':  8, 'name': 'calcn-r',   'jid':  4},
    {'qid':  9, 'name': 'toes-r',    'jid':  5},
    {'qid': 10, 'name': 'femur-l',   'jid':  6},
    {'qid': 11, 'name': 'femur-l',   'jid':  6},
    {'qid': 12, 'name': 'femur-l',   'jid':  6},
    {'qid': 13, 'name': 'tibia-l',   'jid':  7},
    {'qid': 14, 'name': 'talus-l',   'jid':  8},
    {'qid': 15, 'name': 'calcn-l',   'jid':  9},
    {'qid': 16, 'name': 'toes-l',    'jid': 10},
    {'qid': 17, 'name': 'lumbar',    'jid': 11},
    {'qid': 18, 'name': 'lumbar',    'jid': 11},
    {'qid': 19, 'name': 'lumbar',    'jid': 11},
    {'qid': 20, 'name': 'thorax',    'jid': 12},
    {'qid': 21, 'name': 'thorax',    'jid': 12},
    {'qid': 22, 'name': 'thorax',    'jid': 12},
    {'qid': 23, 'name': 'head',      'jid': 13},
    {'qid': 24, 'name': 'head',      'jid': 13},
    {'qid': 25, 'name': 'head',      'jid': 13},
    {'qid': 26, 'name': 'scapula-r', 'jid': 14},
    {'qid': 27, 'name': 'scapula-r', 'jid': 14},
    {'qid': 28, 'name': 'scapula-r', 'jid': 14},
    {'qid': 29, 'name': 'humerus-r', 'jid': 15},
    {'qid': 30, 'name': 'humerus-r', 'jid': 15},
    {'qid': 31, 'name': 'humerus-r', 'jid': 15},
    {'qid': 32, 'name': 'ulna-r',    'jid': 16},
    {'qid': 33, 'name': 'radius-r',  'jid': 17},
    {'qid': 34, 'name': 'hand-r',    'jid': 18},
    {'qid': 35, 'name': 'hand-r',    'jid': 18},
    {'qid': 36, 'name': 'scapula-l', 'jid': 19},
    {'qid': 37, 'name': 'scapula-l', 'jid': 19},
    {'qid': 38, 'name': 'scapula-l', 'jid': 19},
    {'qid': 39, 'name': 'humerus-l', 'jid': 20},
    {'qid': 40, 'name': 'humerus-l', 'jid': 20},
    {'qid': 41, 'name': 'humerus-l', 'jid': 20},
    {'qid': 42, 'name': 'ulna-l',    'jid': 21},
    {'qid': 43, 'name': 'radius-l',  'jid': 22},
    {'qid': 44, 'name': 'hand-l',    'jid': 23},
    {'qid': 45, 'name': 'hand-l',    'jid': 23},
]


QID2JID  = {c['qid']: c['jid'] for c in Q_COMPONENTS}

JID2QIDS = {}
for c in Q_COMPONENTS:
    jid = c['jid']
    JID2QIDS[jid] = [] if jid not in JID2QIDS else JID2QIDS[jid]
    JID2QIDS[jid].append(c['qid'])

JID2DOF  = {jid: len(qids) for jid, qids in JID2QIDS.items()}

DoF1_JIDS = [2, 3, 4, 5, 7, 8, 9, 10, 16, 17, 21, 22]  # (J1=12,)
DoF2_JIDS = [18, 23]  # (J2=2,)
DoF3_JIDS = [0, 1, 6, 11, 12, 13, 14, 15, 19, 20]  # (J3=10,)
DoF1_QIDS = [6, 7, 8, 9, 13, 14, 15, 16, 32, 33, 42, 43]  # (Q1=12,)
DoF2_QIDS = [34, 35, 44, 45]  # (Q2=4,)
DoF3_QIDS = [0, 1, 2, 3, 4, 5, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 37, 38, 39, 40, 41]  # (Q3=30,)


# Copied from the `skel_model.py`.
# Change all axis (except those PinJoint) to positive and update the flip if needed.
JOINTS_DEF = [
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]),             #  0 pelvis
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]),             #  1 femur_r
    WalkerKnee(),                                                                   #  2 tibia_r
    PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]),                  #  3 talus_r Field taken from .osim Joint-> frames -> PhysicalOffsetFrame -> orientation
    PinJoint(parent_frame_ori = [-1.76818999, 0.906223, 1.8196000]),                #  4 calcn_r
    PinJoint(parent_frame_ori = [-3.141589999, 0.6199010, 0]),                      #  5 toes_r
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, -1, -1]),           #  6 femur_l
    WalkerKnee(),                                                                   #  7 tibia_l
    PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]),                  #  8 talus_l
    PinJoint(parent_frame_ori = [1.768189999 ,-0.906223, 1.8196000]),               #  9 calcn_l
    PinJoint(parent_frame_ori = [-3.141589999, -0.6199010, 0]),                     # 10 toes_l
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]),  # 11 lumbar
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]),  # 12 thorax
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]),  # 13 head
    EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, -1, -1]),        # 14 scapula_r
    CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]),             # 15 humerus_r
    CustomJoint(axis=[[0.0494, 0.0366, 0.99810825]], axis_flip=[[1]]),              # 16 ulna_r
    CustomJoint(axis=[[-0.01716099, 0.99266564, -0.11966796]], axis_flip=[[1]]),    # 17 radius_r
    CustomJoint(axis=[[1,0,0], [0,0,1]], axis_flip=[1, -1]),                        # 18 hand_r
    EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, 1, 1]),          # 19 scapula_l
    CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]),             # 20 humerus_l
    CustomJoint(axis=[[-0.0494, -0.0366, 0.99810825]], axis_flip=[[1]]),            # 21 ulna_l
    CustomJoint(axis=[[0.01716099, -0.99266564, -0.11966796]], axis_flip=[[1]]),    # 22 radius_l
    CustomJoint(axis=[[1,0,0], [0,0,1]], axis_flip=[-1, -1]),                       # 23 hand_l
]

N_JOINTS = len(JOINTS_DEF)  # 24
