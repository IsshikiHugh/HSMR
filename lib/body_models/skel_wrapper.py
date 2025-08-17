from lib.kits.basic import *

import pickle
from skel.skel_model import SKEL, SKELOutput
from smplx.vertex_joint_selector import VertexJointSelector
from smplx.vertex_ids import vertex_ids
from smplx.lbs import vertices2joints


class SKELWrapper(SKEL):
    def __init__(
        self,
        *args,
        joint_regressor_custom: Optional[str] = None,
        joint_regressor_extra : Optional[str] = None,
        update_root : bool = False,
        **kwargs
    ):
        ''' This wrapper aims to extend the output joints of the SKEL model which fits SMPL's portal. '''

        super(SKELWrapper, self).__init__(*args, **kwargs)

        # The final joints are combined from three parts:
        # 1. The joints from the standard output.
        #    Map selected joints of interests from SKEL to SMPL. (Not all 24 joints will be used finally.)
        #    Notes: Only these SMPL joints will be used: [0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 45, 46, 47, 48, 49, 50, 51, 52, 53]
        skel_to_smpl = [
             0,
             6,
             1,
            11, # not aligned well; not used
             7,
             2,
            11, # not aligned well; not used
             8, # or 9
             3, # or 4
            12, # not aligned well; not used
            10, # not used
             5, # not used
            12,
            19, # not aligned well; not used
            14, # not aligned well; not used
            13, # not used
            20, # or 19
            15, # or 14
            21, # or 22
            16, # or 17,
            23,
            18,
            23, # not aligned well; not used
            18, # not aligned well; not used
        ]

        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

        self.register_buffer('J_skel_to_smpl', torch.tensor(skel_to_smpl, dtype=torch.long))
        self.register_buffer('J_smpl_to_openpose', torch.tensor(smpl_to_openpose, dtype=torch.long))
        # (SKEL has the same topology as SMPL as well as SMPL-H, so perform the same operation for the other 2 parts.)
        # 2. Joints selected from skin vertices.
        self.vertex_joint_selector = VertexJointSelector(vertex_ids['smplh'])
        # 3. Extra joints from the J_regressor_extra.
        if joint_regressor_extra is not None:
            self.register_buffer(
                'J_regressor_extra',
                torch.tensor(pickle.load(
                    open(joint_regressor_extra, 'rb'),
                    encoding='latin1'
                ), dtype=torch.float32)
            )

        self.custom_regress_joints = joint_regressor_custom is not None
        if self.custom_regress_joints:
            get_logger().info('Using customized joint regressor.')
            with open(joint_regressor_custom, 'rb') as f:
                J_regressor_custom = pickle.load(f, encoding='latin1')
            if 'scipy.sparse' in str(type(J_regressor_custom)):
                J_regressor_custom = J_regressor_custom.todense()  # (24, 6890)
            self.register_buffer(
                'J_regressor_custom',
                torch.tensor(
                        J_regressor_custom,
                        dtype=torch.float32
                    )
            )

        self.update_root = update_root

    def forward(self, **kwargs) -> SKELOutput:  # type: ignore
        ''' Map the order of joints of SKEL to SMPL's. '''

        if 'trans' not in kwargs.keys():
            kwargs['trans'] = kwargs['poses'].new_zeros((kwargs['poses'].shape[0], 3))  # (B, 3)

        skel_output = super(SKELWrapper, self).forward(**kwargs)
        verts = skel_output.skin_verts  # (B, 6890, 3)
        joints = skel_output.joints.clone()  # (B, 24, 3)

        # Update the root joint position (to avoid the root too forward).
        if self.update_root:
            # make root 0 to plane 11-1-6
            hips_middle = (joints[:, 1] + joints[:, 6]) / 2  # (B, 3)
            lumbar2middle = (hips_middle - joints[:, 11])  # (B, 3)
            lumbar2middle_unit = lumbar2middle / torch.norm(lumbar2middle, dim=1, keepdim=True)  # (B, 3)
            lumbar2root = joints[:, 0] - joints[:, 11]
            lumbar2root_proj = \
                torch.einsum('bc,bc->b', lumbar2root, lumbar2middle_unit)[:, None] *\
                lumbar2middle_unit  # (B, 3)
            root2root_proj = lumbar2root_proj - lumbar2root  # (B, 3)
            joints[:, 0] += root2root_proj * 0.7

        # Combine the joints from three parts:
        if self.custom_regress_joints:
            # 1.x. Regress joints from the skin vertices using SMPL's regressor.
            joints = vertices2joints(self.J_regressor_custom, verts)  # (B, 24, 3)
        else:
            # 1.y. Map selected joints of interests from SKEL to SMPL.
            joints = joints[:, self.J_skel_to_smpl]  # (B, 24, 3)
        joints_custom = joints.clone()
        # 2. Concat joints selected from skin vertices.
        joints = self.vertex_joint_selector(verts, joints)  # (B, 45, 3)
        # 3. Map selected joints to OpenPose.
        joints = joints[:, self.J_smpl_to_openpose]  # (B, 25, 3)
        # 4. Add extra joints from the J_regressor_extra.
        joints_extra = vertices2joints(self.J_regressor_extra, verts)  # (B, 19, 3)
        joints = torch.cat([joints, joints_extra], dim=1)  # (B, 44, 3)

        # Update the joints in the output.
        skel_output.joints_backup = skel_output.joints
        skel_output.joints_custom = joints_custom
        skel_output.joints = joints

        return skel_output


    @staticmethod
    def get_static_root_offset(skel_output):
        '''
        Background:
            By default, the orientation rotation is always around the original skel_root.
            In order to make the orientation rotation around the custom_root, we need to calculate the translation offset.
        This function calculates the translation offset in static pose. (From custom_root to skel_root.)
        '''
        custom_root = skel_output.joints_custom[:, 0]  # (B, 3)
        skel_root = skel_output.joints_backup[:, 0]  # (B, 3)
        offset = skel_root - custom_root  # (B, 3)
        return offset