import numpy as np

from carbongym import gymapi
from carbongym_utils.constants import CARBONGYM_ASSETS_PATH
from carbongym_utils.math_utils import transform_to_RigidTransform, vec3_to_np, quat_to_rot, np_to_vec3

from .assets import GymURDFAsset
from time import time


class GymFranka(GymURDFAsset):

    INIT_JOINTS = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04])
    _LOWER_LIMITS = None
    _UPPER_LIMITS = None
    _VEL_LIMITS = None

    _URDF_PATH = 'urdf/franka_description/robots/franka_panda.urdf'
    _URDF_PATH_WITH_DYNAMICS = 'urdf/franka_description/robots/franka_panda_dynamics.urdf'

    @staticmethod
    def _key(env_index, name):
        return (env_index, name)

    def __init__(self, cfg, *args, actuation_mode='joints'):
        if 'urdf' in cfg:
            urdf_path = cfg['urdf']
            assets_root = cfg['assets_root']
        else:
            urdf_path = GymFranka._URDF_PATH_WITH_DYNAMICS
            assets_root = CARBONGYM_ASSETS_PATH
        super().__init__(urdf_path, *args, 
                        shape_props=cfg['shape_props'], 
                        dof_props=cfg['dof_props'], 
                        asset_options=cfg['asset_options'],
                        assets_root=assets_root
                        )

        self._use_custom_ee = False
        if 'custom_ee_rb_name' in cfg:
            self._use_custom_ee = True
            self._custom_ee_rb_name = cfg['custom_ee_rb_name']
        
        self._ee_tool_offset = gymapi.Transform()
        if 'custom_ee_offset' in cfg:
            self._ee_tool_offset = gymapi.Transform((np_to_vec3(cfg['custom_ee_offset'])))
        
        self._gripper_offset = gymapi.Transform(gymapi.Vec3(0, 0, 0.1034))
        self._finger_offset = gymapi.Transform(gymapi.Vec3(0, 0, 0.045))

        self._actuation_mode = actuation_mode
        self._attractor_handles_map = {}
        self._attractor_transforms_map = {}

        self._attractor_stiffness = cfg['attractor_props']['stiffness']
        self._attractor_damping = cfg['attractor_props']['damping']

    def set_gripper_width_target(self, env_ptr, ah, width):
        joints_targets = self.get_joints_targets(env_ptr, ah)
        joints_targets[-2:] = width
        self.set_joints_targets(env_ptr, ah, joints_targets)

    def open_grippers(self, env_ptr, ah):
        self.set_gripper_width_target(env_ptr, ah, 0.04)

    def close_grippers(self, env_ptr, ah):
        self.set_gripper_width_target(env_ptr, ah, 0)

    def set_gripper_width(self, env_ptr, ah, width):
        width = np.clip(width, self._LOWER_LIMITS[-1], self._UPPER_LIMITS[-1])
        self.set_gripper_width_target(env_ptr, ah, width)

        joints = self.get_joints(env_ptr, ah)
        joints[-2] = width
        self.set_joints(env_ptr, ah, joints)

    def get_gripper_width(self, env_ptr, ah):
        return self.get_joints(env_ptr, ah)[-1]

    def get_ee_transform(self, env_ptr, name, offset=True):
        bh = self._gym.get_rigid_handle(env_ptr, name, 'panda_hand')
        ee_transform = self._gym.get_rigid_transform(env_ptr, bh) 
        if offset:
            ee_transform = ee_transform * self._gripper_offset * self._ee_tool_offset
        return ee_transform

    def get_ee_rigid_transform(self, env_ptr, name, offset=True):
        return transform_to_RigidTransform(self.get_ee_transform(env_ptr, name, offset=offset), 
                                                from_frame='panda_ee', to_frame='panda_link0')

    def get_finger_transforms(self, env_ptr, name, offset=True):
        bh_lf = self._gym.get_rigid_handle(env_ptr, name, 'panda_leftfinger')
        bh_rf = self._gym.get_rigid_handle(env_ptr, name, 'panda_rightfinger')
        lf_transform = self._gym.get_rigid_transform(env_ptr, bh_lf)
        rf_transform = self._gym.get_rigid_transform(env_ptr, bh_rf)

        if offset:
            lf_transform = lf_transform * self._finger_offset
            rf_transform = rf_transform * self._finger_offset

        return lf_transform, rf_transform

    def get_desired_ee_transform(self, env_index, name):
        if self._actuation_mode != 'attractors':
            raise ValueError('Can\'t get desired ee transform when not using attractors!')

        key = self._key(env_index, name)
        return self._attractor_transforms_map[key]

    def get_left_finger_ct_forces(self, env_ptr, ah):
        all_ct_forces = self._gym.get_rigid_contact_forces(self._sim)
        rbi_lf = self._gym.get_actor_rigid_body_index(env_ptr, ah, self._rb_names_map['panda_leftfinger'], gymapi.DOMAIN_SIM)
        ct_forces_lf = np.array([all_ct_forces[rbi_lf][k] for k in 'xyz'])

        return ct_forces_lf

    def get_right_finger_ct_forces(self, env_ptr, ah):
        all_ct_forces = self._gym.get_rigid_contact_forces(self._sim)
        rbi_rf = self._gym.get_actor_rigid_body_index(env_ptr, ah, self._rb_names_map['panda_rightfinger'], gymapi.DOMAIN_SIM)
        ct_forces_rf = np.array([all_ct_forces[rbi_rf][k] for k in 'xyz'])

        return ct_forces_rf

    def get_ee_ct_forces(self, env_ptr, ah):
        if self._use_custom_ee:
            all_ct_forces = self._gym.get_rigid_contact_forces(self._sim)
            rbi = self._gym.get_actor_rigid_body_index(env_ptr, ah, self._rb_names_map[self._custom_ee_rb_name], gymapi.DOMAIN_SIM)
            ct_forces = np.array([all_ct_forces[rbi][k] for k in 'xyz'])
        else:
            ct_forces_lf = self.get_left_finger_ct_forces(env_ptr, ah)
            ct_forces_rf = self.get_right_finger_ct_forces(env_ptr, ah)
            ct_forces = (ct_forces_lf + ct_forces_rf) / 2

        return ct_forces

    @property
    def joint_limits_lower(self):
        return self._LOWER_LIMITS
        
    @property
    def joint_limits_upper(self):
        return self._UPPER_LIMITS

    @property
    def joint_max_velocities(self):
        return self._VEL_LIMITS

    def set_actuation_mode(self, mode, env_index, env_ptr, name, ah):
        self._actuation_mode = mode
        if self._actuation_mode == 'attractors':
            self.set_dof_props(env_ptr, ah, {
                'driveMode': [gymapi.DOF_MODE_NONE] * 7 + [gymapi.DOF_MODE_POS] * 2
            })

            key = self._key(env_index, name)
            if key not in self._attractor_handles_map:
                attractor_props = gymapi.AttractorProperties()
                attractor_props.stiffness = self._attractor_stiffness
                attractor_props.damping= self._attractor_damping
                attractor_props.axes = gymapi.AXIS_ALL

                gripper_handle = self._gym.get_rigid_handle(env_ptr, name, 'panda_hand')
                attractor_props.rigid_handle = gripper_handle
                attractor_props.offset = self._gripper_offset * self._ee_tool_offset

                attractor_handle = self._gym.create_rigid_body_attractor(env_ptr, attractor_props)
                self._attractor_handles_map[key] = attractor_handle

            gripper_transform = self.get_ee_transform(env_ptr, name)
            self.set_ee_transform(env_ptr, env_index, name, gripper_transform)
        elif self._actuation_mode == 'joints':
            self.set_dof_props(env_ptr, ah, {
                'driveMode': [gymapi.DOF_MODE_POS] * 9
            })
        elif self._actuation_mode == 'torques':
            self.set_dof_props(env_ptr, ah, {
                'driveMode': [gymapi.DOF_MODE_EFFORT] * 7 + [gymapi.DOF_MODE_POS] * 2
            })
        else:
            raise ValueError('Unknown actuation mode! Must be attractors, joints, or torques!')

    def post_create_actor(self, env_index, env_ptr, name, ah):
        super().post_create_actor(env_index, env_ptr, name, ah)
        self.set_joints(env_ptr, ah, self.INIT_JOINTS)
        self.set_joints_targets(env_ptr, ah, self.INIT_JOINTS)

        if self._LOWER_LIMITS is None or self._UPPER_LIMITS is None or self._VEL_LIMITS is None:
            dof_props = self._gym.get_actor_dof_properties(env_ptr, ah)
            self._LOWER_LIMITS = dof_props['lower']
            self._UPPER_LIMITS = dof_props['upper']
            self._VEL_LIMITS = dof_props['velocity']

        self.set_actuation_mode(self._actuation_mode, env_index, env_ptr, name, ah)

    def set_attractor_props(self, env_index, env_ptr, name, props):
        if self._actuation_mode != 'attractors':
            raise ValueError('Not using attractors!')

        key = self._key(env_index, name)
        ath = self._attractor_handles_map[key]
        attractor_props = self._gym.get_attractor_properties(env_ptr, ath)

        for key, val in props.items():
            setattr(attractor_props, key, val)
        
        self._gym.set_attractor_properties(env_ptr, ath, attractor_props)

    def set_ee_transform(self, env_ptr, env_index, name, transform):
        if self._actuation_mode != 'attractors':
            raise ValueError('Can\'t set ee transform when not using attractors!')

        key = self._key(env_index, name)
        attractor_handle = self._attractor_handles_map[key]

        self._attractor_transforms_map[key] = transform

        self._gym.set_attractor_target(env_ptr, attractor_handle, transform)

    def set_delta_ee_transform(self, env_ptr, env_index, name, transform):
        ''' This performs delta translation in the global frame and
            delta rotation in the end-effector frame.
        '''
        current_transform = self.get_ee_transform(env_ptr, name)
        desired_transform = gymapi.Transform(p=current_transform.p, r=current_transform.r)
        desired_transform.p = desired_transform.p + transform.p
        desired_transform.r = transform.r * desired_transform.r

        self.set_ee_transform(env_ptr, env_index, name, desired_transform)

    def apply_torque(self, env_ptr, ah, tau):
        if len(tau) == 7:
            tau = np.concatenate([tau, np.zeros(2)])

        self.apply_actor_dof_efforts(env_ptr, ah, tau)

    def apply_actions(self, env_ptr, env_index, ah, name, action_type, actions):
        if action_type == 'ee_targets':
            self.set_ee_transform(env_ptr, env_index, name, actions)
        elif action_type == 'delta_ee_targets':
            self.set_delta_ee_transform(env_ptr, env_index, name, actions)
        else:
            super().apply_actions(env_ptr, env_index, ah, name, action_type, actions)

    def get_links_transforms(self, env_ptr, name):
        transforms = []
        for i in range(1, 8):
            link_name = 'panda_link{}'.format(i)
            bh = self._gym.get_rigid_handle(env_ptr, name, link_name)
            transforms.append(self._gym.get_rigid_transform(env_ptr, bh) )

        return transforms

    def get_links_rigid_transforms(self, env_ptr, name):
        transforms = self.get_links_transforms(env_ptr, name)
        return [transform_to_RigidTransform(transform, 
                                        from_frame='panda_link{}'.format(i+1), 
                                        to_frame='panda_link0') 
                for i, transform in enumerate(transforms)]

    def get_jacobian(self, env_ptr, name):
        transforms = self.get_links_transforms(env_ptr, name)
        ee_pos = vec3_to_np(self.get_ee_transform(env_ptr, name).p)

        joints_pos = np.array([vec3_to_np(transform.p) for transform in transforms])
        axes = np.array([quat_to_rot(transform.r)[:, 2] for transform in transforms])
        J = np.r_[np.cross(axes, ee_pos - joints_pos).T, axes.T]

        return J

    def reset_joints(self, env_ptr, ah):
        self.set_joints(env_ptr, ah, self.INIT_JOINTS)
    
