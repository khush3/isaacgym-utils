from abc import ABC, abstractmethod
import numpy as np

from isaacgym import gymapi
from .math_utils import min_jerk, slerp_quat, vec3_to_np, np_to_vec3, \
                    project_to_line, compute_task_space_impedance_control
import isaacgym_utils.snake_gait as sg

class Policy(ABC):

    def __init__(self):
        self._time_horizon = -1

    @abstractmethod
    def __call__(self, scene, env_idx, t_step, t_sim):
        pass

    def reset(self):
        pass

    @property
    def time_horizon(self):
        return self._time_horizon

SNAKES = {
    'sea-snake': 16,
    'ReU': 16
}

class SnakeRandomExploration(Policy):

    def __init__(self, snake, gait, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init = 'best_action'
        self.learning_rate = 0.02
        self.best_directions = 1
        if init=='random': 
            print("Training from random policy")
            self.theta = np.random.rand(6)
        elif init=='best_action':
            self.theta = np.array([0, 0.7, 0, 0, .5, 4, 0]) # Linear progression
        elif init=='zero':
            print("Training from zero matrix policy")
        self.numModules=16
        self.best_return = 0
        self._modules = SNAKES[snake]

    def new_episode(self):
        self.perturbations = 0 * np.random.rand(7) / 3.0
        self.new_theta = self.theta + self.perturbations

    def __call__(self, observation):
        t_sim = observation
        gaitSignal = self.controller(t_sim)
        return np.array(gaitSignal)

    def update(self, ret):
        if ret > self.best_return:
            self.theta = self.new_theta
            self.best_return = ret
        # step = np.zeros(self.theta.shape)
        # for r_pos, r_neg, direction in rollouts:
        #     step += (r_pos - r_neg) * direction
        # self.theta += self.learning_rate / (self.nb_best_directions * sigma_r) * step
        # timestr = time.strftime("%Y%m%d-%H%M%S")

    def controller(self,t):
        A_odd, A_even, beta_odd, beta_even, Ws, Wt, delta = self.new_theta
        angles = np.zeros([1, self.numModules])
        for n in range(self.numModules):
            if n%2 == 1:
                angles[0, n] = beta_odd+A_odd*np.sin(Ws*n-Wt*t+delta)
            else:
                angles[0, n] = beta_even+A_even*np.sin(Ws*n-Wt*t)
        
        signal = self.angleReversals(angles)
        return signal

    def angleReversals(self, angles):
        reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        angles = np.fliplr(angles)
        angles = np.multiply(reversals, angles)
        signal = []
        for i in range(self.numModules):
            signal.append(angles[0,i])
        
        return signal


class Policy(Policy):

    def __init__(self, snake, gait, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._robot = robot
        # self._name = name
        # self._snake = snake
        self._modules = SNAKES[snake]
        self.controller = sg.Gait(numModules=self._modules)
        self.controller.setGait(gait)

    def __call__(self, observation):
        t_sim = observation
        gaitSignal = self.controller.getSignal(t_sim)
        return np.array(gaitSignal)
        # print(gaitSignal)
        # self._robot.set_joints_targets(env_idx, self._name, np.array(gaitSignal))

    # def get_controls(self, parameters):
    #     A_odd = 0.75
    #     A_even = 0
    #     beta_odd = 0
    #     beta_even = 0.25  # for stability
    #     s = 4
    #     w = 2

    #     angles = np.zeros([1, self.numModules])
    #     for n in range(self.numModules):
    #         if n%2 == 1:
    #             angles[0, n] = beta_odd+A_odd*np.sin(n*s+w*t)
    #         else:
    #             angles[0, n] = beta_even+A_even*np.sin(n*s+w*t)
    #     # reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    #     # angles = np.fliplr(angles)
    #     # angles = np.multiply(reversals, angles)
    #     signal = []
    #     for i in range(self.numModules):
    #         signal.append(angles[0,i])
        
    #     return signal


class RandomDeltaJointPolicy(Policy):

    def __init__(self, robot, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._robot = robot
        self._name = name

    def __call__(self, scene, env_idx, _, __):
        delta_joints = np.array([5]*16)#(np.random.random(16) * 2 - 1) * ([5] * 14 + [5] * 2)
        # delta_joints = (np.random.random(self._robot.n_dofs) * 2 - 1) * ([0.05] * 7 + [0.005] * 2)
        self._robot.apply_delta_joint_targets(env_idx, self._name, delta_joints)


class GraspBlockPolicy(Policy):

    def __init__(self, robot, robot_name, block, block_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._robot = robot
        self._robot_name = robot_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._robot.get_ee_transform(env_idx, self._robot_name)

        if t_step == 0:
            self._init_ee_transforms.append(ee_transform)
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._robot, self._robot_name, ee_transform, ee_transform, T=20)
            )

        if t_step == 20:
            block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
            grasp_transform = gymapi.Transform(p=block_transform.p, r=self._init_ee_transforms[env_idx].r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p + gymapi.Vec3(0, 0, 0.2), r=grasp_transform.r)

            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._robot, self._robot_name, ee_transform, self._pre_grasp_transforms[env_idx], T=180
                )

        if t_step == 200:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._robot, self._robot_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )

        if t_step == 300:
            self._robot.close_grippers(env_idx, self._robot_name)
        
        if t_step == 400:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._robot, self._robot_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
                )

        if t_step == 500:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._robot, self._robot_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )

        if t_step == 600:
            self._robot.open_grippers(env_idx, self._robot_name)

        if t_step == 700:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._robot, self._robot_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
                )

        if t_step == 800:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._robot, self._robot_name, self._pre_grasp_transforms[env_idx], self._init_ee_transforms[env_idx], T=100
                )

        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)


class GraspPointPolicy(Policy):

    def __init__(self, robot, robot_name, grasp_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._robot = robot
        self._robot_name = robot_name
        self._grasp_transform = grasp_transform

        self._time_horizon = 710

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []

    def __call__(self, scene, env_idx, t_step, _):
        t_step = t_step % self._time_horizon

        if t_step == 0:
            self._init_joints = self._robot.get_joints(env_idx, self._robot_name)
            self._init_rbs = self._robot.get_rb_states(env_idx, self._robot_name)

        if t_step == 20:
            ee_transform = self._robot.get_ee_transform(env_idx, self._robot_name)
            self._init_ee_transforms.append(ee_transform)

            pre_grasp_transfrom = gymapi.Transform(p=self._grasp_transform.p, r=self._grasp_transform.r)
            pre_grasp_transfrom.p.z += 0.2

            self._grasp_transforms.append(self._grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._robot.set_ee_transform(env_idx, self._robot_name, self._pre_grasp_transforms[env_idx])

        if t_step == 100:
            self._robot.set_ee_transform(env_idx, self._robot_name, self._grasp_transforms[env_idx])

        if t_step == 150:
            self._robot.close_grippers(env_idx, self._robot_name)
        
        if t_step == 250:
            self._robot.set_ee_transform(env_idx, self._robot_name, self._pre_grasp_transforms[env_idx])

        if t_step == 350:
            self._robot.set_ee_transform(env_idx, self._robot_name, self._grasp_transforms[env_idx])

        if t_step == 500:
            self._robot.open_grippers(env_idx, self._robot_name)

        if t_step == 550:
            self._robot.set_ee_transform(env_idx, self._robot_name, self._pre_grasp_transforms[env_idx])

        if t_step == 600:
            self._robot.set_ee_transform(env_idx, self._robot_name, self._init_ee_transforms[env_idx])

        if t_step == 700:
            self._robot.set_joints(env_idx, self._robot_name, self._init_joints)
            self._robot.set_rb_states(env_idx, self._robot_name, self._init_rbs)


class robotEEImpedanceController:

    def __init__(self, robot, robot_name):
        self._robot = robot
        self._robot_name = robot_name
        self._elbow_joint = 3

        Kp_0, Kr_0 = 200, 8
        Kp_1, Kr_1 = 200, 5
        self._Ks_0 = np.diag([Kp_0] * 3 + [Kr_0] * 3)
        self._Ds_0 = np.diag([4 * np.sqrt(Kp_0)] * 3 + [2 * np.sqrt(Kr_0)] * 3)
        self._Ks_1 = np.diag([Kp_1] * 3 + [Kr_1] * 3)
        self._Ds_1 = np.diag([4 * np.sqrt(Kp_1)] * 3 + [2 * np.sqrt(Kr_1)] * 3)

    def compute_tau(self, env_idx, target_transform):
        # primary task - ee control
        ee_transform = self._robot.get_ee_transform(env_idx, self._robot_name)

        J = self._robot.get_jacobian(env_idx, self._robot_name)
        q_dot = self._robot.get_joints_velocity(env_idx, self._robot_name)[:7]
        x_vel = J @ q_dot

        tau_0 = compute_task_space_impedance_control(J, ee_transform, target_transform, x_vel, self._Ks_0, self._Ds_0)

        # secondary task - elbow straight
        link_transforms = self._robot.get_links_transforms(env_idx, self._robot_name)
        elbow_transform = link_transforms[self._elbow_joint]

        u0 = vec3_to_np(link_transforms[0].p)[:2]
        u1 = vec3_to_np(link_transforms[-1].p)[:2]
        curr_elbow_xyz = vec3_to_np(elbow_transform.p)
        goal_elbow_xy = project_to_line(curr_elbow_xyz[:2], u0, u1)
        elbow_target_transform = gymapi.Transform(
            p=gymapi.Vec3(goal_elbow_xy[0], goal_elbow_xy[1], curr_elbow_xyz[2] + 0.2),
            r=elbow_transform.r
        )

        J_elb = self._robot.get_jacobian(env_idx, self._robot_name, target_joint=self._elbow_joint)
        x_vel_elb = J_elb @ q_dot

        tau_1 = compute_task_space_impedance_control(J_elb, elbow_transform, elbow_target_transform, x_vel_elb, self._Ks_1, self._Ds_1)
        
        # nullspace projection
        JT_inv = np.linalg.pinv(J.T)
        Null = np.eye(7) - J.T @ (JT_inv)
        tau = tau_0 + Null @ tau_1

        return tau


class EEImpedanceWaypointPolicy(Policy):

    def __init__(self, robot, robot_name, init_ee_transform, goal_ee_transform, T=300):
        self._robot = robot
        self._robot_name = robot_name

        self._T = T
        self._ee_impedance_ctrlr = robotEEImpedanceController(robot, robot_name)

        init_ee_pos = vec3_to_np(init_ee_transform.p)
        goal_ee_pos = vec3_to_np(goal_ee_transform.p)
        self._traj = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(init_ee_pos, goal_ee_pos, t, self._T)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, self._T),
            )
            for t in range(self._T)
        ]

    @property
    def horizon(self):
        return self._T

    def __call__(self, scene, env_idx, t_step, t_sim):
        target_transform = self._traj[min(t_step, self._T - 1)]
        tau = self._ee_impedance_ctrlr.compute_tau(env_idx, target_transform)
        self._robot.apply_torque(env_idx, self._robot_name, tau)
