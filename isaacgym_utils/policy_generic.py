from abc import ABC, abstractmethod
import numpy as np

from .math_utils import min_jerk, slerp_quat, vec3_to_np, np_to_vec3, \
                    project_to_line, compute_task_space_impedance_control


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


class SnakeRandomExploration(Policy):

    def __init__(self, perturbation_factor=0.3, init='best_action', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.02
        self.best_directions = 1
        self.perturbation_factor = perturbation_factor
        if init=='random': 
            print("Training from random policy")
            self.theta = np.random.rand(7)
        elif init=='best_action':
            self.theta = np.array([0, 0.7, 0, 0, .5, 4, 0]) # Linear progression
        elif init=='zero':
            print("Training from zero matrix policy")
        self.best_return = 0
        self.new_theta = self.theta + self.perturbate

    def __call__(self, observation):
        return self.new_theta

    def update(self, ret):
        if ret > self.best_return:
            self.theta = self.new_theta
            self.best_return = ret
        self.new_theta = self.theta + self.perturbate

    @property
    def perturbate(self):
        return np.random.rand(7) * self.perturbation_factor
