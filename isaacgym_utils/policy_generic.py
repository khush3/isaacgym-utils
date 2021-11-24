from abc import ABC, abstractmethod
import numpy as np


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

    def __init__(self, n_envs, perturbation_factor=0.08, init='best_action', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.02
        self.best_directions = 1
        self.envs = range(n_envs)
        self.perturbation_factor = perturbation_factor
        if init=='random': 
            print("Training from random policy")
            self.theta = np.random.rand(7)
        elif init=='best_action':
            # self.theta = np.array([0.21772525, 0.56285663, 0.25177039, 0.0944957, 0.03984703, 0.44255845, 0.36545371]) # Linear progression
            self.theta = np.array([0.75, 0., 0., 0.25, 4, 2, np.pi/2]) # Linear progression
        elif init=='zero':
            print("Training from zero matrix policy")
            self.theta = np.zeros(7)
        self.best_return = -1e10
        # self.perturbate()

    def __call__(self, observation):
        return self.new_thetas

    def update(self, rets):
        for env, ret in zip(self.envs, rets):
            if ret > self.best_return:
                print('Yo updated')
                self.theta = self.new_thetas[env]
                self.best_return = ret
        self.perturbate()

    def perturbate(self):
        self.new_thetas = [self.theta + np.random.rand(7) * self.perturbation_factor for _ in self.envs]