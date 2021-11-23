import numpy as np
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymSnake, GymBoxAsset
from isaacgym_utils.draw import draw_transforms


class SnakeEnv():
    def __init__(self, cfg):
        self.n_envs = cfg['scene']['n_envs']
        self.dt = cfg['scene']['gym']['dt']
        self._cts = cfg.get('cts', False)
        self.time_horizon = None
        self.cfg = cfg
        self._name = 'snake'
        self.env_idxs = range(self.n_envs)

        self.scene = GymScene(cfg['scene'])
    
        if self.cfg['table'] is not None:
            self.table = GymBoxAsset(self.scene, **cfg['table']['dims'], 
                                shape_props=cfg['table']['shape_props'], 
                                asset_options=cfg['table']['asset_options']
                                )
            self.table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))

        self.snake = GymSnake(cfg['snake'], self.scene)

        self.snake_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    
        self.scene.setup_all_envs(self.setup)

        self.init_rb_transform = self.snake.get_rb_transforms(0, self._name)


    def reset(self):
        self.t_sim = 0
        for i in self.env_idxs: self.snake.reset(i, self._name, self.init_rb_transform)
        observations = self.get_observation()
        return observations

    def step(self, actions):
        self.t_sim += self.dt

        for env_idx in self.env_idxs:
            target_angles = self.snake.controller(actions[env_idx], self.t_sim)
            self.snake.set_joints_targets(env_idx, self._name, target_angles)

        self.scene.step()
        self.scene.render(custom_draws=self.custom_draws)
        observations = self.get_observation()
        rewards = self.get_reward()
        dones = self.termination()

        return observations, rewards, dones, None

    def get_observation(self):
        return [np.concatenate([self.get_dof_pose(i), self.get_dof_vel(i), self.get_base_info(i)]) for i in self.env_idxs]

    def get_reward(self):
        return [self.get_base_info(i)[1] for i in self.env_idxs] # Incentivize side movement 

    def termination(self):
        return np.zeros(len(self.env_idxs))

    def get_base_info(self, i):
        return self.snake.get_rb_poses_as_np_array(i, 'snake')[1]

    def get_dof_pose(self, i):
        return self.snake.get_dof_states(i, 'snake')['pos']

    def get_dof_vel(self, i):
        return self.snake.get_dof_states(i, 'snake')['vel']

    def custom_draws(self, scene):
        draw_transforms(scene, scene.env_idxs, [self.snake_transform], length=0.2)


    def setup(self, scene, _):
        if self.cfg['table'] is not None: scene.add_asset('table', self.table, self.table_transform)
        scene.add_asset('snake', self.snake, self.snake_transform, collision_filter=0) # avoid self-collision
