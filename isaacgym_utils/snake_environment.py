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


    def reset(self):
        self.t_step = 0
        self.t_sim = self.t_step * self.dt
        time_steps = [self.t_sim for _ in range(self.n_envs)]
        observations = [np.concatenate([self.get_dof_pose(i), self.get_dof_vel(i), self.get_base_info(i)]) for i in self.env_idxs]
        self.observations_prev = observations
        self.base_prev = [self.snake.get_rb_poses_as_np_array(i, 'snake')[1] for i in self.env_idxs]
        return observations, time_steps


    def step(self, actions):
        self.t_sim = self.t_step * self.dt

        if self.time_horizon is not None and self.t_step >= self.time_horizon:
            print("Reached time horizon. Reseting!")

        for env_idx in self.env_idxs:
            self.snake.set_joints_targets(env_idx, self._name, actions[env_idx])

        self.scene.step()
        self.scene.render(custom_draws=self.custom_draws)
        observation, reward, dones, time_steps = self.get_ord()

        self.t_step += 1

        return observation, reward, dones, None, time_steps


    def get_ord(self):
        time_steps = [self.t_sim for _ in range(self.cfg['scene']['n_envs'])]
        observations = [np.concatenate([self.get_dof_pose(i), self.get_dof_vel(i), self.get_base_info(i)]) for i in self.env_idxs]
        rewards = [self.get_base_info(i) for i in self.env_idxs]#[p[0] - p_[0] for p, p_ in zip(base_curr, self.base_prev)]
        dones = np.zeros(len(self.env_idxs))
        # observations = np.array([[self.snake.get_dof_states(i, 'snake')['pos'], self.snake.get_dof_states(i, 'snake')['vel']] for i in self.env_idxs])
        # rewards = [p[0] - p_[0] for p, p_ in zip(base_curr, self.base_prev)]
        # self.base_prev = base_curr
        # rewards = [(obs[:][0] - obs_prev[:][0]).mean() for obs, obs_prev in zip(observations, self.observations_prev)]
        return observations, rewards, dones, time_steps

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
