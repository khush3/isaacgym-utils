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
        observations = [self.t_sim for _ in range(self.n_envs)]
        return observations


    def step(self, actions):
        self.t_sim = self.t_step * self.dt

        if self.time_horizon is not None and self.t_step >= self.time_horizon:
            print("Reached time horizon. Reseting!")

        for env_idx in self.env_idxs:
            self.snake.set_joints_targets(env_idx, self._name, actions[env_idx])

        self.scene.step()
        self.scene.render(custom_draws=self.custom_draws)
        observation, reward, dones = self.get_ord()

        self.t_step += 1

        return observation, reward, dones, None


    def get_ord(self):
        observations = [self.t_sim for _ in range(self.cfg['scene']['n_envs'])]
        rewards, dones = None, None
        dofs = np.array([self.snake.get_dof_states(i, 'snake') for i in self.env_idxs])
        return observations, rewards, dones


    def custom_draws(self, scene):
        draw_transforms(scene, scene.env_idxs, [self.snake_transform], length=0.2)


    def setup(self, scene, _):
        if self.cfg['table'] is not None: scene.add_asset('table', self.table, self.table_transform)
        scene.add_asset('snake', self.snake, self.snake_transform, collision_filter=0) # avoid self-collision
