import argparse

from autolab_core import YamlConfig
import sys

sys.path.append('..')
sys.path.append('.')
sys.path.append('./isaacgym_utils')
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymSnake, GymBoxAsset
from isaacgym_utils.policy import RandomDeltaJointPolicy
from isaacgym_utils.draw import draw_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/snake.yml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    
    # table = GymBoxAsset(scene, **cfg['table']['dims'], 
    #                     shape_props=cfg['table']['shape_props'], 
    #                     asset_options=cfg['table']['asset_options']
    #                     )
    snake = GymSnake(cfg['snake'], scene)

    # table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    snake_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    
    def setup(scene, _):
        # scene.add_asset('table', table, table_transform)
        scene.add_asset('snake', snake, snake_transform, collision_filter=1) # avoid self-collision
    scene.setup_all_envs(setup)

    def custom_draws(scene):
        draw_transforms(scene, scene.env_idxs, [snake_transform], length=0.2)

    policy = RandomDeltaJointPolicy(snake, 'snake')
    scene.run(policy=policy, custom_draws=custom_draws)
