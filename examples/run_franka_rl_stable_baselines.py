'''
This script doesn't learn any useful policies.
It's an example of how to interface with stable baselines.
'''
import argparse
from autolab_core import YamlConfig

from carbongym_utils.rl.stable_baselines import GymFrankaBlockVecEnvStableBaselines
from carbongym_utils.draw import draw_transforms

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_rl_stable_baselines.yaml')
    parser.add_argument('--logdir', '-l', type=str, default='outs/tb')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)
    
    vec_env = GymFrankaBlockVecEnvStableBaselines(cfg)

    def custom_draws(scene):
        franka = scene.get_asset('franka0')
        for env_ptr in scene.env_ptrs:
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform])

    def learn_cb(local_vars, global_vars):        
        vec_env.render(custom_draws=custom_draws)

    model = PPO2(MlpPolicy, env=vec_env, verbose=1, tensorboard_log=args.logdir, **cfg['rl']['ppo'])
    model.learn(total_timesteps=cfg['rl']['total_timesteps'], callback=learn_cb, reset_num_timesteps=False)
