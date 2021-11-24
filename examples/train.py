import argparse

from autolab_core import YamlConfig
import sys

sys.path.append('..')
sys.path.append('.')
sys.path.append('./isaacgym_utils')

from isaacgym_utils.policy_generic import SnakeRandomExploration
from isaacgym_utils.snake_environment import SnakeEnv
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/snake.yml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    policy = SnakeRandomExploration(n_envs=cfg['scene']['n_envs'])
    envs = SnakeEnv(cfg)

    for ep in range(cfg['training']['n_episodes']):
        observations = envs.reset()
        rets = np.array([0.] * cfg['scene']['n_envs'])
        for _ in range(cfg['training']['episode_len']):
            actions = policy(observations)
            observations, rewards, dones, _ = envs.step(actions)
            rets += np.array(rewards)
        policy.update(rets)    
        print(f'Episode {ep}, Return: {rets}, Best Ret: {policy.best_return}, best coeffs: {policy.theta}')
