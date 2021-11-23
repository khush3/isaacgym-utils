import argparse

from autolab_core import YamlConfig
import sys

sys.path.append('..')
sys.path.append('.')
sys.path.append('./isaacgym_utils')

from isaacgym_utils.policy_generic import SnakeGait
from isaacgym_utils.snake_environment import SnakeEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/snake.yml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    policy = SnakeRandomExploration('sea-snake', 'sidewinding')
    envs = SnakeEnv(cfg)

    for ep in range(cfg['scene']['gym']['n_episodes']):
        policy.new_episode()
        observations, time_steps = envs.reset()
        ret = 0
        for _ in range(cfg['scene']['gym']['episode_len']):
            actions = [policy(i) for i in time_steps]
            observations, rewards, dones, _, time_steps = envs.step(actions)
            ret += rewards[0]
        policy.update(ret)    
        print(f'Episode {ep}, Return: {ret}')
