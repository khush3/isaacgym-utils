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

    policy = SnakeGait('sea-snake', 'sidewinding')
    envs = SnakeEnv(cfg)

    observations = envs.reset()

    for _ in range(5000):
        actions = [policy(i) for i in observations]
        observations, rewards, dones, _ = envs.step(actions)