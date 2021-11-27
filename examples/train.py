from comet_ml import Experiment
import argparse
from autolab_core import YamlConfig
import sys, os
import numpy as np
import torch
import ipdb

sys.path.append('..')
sys.path.append('.')
sys.path.append('./isaacgym_utils')

from isaacgym_utils.snake_environment import SnakeEnv
from utils import Logger

PI = np.pi

class HyperParameters():

    def __init__(self, num_envs=1024, init_policy=0, episodes=1000, episode_length=400,
                 learning_rate=0.02, best_directions=128, noise=0.03, curilearn=0, eval_episode=3):
        self.num_envs = num_envs
        self.episodes = episodes
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.best_directions = best_directions
        assert self.best_directions <= self.num_envs
        self.noise = noise
        self.init_policy = init_policy
        self.curilearn = curilearn
        self.eval_episode = eval_episode
        
class Policy():

    def __init__(self, input_size, output_size, init_policy):
        try:
            print("Training from guided policy")
            # init_policy = path_to_init_policy
            self.theta = np.load(init_policy)
            print(self.theta)
        except:
            if (init_policy):
                print("Training from random policy")
                self.theta = np.random.randn(output_size, input_size)
            else:
                print("Training from zero policy")
                self.theta = np.zeros((output_size, input_size))

        print("Starting policy theta=", self.theta)

    def get_policy(self, deltas, mode, hp):
        if mode == "evaluate":
            return torch.from_numpy(self.theta).repeat(hp.num_envs, 1, 1)
        elif mode == "explore":
            perturb = torch.cat((hp.noise*deltas, -hp.noise*deltas))
            return torch.from_numpy(self.theta).repeat(hp.num_envs, 1, 1) + perturb

    def sample_deltas(self):
        return torch.randn((hp.num_envs/2), self.theta.shape[0], self.theta.shape[0])

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, direction in rollouts:
            step += (float(r_pos) - float(r_neg)) * direction.numpy()
        self.theta += hp.learning_rate / (hp.best_directions * sigma_r) * step

def run_episode(env, policy, mode, deltas, hp):
    state = env.reset()
    done = False
    num_plays = 0
    sum_rewards = torch.tensor([0]*hp.num_envs)
    while num_plays < hp.episode_length:
        policy = policy.get_policy(deltas, mode, hp)
        action = torch.matmul(policy, state).reshape(hp.num_envs, 1, -1)
        state, reward, done, _ = env.step(action)
        sum_rewards += reward
        # ipdb.set_trace()
        num_plays += 1
    return sum_rewards

# Training the AI
def train(env, policy, hp):
    best_return = -1e5

    for episode in range(hp.episodes):
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()

        rewards = run_episode(env, policy, "explore", deltas, hp)
        positive_rewards, negative_rewards = torch.split(rewards, hp.num_envs)

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(float(r_pos), float(r_neg))
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:int(hp.best_directions)]
        rollouts = [(float(positive_rewards[k]), float(negative_rewards[k]), deltas[k]) for k in order]

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        best_rewards = np.array([x[0] for x in rollouts] + [x[1] for x in rollouts])
        sigma_r = float(best_rewards.std())  # Standard deviation of only rewards in the best directions is what it should be
        # Updating our policy
        policy.update(rollouts, sigma_r)
        print("episode", episode)

        if episode % hp.eval_episode == 0:
            # print("evaluating policy this episode")
            reward = run_episode(env, policy, "evaluate", None, hp)
            reward = float(reward.sum())/hp.num_envs
            if (reward > best_return):
                best_return = reward
            logger.update(reward, policy.theta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--init_policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument('--cfg', help='config file', type=str, default=os.path.abspath(os.getcwd())+'/../cfg/snake.yml')
    parser.add_argument('--msg', help='msg to save in a text file', type=str, default='')
    args = parser.parse_args()

    cfg = YamlConfig(args.cfg)
    hp = HyperParameters()
    logger = Logger(cfg)
    env = SnakeEnv(cfg)
    np.random.seed(123)

    # args.init_policy = './initial_policies/' + args.policy

    state_dim = cfg['training']['state_dim']
    action_dim = cfg['training']['action_dim']
    hp.num_envs = cfg['training']['num_envs']
    hp.episodes = cfg['training']['n_episodes']
    hp.episode_length = cfg['training']['episode_length']
    hp.learning_rate = cfg['training']['learning_rate']
    hp.noise = cfg['training']['noise']
    hp.best_directions = int(hp.num_envs * cfg['training']['directions'])
    hp.curilearn = cfg['training']['curi_learn']
    hp.eval_episode = cfg['training']['eval_ep']
    hp.init_policy = args.init_policy

    # args.msg to Logger

    policy = Policy(state_dim, action_dim, hp.init_policy)
    # print("start training")
    train(env, policy, hp)