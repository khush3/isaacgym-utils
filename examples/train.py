import argparse
from autolab_core import YamlConfig
import sys, os
import ipdb

sys.path.append('..')
sys.path.append('.')
sys.path.append('./isaacgym_utils')

from isaacgym_utils.policy_generic import SnakeRandomExploration
from isaacgym_utils.snake_environment import SnakeEnv
import numpy as np


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', '-c', type=str, default='cfg/snake.yml')
#     args = parser.parse_args()
#     cfg = YamlConfig(args.cfg)

#     policy = SnakeRandomExploration(n_envs=cfg['scene']['n_envs'])
#     envs = SnakeEnv(cfg)

#     for ep in range(cfg['training']['n_episodes']):
#         observations = envs.reset()
#         rets = np.array([0.] * cfg['scene']['n_envs'])
#         for _ in range(cfg['training']['episode_length']):
#             # actions = policy(observations)
#             actions = [np.array([0.6, 0.6, 0.0, 0.0, 0.5, 2, np.pi/4])]
#             observations, rewards, dones, _ = envs.step(actions)
#             rets += np.array(rewards)
#         policy.update(rets)    
#         print(f'Episode {ep}, Return: {rets}, Best Ret: {policy.best_return}, best coeffs: {policy.theta}')

##############################################################

import multiprocessing as mp
from multiprocessing import Pipe
import math

PI = math.pi

class HyperParameters():
    """
    This class is basically a struct that contains all the hyperparameters that you want to tune
    """

    def __init__(self, init_policy=0, msg='', nb_steps=10000, episode_length=400, learning_rate=0.02,
                 nb_directions=16, nb_best_directions=8, noise=0.03, seed=1, curilearn=60, evalstep=3):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise
        self.seed = seed
        self.init_policy = init_policy
        self.msg = msg
        self.curilearn = curilearn
        self.evalstep = evalstep
        self.domain_Rand = 1
        self.logdir = ""
        self.anti_clock_ori = True

    def to_text(self, path):
        res_str = ''
        res_str = res_str + 'learning_rate: ' + str(self.learning_rate) + '\n'
        res_str = res_str + 'noise: ' + str(self.noise) + '\n'
        res_str = res_str + 'env_name: ' + str(self.env_name) + '\n'
        res_str = res_str + 'episode_length: ' + str(self.episode_length) + '\n'
        res_str = res_str + 'direction ratio: ' + str(self.nb_directions / self.nb_best_directions) + '\n'
        res_str = res_str + 'Normal initialization: ' + str(self.normal) + '\n'
        res_str = res_str + 'Gait: ' + str(self.gait) + '\n'
        res_str = res_str + 'Incline_Orientaion_Anti-Clockwise: ' + str(self.anti_clock_ori) + '\n'
        res_str = res_str + 'domain_Randomization: ' + str(self.domain_Rand) + '\n'
        res_str = res_str + 'Curriculmn introduced at iteration: ' + str(self.curilearn) + '\n'
        res_str = res_str + self.msg + '\n'
        fileobj = open(path, 'w')
        fileobj.write(res_str)
        fileobj.close()

# Multiprocess Exploring the policy on one specific direction and over one episode

_RESET = 1
_CLOSE = 2
_EXPLORE = 3

def ExploreWorker(rank, childPipe, args):
    env = SnakeEnv(cfg)
    n = 0
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not childPipe.poll(0.001):
                continue
            message, payload = childPipe.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if message == _RESET:
            childPipe.send(["reset ok"])
            continue
        if message == _EXPLORE:
            policy = payload[0]
            hp = payload[1]
            direction = payload[2]
            delta = payload[3]
            state = env.reset()
            # ipdb.set_trace()
            done = False
            num_plays = 0
            sum_rewards = 0
            while num_plays < hp.episode_length:
                action = policy.evaluate(state[0], delta, direction, hp)
                state, reward, done, _ = env.step(action)
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards, num_plays])
            continue
        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()

# Building the AI

class Policy():

    def __init__(self, input_size, output_size, init_policy, args):
        try:
            print("Training from guided policy")
            self.theta = np.load(args.init_policy)
            print(self.theta)
        except:
            if (init_policy):
                print("Training from random policy")
                self.theta = np.random.randn(output_size, input_size)
            else:
                print("Training from zero policy")
                self.theta = np.zeros((output_size, input_size))                

        print("Starting policy theta=", self.theta)

    def evaluate(self, input, delta, direction, hp):
        if direction is None:
            p = self.theta
            return p.dot(input).reshape(1,-1)
        elif direction == "positive":
            p = self.theta + hp.noise * delta
            return p.dot(input).reshape(1,-1)
        else:
            p = self.theta - hp.noise * delta
            return p.dot(input).reshape(1,-1)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r, args):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, direction in rollouts:
            step += (r_pos - r_neg) * direction
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Exploring the policy on one specific direction and over one episode

def explore(env, policy, direction, delta, hp):
    state = env.reset()
    done = False
    num_plays = 0
    sum_rewards = 0
    while num_plays < hp.episode_length:
        action = policy.evaluate(state[0], delta, direction, hp)
        state, reward, done, _ = env.step(action)
        sum_rewards += reward[0]
        # ipdb.set_trace()
        num_plays += 1
    return sum_rewards

# Training the AI
def train(env, policy, hp, parentPipes, args):
    # args.logdir = "experiments/" + args.logdir
    # logger = DataLog()
    total_steps = 0
    best_return = -99999999

    # Logging, saving data
    # working_dir = os.getcwd()
    # if os.path.isdir(args.logdir) == False:
    #     os.mkdir(args.logdir)
    # previous_dir = os.getcwd()
    # os.chdir(args.logdir)
    # if os.path.isdir('iterations') == False: os.mkdir('iterations')
    # if os.path.isdir('logs') == False: os.mkdir('logs')
    # hp.to_text('hyperparameters')
    # log_dir = os.getcwd()
    # os.chdir(working_dir)

    for step in range(hp.nb_steps):
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        if (parentPipes):
            process_count = len(parentPipes)
        if parentPipes:
            p = 0
            while (p < hp.nb_directions):
                temp_p = p
                n_left = hp.nb_directions - p  # Number of processes required to complete the search
                for k in range(min([process_count, n_left])):
                    parentPipe = parentPipes[k]
                    parentPipe.send([_EXPLORE, [policy, hp, "positive", deltas[temp_p]]])
                    temp_p = temp_p + 1
                temp_p = p
                for k in range(min([process_count, n_left])):
                    positive_rewards[temp_p], step_count = parentPipes[k].recv()
                    total_steps = total_steps + step_count
                    temp_p = temp_p + 1
                temp_p = p

                for k in range(min([process_count, n_left])):
                    parentPipe = parentPipes[k]
                    parentPipe.send([_EXPLORE, [policy, hp, "negative", deltas[temp_p]]])
                    temp_p = temp_p + 1
                temp_p = p

                for k in range(min([process_count, n_left])):
                    negative_rewards[temp_p], step_count = parentPipes[k].recv()
                    total_steps = total_steps + step_count
                    temp_p = temp_p + 1
                p = p + process_count
                print('total steps till now: ', total_steps, 'Processes done: ', p)

        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(env, policy, "positive", deltas[k], hp)

            # Getting the negative rewards in the negative/opposite directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(env, policy, "negative", deltas[k], hp)

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:int(hp.nb_best_directions)]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array([x[0] for x in rollouts] + [x[1] for x in rollouts])
        sigma_r = all_rewards.std()  # Standard deviation of only rewards in the best directions is what it should be
        # Updating our policy
        policy.update(rollouts, sigma_r, args)
        print("step", step)

        if step % hp.evalstep == 0:
            # reward_evaluation = policyevaluation(env, policy, hp)
            print("eval step")
            reward = explore(env, policy, None, None, hp)
            # logger.log_kv('steps', step)
            # logger.log_kv('return', reward)
            if (reward > best_return):
                best_policy = policy.theta
                best_return = reward
                # np.save(log_dir + "/iterations/best_policy.npy", best_policy)
                print("Best Policy:", best_policy)
            print('Step:', step, 'Reward:', reward)
            # policy_path = log_dir + "/iterations/" + "policy_" + str(step)
            # np.save(policy_path, policy.theta)

            # logger.save_log(log_dir + "/logs/")
            # make_train_plots_ars(log=logger.log, keys=['steps', 'return'], save_loc=log_dir + "/logs/")


# Running the main code
# def mkdir(base, name):
#     path = os.path.join(base, name)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--init_policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument('--cfg', help='config file', type=str, default='cfg/snake.yml')
    parser.add_argument('--init_policy', help='initial policy (zero matrix or random)', type=int, default=1)
    parser.add_argument('--logdir', help='Directory root to log policy files (npy)', type=str, default='logdir_name')
    parser.add_argument('--episode_length', help='length of each episode', type=int, default=10)
    parser.add_argument('--curi_learn', help='after how many iteration steps second stage of curriculum learning should start', type=int, default=0) # default 10, removing curriculum learning
    parser.add_argument('--eval_step', help='policy evaluation after how many steps should take place', type=int, default=3)
    parser.add_argument('--msg', help='msg to save in a text file', type=str, default='')
    args = parser.parse_args()

    cfg = YamlConfig(args.cfg)
    state_dim = cfg['training']['state_dim']
    action_dim = cfg['training']['action_dim']

    hp = HyperParameters()
    # args.init_policy = './initial_policies/' + args.policy
    hp.msg = args.msg
    env = SnakeEnv(cfg)
    hp.seed = 123
    hp.nb_steps = cfg['training']['n_episodes']
    hp.episode_length = cfg['training']['episode_length']
    hp.learning_rate = cfg['training']['learning_rate']
    hp.noise = cfg['training']['noise']
    hp.nb_directions = state_dim * action_dim
    hp.nb_best_directions = int(hp.nb_directions / cfg['training']['directions'])
    hp.init_policy = args.init_policy
    # hp.curilearn = args.curi_learn
    hp.evalstep = args.eval_step
    # print("log dir", args.logdir)
    hp.logdir = args.logdir
    np.random.seed(hp.seed)
    max_processes = cfg['training']['max_processes']
    parentPipes = None
    if cfg['training']['multiprocessing']:
        num_processes = min([hp.nb_directions, max_processes])
        print('processes: ', num_processes)
        processes = []
        childPipes = []
        parentPipes = []

        for pr in range(num_processes):
            parentPipe, childPipe = Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(num_processes):
            p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank], args))
            p.start()
            processes.append(p)

    policy = Policy(state_dim, action_dim, hp.init_policy, args)
    print("start training")

    train(env, policy, hp, parentPipes, args)

    if cfg['training']['multiprocessing']:
        for parentPipe in parentPipes:
            parentPipe.send([_CLOSE, "pay2"])

        for p in processes:
            p.join()