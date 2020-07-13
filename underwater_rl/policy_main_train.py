#!/usr/bin/env python3
import argparse
import logging
import math
import os
import pickle
import random
import shutil
import time
import warnings
from collections import namedtuple
from torch.distributions import Categorical
from torch.nn import Softmax
from itertools import count
from torch.autograd import Variable
import gym
from matplotlib import pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

from memory import ReplayMemory, PrioritizedReplay
from models import *
from wrappers import *
from utils import convert_images_to_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
warnings.filterwarnings("ignore", category=UserWarning)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def select_action(state):
    # state = torch.FloatTensor(state).unsqueeze(0).to(device)
    # print('state : ', state)
    #with torch.no_grad():
    p = policy_net.forward(state.to(device))
    #print(p)
    m = Softmax(dim=1)
    action_prob = m(p)
    #print(action_prob)
    c = Categorical(action_prob)
    a = c.sample()
    #print(a)
    return a.item(), p

def discount_reward(r_dic, gamma):
    """
    change reward like below
    [0, 0, 0, 0, 1] -> [0.99^4, 0.99^3, 0.99^2, 0.99, 1]
    and then normalize
    """
    r = 0
    for i in range(len(r_dic) - 1, -1, -1):
        if r_dic[i] != 0:
            r = r_dic[i]
        else:
            r = r * gamma
            r_dic[i] = r
    r_dic = (r_dic - r_dic.mean()) / (r_dic.std() + 1e-8)
    return r_dic

def optimize_model(act_p, act, reward):

    label = act
    act_p = act_p.squeeze()
    #print(act_p)
    #print(reward)
    #print(act)
    label = label.long()
    loss_fn = nn.CrossEntropyLoss(reduction="none") # for mulitple output
    loss_value = loss_fn(act_p, label)
    #print(loss_value.size())
    loss = torch.dot(loss_value, reward)
    loss = Variable(loss, requires_grad = True)
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(env, n_episodes, history, render=False):
    global epoch
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        state = get_state(obs)  # torch.Size([1, 4, 84, 84])
        total_reward = 0.0
        action_prob_pool = []
        action_pool = []
        reward_pool = []
        for t in count():
            action, action_prob  = select_action(state)

            if render:
                save_dir = os.path.join(args.store_dir, 'video')
                env.render(mode=render, save_dir=save_dir)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            if args.debug:
                display_state(next_state)


            action_pool.append(action)
            action_prob_pool.append(action_prob)
            reward_pool.append(reward)

            state = next_state

            if steps_done > INITIAL_MEMORY:
                #optimize_model()
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epoch += 1

        history.append((total_reward, t))

        action_prob_pool = torch.stack(action_prob_pool)

        action_pool = np.array(action_pool)
        action_pool = torch.from_numpy(action_pool).float().to(device)

        reward_pool = np.array(reward_pool)
        reward_pool = torch.from_numpy(reward_pool).float().to(device)

        reward_pool = discount_reward(reward_pool, GAMMA)
        optimize_model(action_prob_pool, action_pool, reward_pool)

        if episode % LOG_INTERVAL == 0:
            avg_reward = sum([h[0] for h in history[-LOG_INTERVAL:]]) / LOG_INTERVAL
            avg_steps = int(sum([h[1] for h in history[-LOG_INTERVAL:]]) / LOG_INTERVAL)
            logger.info(f'Total steps: {steps_done}\tEpisode: {epoch}/{t}\tAvg reward: {avg_reward:.2f}\t'
                        f'Avg steps: {avg_steps}')
        if episode % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(args.store_dir)

    env.close()

    if render == 'png':
        convert_images_to_video(image_dir=save_dir, save_dir=os.path.dirname(save_dir))
        shutil.rmtree(save_dir)

    return history





def get_logger(store_dir):
    log_path = os.path.join(store_dir, 'output.log')
    logger = logging.Logger('train_status', level=logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter('%(levelname)s\t%(message)s'))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def save_checkpoint(store_dir):
    global steps_done
    global epoch
    torch.save(
        {'Net': policy_net.state_dict(), 'Optimizer': optimizer.state_dict(), 'Steps_Done': steps_done, 'Epoch': epoch},
        os.path.join(store_dir, 'dqn_pong_model'))
    pickle.dump(history, open(os.path.join(store_dir, 'history.p'), 'wb'))


# noinspection PyProtectedMember
def get_args_status_string(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """
    Returns a formatted string of the passed arguments.

    :param parser: The `argparse.ArgumentParser` object to read from
    :param args: The values of the parsed arguments from `parser.parse_args()`
    :return: "--width 160 --height 160 --ball 3.0 ..."
    """
    args_info = parser._option_string_actions
    s = ''
    for k, v in args_info.items():
        if isinstance(v, argparse._StoreAction):
            s += f'{k} {args.__dict__[v.dest]} '
    return s

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Dynamic Pong RL')

    '''environment args'''
    env_args = parser.add_argument_group('Environment', "Environment controls")
    env_args.add_argument('--width', default=160, type=int,
                          help='canvas width (default: 160)')
    env_args.add_argument('--height', default=160, type=int,
                          help='canvas height (default: 160)')
    env_args.add_argument('--ball', default=3.0, type=float,
                          help='ball speed (default: 3.0)')
    env_args.add_argument('--ball-size', dest='ball_size', default=2.0, type=float,
                          help='ball size (default: 2.0)')
    env_args.add_argument('--snell', default=3.0, type=float,
                          help='snell speed (default: 3.0)')
    env_args.add_argument('--no-refraction', dest='no_refraction', default=False, action='store_true',
                          help='set to disable refraction')
    env_args.add_argument('--uniform-speed', dest='uniform_speed', default=False, action='store_true',
                          help='set to disable a different ball speed in the Snell layer')
    env_args.add_argument('--snell-width', dest='snell_width', default=40.0, type=float,
                          help='snell speed (default: 40.0)')
    env_args.add_argument('--snell-change', dest='snell_change', default=0, type=float,
                          help='Standard deviation of the speed change per step (default: 0)')
    env_args.add_argument('--snell-visible', dest='snell_visible', default='none', type=str,
                          choices=['human', 'machine', 'none'],
                          help="Determine whether snell is visible to when rendering ('render') or to the agent and "
                               "when rendering ('machine')")
    env_args.add_argument('--paddle-speed', default=3.0, type=float,
                          help='paddle speed (default: 3.0)')
    env_args.add_argument('--paddle-angle', default=45, type=float,
                          help='Maximum angle the ball can leave the paddle (default: 45deg)')
    env_args.add_argument('--paddle-length', default=45, type=float,
                          help='paddle length (default: 45)')
    env_args.add_argument('--update-prob', dest='update_prob', default=0.2, type=float,
                          help='Probability that the opponent moves in the direction of the ball (default: 0.2)')
    env_args.add_argument('--state', default='binary', type=str, choices=['binary', 'color'],
                          help='state representation (default: binary)')

    '''RL args'''
    rl_args = parser.add_argument_group("Model", "Reinforcement learning model parameters")
    rl_args.add_argument('--learning-rate', default=1e-4, type=float,
                         help='learning rate (default: 1e-4)')
    rl_args.add_argument('--network', default='dqn_pong_model',
                         help='choose a network architecture (default: dqn_pong_model)')
    rl_args.add_argument('--double', default=False, action='store_true',
                         help='switch for double dqn (default: False)')
    rl_args.add_argument('--pretrain', default=False, action='store_true',
                         help='switch for pretrained network (default: False)')
    rl_args.add_argument('--test', default=False, action='store_true',
                         help='Run the model without training')
    rl_args.add_argument('--render', default=False, type=str, choices=['human', 'png'],
                         help="Rendering mode. Omit if no rendering is desired.")
    rl_args.add_argument('--epsdecay', default=1000, type=int,
                         help="epsilon decay (default: 1000)")
    rl_args.add_argument('--stepsdecay', default=False, action='store_true',
                         help="switch to use default step decay")
    rl_args.add_argument('--episodes', dest='episodes', default=4000, type=int,
                         help='Number of episodes to train for (default: 4000)')
    rl_args.add_argument('--replay', default=10000, type=int,
                         help="change the replay mem size (default: 10000)")
    rl_args.add_argument('--priority', default=False, action='store_true',
                         help='switch for prioritized replay (default: False)')
    rl_args.add_argument('--rankbased', default=False, action='store_true',
                         help='switch for rank-based prioritized replay (omit if proportional)')

    '''resume args'''
    resume_args = parser.add_argument_group("Resume", "Store experiments / Resume training")
    resume_args.add_argument('--resume', dest='resume', action='store_true',
                             help='Resume training switch. (omit to start from scratch)')
    resume_args.add_argument('--checkpoint', default='dqn_pong_model',
                             help='Checkpoint to load if resuming (default: dqn_pong_model)')
    resume_args.add_argument('--history', default='history.p',
                             help='History to load if resuming (default: history.p)')
    resume_args.add_argument('--store-dir', dest='store_dir',
                             default=os.path.join('..', 'experiments', time.strftime("%Y-%m-%d %H.%M.%S")),
                             help='Path to directory to store experiment results (default: ./experiments/<timestamp>/')

    '''debug args'''
    debug_args = parser.add_argument_group("Debug")
    debug_args.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # create storage directory
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = args.epsdecay
    TARGET_UPDATE = 1000
    RENDER = args.render
    lr = args.learning_rate
    INITIAL_MEMORY = args.replay
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    DOUBLE = args.double
    STEPSDECAY = args.stepsdecay
    PRIORITY = args.priority

    # number episodes between logging and saving
    LOG_INTERVAL = 20
    CHECKPOINT_INTERVAL = 100

    resume = args.resume

    logger = get_logger(args.store_dir)
    logger.info(get_args_status_string(parser, args))
    logger.info(f'Device: {device}')

    # create environment
    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make(
        "gym_dynamic_pong:dynamic-pong-v1",
        max_score=20,
        width=args.width,
        height=args.height,
        default_speed=args.ball,
        snell_speed=args.snell,
        snell_width=args.snell_width,
        snell_change=args.snell_change,
        snell_visible=args.snell_visible,
        refract=not args.no_refraction,
        uniform_speed=args.uniform_speed,
        our_paddle_speed=args.paddle_speed,
        their_paddle_speed=args.paddle_speed,
        our_paddle_height=args.paddle_length,
        their_paddle_height=args.paddle_length,
        our_paddle_angle=math.radians(args.paddle_angle),
        their_paddle_angle=math.radians(args.paddle_angle),
        their_update_probability=args.update_prob,
        ball_size=args.ball_size,
        state_type=args.state,
    )
    # TODO: consider removing some of the wrappers - may improve performance
    env = make_env(env, episodic_life=True, clip_rewards=True)

    # create networks
    #architecture = args.network
    pretrain = args.pretrain

    policy_net = PolicyGradient(n_actions=env.action_space.n).to(device)
    target_net = PolicyGradient(n_actions=env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0
    epoch = 0

    if resume:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy_net.load_state_dict(checkpoint['Net'])
        optimizer.load_state_dict(checkpoint['Optimizer'])
        target_net.load_state_dict(policy_net.state_dict())
        steps_done = checkpoint['Steps_Done']
        epoch = checkpoint['Epoch']
        logger.info("Loading the trained model")
        history = pickle.load(open(args.history, 'rb'))
    else:
        history = []

    if args.test:  # test
        test(env, 1, policy_net, render=RENDER)
    else:  # train
        history = train(env, args.episodes, history, render=RENDER)
        save_checkpoint(args.store_dir)
