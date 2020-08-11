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

def select_action(p):
    # state = torch.FloatTensor(state).unsqueeze(0).to(device)
    # print('state : ', state)
    #with torch.no_grad():
    #p = policy_net.forward(state.to(device))
    m = Softmax(dim=1)
    probs = m(p)
    c = Categorical(probs)
    a = c.sample()
    #print(a)
    return a.item()

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
    #r_dic = (r_dic - r_dic.mean()) / (r_dic.std() + 1e-8)
    return r_dic

def update_target_net():
    if steps_done % TARGET_UPDATE == 0:
        target_net.load_state_dict(value_net.state_dict())

def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    state_action_values = value_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # action_logits = policy_net(state_batch)
    # label = action_batch.squeeze()
    # # #print(label)
    # # #print(action_logits)
    # policy_loss_fn = nn.CrossEntropyLoss(reduction="none")
    # policy_loss_value = policy_loss_fn(action_logits, label)
    # # #print(policy_loss_value.size())
    # # #print(state_action_values.size())
    # q_values = state_action_values.squeeze()
    # policy_loss = torch.dot(policy_loss_value, q_values.detach())


    # optimizerP.zero_grad()
    optimizerV.zero_grad()
    loss.backward()
    # policy_loss.backward()
    #for param in value_net.parameters():
        #param.grad.data.clamp_(-1, 1)
        #print(param.grad)
    # optimizerP.step()
    optimizerV.step()

def optimize_policy_model():
    if steps_done % TARGET_UPDATE != 0:
        return
    if len(policy_memory) < 1000:
        return

    transitions = policy_memory.sample(1000)

    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    state_action_values = value_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(1000, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()
    action_logits = policy_net(state_batch)
    label = action_batch.squeeze()
    # #print(label)
    # #print(action_logits)
    policy_loss_fn = nn.CrossEntropyLoss(reduction="none")
    policy_loss_value = policy_loss_fn(action_logits, label)
    # #print(policy_loss_value.size())
    #print(state_action_values)
    q_values = state_action_values.squeeze()
    advantage = expected_state_action_values - q_values
    if architecture == 'a2c':
        #print("a2c")
        policy_loss = torch.dot(policy_loss_value, advantage)
    else:
        policy_loss = torch.dot(policy_loss_value, q_values.detach())


    optimizerP.zero_grad()
    policy_loss.backward()
    #for param in policy_net.parameters():
        #param.grad.data.clamp_(-1, 1)
        #print(param.grad)
    optimizerP.step()



def train(env, n_episodes, history, render=False):
    global epoch
    global steps_done
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        state = get_state(obs)  # torch.Size([1, 4, 84, 84])
        total_reward = 0.0

        #action_prob_pool = []
        #action_pool = []
        #reward_pool = []
        #state_pool = []
        #value_pool = []
        #action_tensor_pool = []
        #next_value_pool = []
        #masks = []

        for t in count():
            action_prob = policy_net.forward(state.to(device))
            value = value_net.forward(state.to(device))
            steps_done += 1
            #print(value)
            action  = select_action(action_prob)
            #print(action)

            if render:
                save_dir = os.path.join(args.store_dir, 'video')
                env.render(mode=render, save_dir=save_dir)

            obs, reward, done, info = env.step(action)

            reward_tensor = torch.tensor([reward], device=device)
            action_tensor = torch.tensor([[action]], device=device, dtype=torch.long)

            if len(memory) > 1000:
                optimize_model()
                optimize_policy_model()
                update_target_net()


            total_reward += reward

            if (reward != 0.0):
                next_state = None

            elif not done:
                next_state = get_state(obs)
                # next_value = value_net.forward(next_state.to(device)).detach()
                #next_value_pool.append(next_value)
                #print(next_value)
            else:
                next_state = None
                # next_value = torch.tensor([[0.0]], dtype=torch.float, device=device)

            if args.debug:
                display_state(next_state)

            memory.store(state, action_tensor.to('cpu'), next_state, reward_tensor.to('cpu'))
            policy_memory.store(state, action_tensor.to('cpu'), next_state, reward_tensor.to('cpu'))
            #action_pool.append(action)
            #action_tensor_pool.append(action_tensor)
            #action_prob_pool.append(action_prob)
            #reward_pool.append(reward)
            #state_pool.append(state)
            #value_pool.append(value)
            #next_value_pool.append(next_value)
            #masks.append(1-done)

            if (reward != 0.0):
                next_state = get_state(obs)
            state = next_state

            # if steps_done > INITIAL_MEMORY:
            #     #optimize_model()
            #     if steps_done % TARGET_UPDATE == 0:
            #         target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epoch += 1
        #print(len(memory))

        history.append((total_reward, t))

        #action_prob_pool = torch.stack(action_prob_pool)
        #action_pool = np.array(action_pool)
        #action_pool = torch.from_numpy(action_pool).float().to(device)
        #
        #action_tensor_pool = torch.stack(action_tensor_pool)
        #action_tensor_pool = action_tensor_pool.squeeze(1)

        #masks = np.array(masks)
        #masks = torch.from_numpy(masks).to(device)
        #
        #reward_pool = np.array(reward_pool)
        #reward_pool = torch.from_numpy(reward_pool).float().to(device)

        #value_pool = torch.stack(value_pool)
        #value_pool = value_pool.squeeze()
        #print(action_pool)
        #print(value_pool.gather(1, action_tensor_pool).squeeze())
        #state_action_value = value_pool.gather(1, action_tensor_pool).squeeze()
        #state_action_value = state_action_value/20.0
        # print(state_action_value.size())

        # next_value_pool = torch.stack(next_value_pool)
        # next_value_pool = next_value_pool.squeeze()

        #reward_pool = discount_reward(reward_pool, GAMMA)

        #label = action_pool
        #act_p = action_prob_pool.squeeze()
        #vals = value_pool.detach()
        #label = label.long()
        #print(vals)
        #print(act_p.size())
        #print(value_pool)
        #print(reward_pool.size())
        # advantage = reward_pool + GAMMA*next_value_pool - vals
        # # advantage_loss = reward_pool + GAMMA*next_value_pool - value_pool
        #policy_loss_fn = nn.CrossEntropyLoss(reduction="none")
        #policy_loss_value = policy_loss_fn(act_p, label)
        #print(state_action_value)
        #policy_loss = torch.dot(policy_loss_value, state_action_value.detach())
        #
        # #value_loss_fn = nn.MSELoss()
        # # value_loss = advantage_loss.pow(2).mean()
        #
        #optimizerP.zero_grad()
        # # optimizerV.zero_grad()
        #
        #policy_loss.backward()
        # # value_loss.backward()
        #for param in policy_net.parameters():
             #param.grad.data.clamp_(-1, 1)
            #print(param.grad)
        #optimizerP.step()
        # optimizerV.step()
        #optimize_model(action_prob_pool, action_pool, reward_pool)

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
        {'Net': policy_net.state_dict(), 'Optimizer': optimizerP.state_dict(), 'Steps_Done': steps_done, 'Epoch': epoch},
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
    rl_args.add_argument('--epsdecay', default=10000, type=int,
            help="epsilon decay (default: 10000)")
    rl_args.add_argument('--stepsdecay', default=False, action='store_true',
                         help="switch to use default step decay")
    rl_args.add_argument('--episodes', dest='episodes', default=4000, type=int,
                         help='Number of episodes to train for (default: 4000)')
    rl_args.add_argument('--replay', default=1000, type=int,
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
    architecture = args.network
    pretrain = args.pretrain

    policy_net = Actor(n_actions=env.action_space.n).to(device)
    value_net = Critic(n_actions=env.action_space.n).to(device)
    target_net = Critic(n_actions=env.action_space.n).to(device)
    target_net.load_state_dict(value_net.state_dict())

    # setup optimizer
    optimizerP = optim.Adam(policy_net.parameters(), lr=lr)
    optimizerV = optim.Adam(value_net.parameters(), lr=lr)
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

    memory = ReplayMemory(MEMORY_SIZE)
    policy_memory = ReplayMemory(1000)

    if args.test:  # test
        test(env, 1, policy_net, render=RENDER)
    else:  # train
        logger.info("Training the Policy gradient model")
        history = train(env, args.episodes, history, render=RENDER)
        save_checkpoint(args.store_dir)
