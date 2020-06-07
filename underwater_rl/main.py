import math
import random
import time
from collections import namedtuple
from itertools import count
import warnings
import argparse

from underwater_rl.memory import ReplayMemory
from underwater_rl.models import *
from underwater_rl.wrappers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


warnings.filterwarnings("ignore", category=UserWarning)

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        # TODO: should this just go the the CPU?
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)


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

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        if episode % 20 == 0:
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()
    return


def test(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video', force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Dynamic Pong RL')
    parser.add_argument('--width', default=224, type=int, 
                        help='canvas width (default: 224)')
    parser.add_argument('--height', default=224, type=int,
                        help='canvas height (default: 224)')
    parser.add_argument('--ball', default=3.0, type=float,
                        help='ball speed (default: 3.0)')
    parser.add_argument('--snell', default=3.0, type=float,
                        help='snell speed (default: 3.0)')
    parser.add_argument('--ps','--paddle-speed', default=3.0, type=float,
                        help='paddle speed (default: 3.0)')
    parser.add_argument('--pl','--paddle-length', default=45, type=int,
                        help='paddle length (default: 45)')
    parser.add_argument('--lr','--learning-rate', default=1e-4, type=float,
                        help='learning rate (default: 1e-4)')

    args = parser.parse_args()
    parser.print_help()
    
    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    resume = False

    # create environment
    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make(
        "gym_dynamic_pong:dynamic-pong-v0",
        max_score=20,
        width=args.width,
        height=args.height,
        default_speed=args.ball,
        snell_speed=args.snell,
        our_paddle_speed=args.ps,
        their_paddle_speed=args.ps,
        our_paddle_height=args.pl,
        their_paddle_height=args.pl,
    )
    # TODO: consider removing some of the wrappers - may improve performance
    env = make_env(env, episodic_life=True, clip_rewards=True)

    # create networks
    '''
    policy_net = DQN(n_actions=env.action_space.n).to(device)
    target_net = DQN(n_actions=env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    '''
    policy_net = resnet18(num_classes=env.action_space.n).to(device)
    target_net = resnet18(num_classes=env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    if (resume):
        checkpoint = torch.load("dqn_pong_model", map_location=device)
        policy_net.load_state_dict(checkpoint['Net'])
        optimizer.load_state_dict(checkpoint['Optimizer'])
        target_net.load_state_dict(policy_net.state_dict())

    steps_done = 0

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model
    train(env, 2000, render=RENDER)
    torch.save({'Net': policy_net.state_dict(), 'Optimizer': optimizer.state_dict()}, "dqn_pong_model")
    # policy_net = torch.load("dqn_pong_model")
    checkpoint = torch.load("dqn_pong_model", map_location=device)
    policy_net.load_state_dict(checkpoint['Net'])
    test(env, 1, policy_net, render=RENDER)
