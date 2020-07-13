import scipy.misc
import numpy as np
import torch
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from policy import Policy_gradient


def prepro(o, image_size=[80, 80]):
    """
    turn rgb -> gray
    Input:
        np.array [210, 160, 3]
    Output:
        np.array [1, 1, 80, 80]
    https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    """
    o = o[35:,:,:]
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    new_img = resized.reshape(1, 1, 80, 80)
    return new_img


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


class Agent():
    def __init__(self, args):
        print("Init Agent")
        # env
        self.env = gym.make('Pong-v0')
        self.env.seed(11037)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # state
        self.pre_state = None
        self.diff_state = None
        # model
        self.model = Policy_gradient(args).to(self.device)
        # visualize
        self.comet_experiment = None
        if args.comet:
            self.comet_experiment = Experiment(
                api_key="DFqdpuCkMgoVhT4sJyOXBYRRN",
                project_name="pong-v0"
            )

    def _set_env(self):
        _ = self.env.reset()

        state, reward, done, info = self.env.step(0)
        state = prepro(state)
        self.pre_state = state
        
        state, reward, done, info = self.env.step(0)
        state = prepro(state)
        self.diff_state = state - self.pre_state
        self.pre_state = state
        
        return done

    def train(self, args):
        print("start training")
        # for visualize
        step = 0
        reward_per_game = 0
        
        # start train
        for e in range(args.epochs):
            # init training set for an epoch
            act_dic = []
            act_p_dic = []
            r_dic = []
            num_game = 0
            done = self._set_env()
            while True:
                if done:
                    # end of the game
                    if self.comet_experiment != None:
                        self.comet_experiment.log_metric("reward", reward_per_game, step=step)
                    num_game = num_game + 1
                    step = step + 1
                    reward_per_game = 0
                    # end of the epoch
                    if num_game == args.batch_size:
                        break
                    done = self._set_env()
                else:
                    # play the game
                    self.diff_state = torch.from_numpy(self.diff_state).float().to(self.device)
                    act, act_p = self.model.get_action(self.diff_state)
                    state, reward, done, _ = self.env.step(act)
                    state = prepro(state)
                    self.diff_state = state - self.pre_state
                    self.pre_state = state
                    # remember act, and the posibility of it
                    act_dic.append(act)
                    act_p_dic.append(act_p)
                    r_dic.append(reward)

                    reward_per_game = reward_per_game + reward

            # dic -> numpy -> tensor
            act_p_dic = torch.stack(act_p_dic)
                
            act_dic = np.array(act_dic)
            act_dic = torch.from_numpy(act_dic).float().to(self.device)
                
            r_dic = np.array(r_dic)
            r_dic = torch.from_numpy(r_dic).float().to(self.device)
            
            # discount_reward
            r_dic = discount_reward(r_dic, args.gamma)
            
            # update model
            loss = self.model.update(act_p_dic, act_dic, r_dic)
            if self.comet_experiment != None:
                self.comet_experiment.log_metric("loss", loss, step=step)
            # save model
            if(e % 100 == 0 and e != 0):
                self.model.save(e) 
                print("save:", e) 
    
    def load(self):
        pass

    def test(self):
        pass


def save_img(path, img):
    img = img.reshape(80, 80)
    scipy.misc.imsave(path, img)

class LinearPolicy(nn.Module):
    def __init__(self):
        super(LinearPolicy, self).__init__()
        self.linear1 = nn.Linear(80 * 80, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, observation, train=True):
        x = observation.view(-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # to get act_int
        action = self.softmax(x)
        if train:
            if action[0] > np.random.uniform():
                act_int = 2
            else:
                act_int = 3
        else:
            if action[0] > 0.5:
                act_int = 2
            else:
                act_int = 3
        return act_int, x


class Policy_gradient(nn.Module):
    def __init__(self, args):
        super(Policy_gradient, self).__init__()
        self.Policy = LinearPolicy()
        self.opt = torch.optim.Adam(self.Policy.parameters(), args.lr)

    def get_action(self, state, train=True):
        act_int, act_p = self.Policy(state, train)
        return act_int, act_p

    def update(self, act_p, act, reward):
        """
        act_p : [N, 2]
        act : [N]
        reward : [N]
        """
        # get label
        label = act - 2 # act is 2 and 3 -> 0, 1
        label = label.long() # pytorch CrossEntropyLoss format
        # loss
        self.opt.zero_grad()
        loss_fn = nn.CrossEntropyLoss(reduction="none") # for mulitple output
        loss = loss_fn(act_p, label)
        loss = torch.dot(loss, reward)
        loss.backward()
        self.opt.step()
        return loss.data.cpu().numpy()

    def save(self, e=0):
        torch.save(self.Policy, "./model/Linear_" + str(e) + ".model")

    def load(self, path):
        self.Policy = torch.load(path)

    
if __name__ == "__main__":
    print("development mode")
