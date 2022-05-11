from re import T
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

class PiNN(torch.nn.Module):
    # n_hidden = number of nodes in hidden layer
    # n_layers = number of hidden layers (UNUSED)
    # n_feature = number of inputs
    # n_output = number of outputs
    def __init__(self, n_feature, n_layers : int, n_hidden : int, n_output):
        super(PiNN, self).__init__()
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        layer_output = F.relu(self.hidden0(input))
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        actions = self.predict(layer_output)
        network_output = F.softmax(actions)    # linear output
        action_probs = Categorical(network_output)
        action = action_probs.sample()
        return action, action_probs

class VNN(torch.nn.Module):
    def __init__(self, n_feature, n_layers, n_hidden : int, n_output):
        super(VNN, self).__init__()
        self.hidden0 = (torch.nn.Linear(n_feature, n_hidden))
        self.hidden1 = (torch.nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, input):
        layer_output = F.relu(self.hidden0(input))
        layer_output = F.relu(self.hidden1(layer_output))      # activation function for hidden layer
        network_output = self.predict(layer_output)    # linear output
        return network_output


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.nn = PiNN(state_dims, 2, 32, num_actions)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), betas=[.9, .999], lr=alpha)
        # self.probabilities = dict()

    # def get_action_prob(self, s, a):
    #     self.nn.eval()
    #     action_probs = self.nn(torch.from_numpy(s).float())
    #     return action_probs[a]

    def __call__(self,s) -> int:
        self.nn.eval()
        action, action_probs = self.nn(torch.from_numpy(s).float())
        torch.autograd.set_detect_anomaly(True)
        # self.probabilities[(tuple(s), action)] = torch.log(action_probs.squeeze(0)[action])
        return action, action_probs

    def update(self, s, a, gamma_t, delta, probs):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.nn.train()
        _, probs = self.nn(torch.from_numpy(s).float())
        loss = torch.tensor(gamma_t) * torch.tensor(delta) * -1 * probs.log_prob(a)
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        self.optimizer.step()

        return None

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.alpha = alpha
        self.nn = VNN(state_dims, 2, 32, 1) # num_layers not used :)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), betas=[.9, .999], lr=alpha)
        

    def __call__(self,s) -> float:
        self.nn.eval()
        val = self.nn(torch.from_numpy(s).float())
        return val.item()

    def update(self,s,G):
        self.nn.train()
        self.optimizer.zero_grad()
        prediction = self.nn(torch.from_numpy(s).float()) # input: state, output : scalar
        loss_function = torch.nn.MSELoss()
        loss = loss_function(prediction, torch.tensor([G]))
        # loss = G - prediction
        loss.backward() # computes the gradients of the network parameters with respect to the loss
        self.optimizer.step()

from generate_gif import save_frames_as_gif
def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_time_steps:int,
    pi:PiApproximationWithNN,
    V:Baseline, logger) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    rewards = [0.0 for i in range(num_time_steps)]
    time_step = 0
    reached_max = False
    already_saved_video = False
    while time_step < num_time_steps:
        Episode = []
        s = env.reset()
        a, probs = pi(s)
        done = False
        save_time_step = time_step
        ep_reward = 0.0
        frames = []
        while not done:
            s_p, r, done, info = env.step(a.item())
            Episode.append((s, a, r, probs))
            s = s_p
            a, probs = pi(s_p)
            if not already_saved_video and logger['render'] and reached_max:
                frames.append(env.render(mode="rgb_array"))
            # if done:
                # print(s_p)
                # env.render()
            time_step += 1
            ep_reward += r
        if ep_reward >= 499.0:
            reached_max = True
        if not already_saved_video and ep_reward >= 499.0 and len(frames) >= 499:
            already_saved_video = True
            file = 'REINFORCE.gif'
            if isinstance(V, VApproximationWithNN):
                file = 'REINFORCE-BASELINE.gif'
            save_frames_as_gif(frames, path='../data/videos/', filename=file)
            print("SAVED VIDEO")
        
        for i in range(save_time_step, time_step):
            if i >= len(rewards):
                break
            rewards[i] = ep_reward

        T = len(Episode)
        for t, step in enumerate(Episode):
            s, a, r, probs = step 
            G = sum([pow(gamma, idx)*Episode[idx][2] for idx in range(t, T)])
            # if t == 0:
            #     to_return.append(G)
            # print("state : " + str(s))
            delta = G - V(s)
            # print(delta)
            V.update(s, G)
            pi.update(s, a, pow(gamma, t), delta, probs)
        
        # print("time steps " + str(time_step))
    return rewards

    


