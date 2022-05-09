import gym
import torch
import numpy as np
import os
import time
from nn import ValueNN, PolicyNN, CombinedNN

gym.make("CartPole-v1")

total_rewards_file = '../data/total_rewards-'

class Logger:

    def __init__(self, num_traj, num_training):
        self.trajectory_idx = 0
        self.training_idx = 0
        self.rewards = np.array([[0 for _ in range(num_traj)] for _ in range(num_training)])
        self.folder_path = '../data/' + time.strftime("%Y%m%d-%H%M%S")

    def add_trajectory_reward(self, reward):
        self.rewards[self.training_idx, self.trajectory_idx] = reward
        self.trajectory_idx += 1

    def new_training(self):
        self.training_idx += 1
        self.trajectory_idx = 0
    
    def output_training_step(self):
        print("all rewards " + str(self.rewards[self.training_idx]))
        print("avg reward " + str(sum(self.rewards[self.training_idx])/len(self.rewards[self.training_idx])))

    def average_rewards(self):
        return [sum(x)/len(x) for x in self.rewards]

    
    def save(self, hyperparams : dict, model):
        os.mkdir(self.folder_path)
        f = open(self.folder_path + "/meta.txt", "w")
        f.write('Hyperparameters \n')
        for param in hyperparams:
            f.write(param + " : " + str(hyperparams[param]) + "\n")
        f.close()
        torch.save(model, self.folder_path + "/model.pth")
        f = open(total_rewards_file + str(hyperparams["num training cycles"]) + '.txt', "a")
        f.write(self.average_rewards().__str__() + "\n")
        np.save('rewards.npy', np.array(self.rewards))


import numpy
def generate_trajectory(env, model : CombinedNN, render_last_step: bool=True, logger : Logger = None,  ):
    episode = []
    s = env.reset()
    probs, value = model(torch.tensor(s))
    a = probs.sample()
    done = False
    tot_reward = 0
    while not done:
        with torch.no_grad():
            s_p, r, done, info = env.step(a.item())
            a = probs.sample()
            episode.append((s, a, r, probs.log_prob(a), value))
            s = s_p
            probs, value = model(torch.tensor(s_p))
            if done and render_last_step:
                env.render()
            tot_reward += r
    logger.add_trajectory_reward(tot_reward)
    return episode

from typing import List
def advantage_estimates(trajectory, gamma : float, lmbda, model):
    advantages = torch.zeros(len(trajectory))
    values = torch.zeros(len(trajectory))
    prev_adv = 0.0
    for t in range(len(trajectory)-2, -1, -1):
        with torch.no_grad():
            s_t, a_t, r_t, _, _ = trajectory[t]
            s_t_1, _, _, _, _ = trajectory[t+1]
            _, curr_val = model(torch.tensor(s_t)) 
            delta = r_t + gamma * model(torch.tensor(s_t_1))[1] - curr_val
            prev_adv = delta + gamma * lmbda * prev_adv
            values[t] = curr_val
            advantages[t] = prev_adv
    return advantages, values


def calc_loss(trajectories, old_log_probs, adv_tensors, old_values, sampled_returns, epsilon : float, model : CombinedNN):

    size_of_tensors = sum([len(traj) for traj in trajectories])
    new_log_probs = torch.zeros(size_of_tensors)
    new_values = torch.zeros(size_of_tensors)
    entropies = torch.zeros(size_of_tensors)

    tensor_idx = 0
    for i, adv in enumerate(trajectories):
        for j, step in enumerate(adv):
            state = torch.tensor(trajectories[i][j][0])
            new_probs, new_value = model(state)
            new_log_probs[tensor_idx] = new_probs.log_prob(trajectories[i][j][1])
            entropies[tensor_idx] = new_probs.entropy()
            new_values[tensor_idx] = (new_value)
            tensor_idx += 1

    norm_sampled_adv = (adv_tensors - adv_tensors.mean()) / (adv_tensors.std() + 1e-8)



    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(min=1.0 - epsilon, max=1.0 + epsilon)
    policy_objective = torch.min(ratio * norm_sampled_adv, clipped_ratio * norm_sampled_adv)
    # print(policy_objective)
    policy_objective = policy_objective.mean()
    entropy_bonus = entropies.mean()

    # calculate value parameter delta
    clipped_value = old_values + (new_values - old_values).clamp(min=-epsilon, max=epsilon)
    # print(new_values-old_values)
    value_loss = torch.max((new_values - sampled_returns) ** 2, (clipped_value - sampled_returns) ** 2)
    value_loss = .5 * value_loss.mean()

    loss = -(policy_objective - 0.5 * value_loss + .01*entropy_bonus)
    # loss = -(policy_objective - 0.5 * value_loss)

    return loss

def main():

    env = gym.make("CartPole-v1")
    lr = 1e-2
    gamma = .99
    lmbda = .95
    epsilon = .2

    NUM_TRAINING = 150
    NUM_EPOCHS = 3
    NUM_TRAJECTORIES = 15
    model = CombinedNN(
        env.observation_space.shape[0],
        2,
        32,
        env.action_space.n
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hyperparams = {
        "lr" : lr,
        "gamma" : gamma,
        "lmbda" : lmbda,
        "epsilon" : epsilon,
        "model_parameters" : num_params,
        "num training cycles": NUM_TRAINING,
        "num epochs": NUM_EPOCHS,
        "num trajectories" : NUM_TRAJECTORIES
    }


    model.train()


    logger = Logger(NUM_TRAJECTORIES, NUM_TRAINING)
    model_optim = torch.optim.Adam(model.parameters(),  lr=lr)
    for training_idx in range(NUM_TRAINING):
        trajectories = []
        advantages = []
        values = []
        for i in range(NUM_TRAJECTORIES):
            traj = generate_trajectory(env, model, False, logger=logger)
            trajectories.append(traj)
            adv, vals = advantage_estimates(traj, gamma, lmbda, model)
            advantages.append(adv)
            values.append(vals)

        size_of_tensors = sum([len(traj) for traj in trajectories])
        sampled_returns = torch.zeros(size_of_tensors)
        old_values = torch.zeros(size_of_tensors)
        adv_tensors = torch.empty(0)
        old_log_probs = torch.zeros(size_of_tensors)
        tensor_idx = 0
        # do not include the old policy/samples in the pytorch computational graph
        # will prevent us from calling multiple backprops on the same training trajectories
        with torch.no_grad():
            for i, adv in enumerate(advantages):
                for j, step in enumerate(adv):
                
                        sampled_returns[tensor_idx]=(values[i][j] + adv[j]) 
                        old_log_probs[tensor_idx] = trajectories[i][j][3]
                        old_values[tensor_idx] = (trajectories[i][j][4])
                        tensor_idx += 1

                adv_tensors = torch.cat((adv_tensors, adv))

        for epoch_idx in range(NUM_EPOCHS):
            
            loss = calc_loss(trajectories, old_log_probs, adv_tensors, old_values, sampled_returns, epsilon, model)
            model_optim.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            model_optim.step()
            # print(loss)

        # downgrade learning rate for more stability towards the end of training
        # new_lr = lr * (((NUM_TRAINING - training_idx)/NUM_TRAINING) ** 2)
        # print(new_lr)
        # have to go into optimizer to modify learning rate
        # for pg in model_optim.param_groups:
            # pg['lr'] = new_lr
        # downgrade clipping rate (epsilon)
        # epsilon = .2 * ((NUM_TRAINING - training_idx)/NUM_TRAINING)
        print(str(training_idx) + "/" + str(NUM_TRAINING))
        logger.output_training_step()
        logger.new_training()
    
    logger.save(hyperparams, model)



if __name__ == "__main__":
    main()