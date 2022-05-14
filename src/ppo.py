import gym
import torch
import numpy as np
import os
import time
import sys
from nn import ValueNN, PolicyNN, CombinedNN

gym.make("CartPole-v1")

total_rewards_file = '../data/total_rewards-'
from generate_gif import save_frames_as_gif

class Logger:

    def __init__(self, num_traj, num_time_steps, record_frames=False):
        self.time_step = 0
        self.rewards = np.array([0 for _ in range(num_time_steps)])
        self.folder_path = '../data/PPO_runs/' + time.strftime("%Y%m%d-%H%M%S")
        self.tmp_rewards = []
        self.should_record_frames = record_frames
        self.frames = []
        self.reached_max = False
        self.already_recorded = False

    def add_trajectory_reward(self, reward, steps_elapsed):
        for i in range(self.time_step, self.time_step + steps_elapsed):
            if i >= len(self.rewards):
                break
            self.rewards[i] = reward
        self.tmp_rewards.append(reward)
        self.time_step += steps_elapsed
        if reward >= 499.0:
            self.reached_max = True
    
    def record_frames(self):
        file = 'PPO.gif'
        print("RECORDED FRAMES PPO")
        save_frames_as_gif(self.frames, path='../data/videos/', filename=file)

    
    # def output_training_step(self):
        # print("all rewards " + str(self.rewards[self.training_idx]))
        # print("avg reward " + str(sum(self.rewards[self.training_idx])/len(self.rewards[self.training_idx])))

    def average_rewards(self):
        print(self.tmp_rewards)
        print(sum(self.tmp_rewards)/len(self.tmp_rewards))
        self.tmp_rewards = []

    
    def save(self, hyperparams : dict, model):
        os.mkdir(self.folder_path)
        f = open(self.folder_path + "/meta.txt", "w")
        f.write('Hyperparameters \n')
        for param in hyperparams:
            f.write(param + " : " + str(hyperparams[param]) + "\n")
        f.close()
        torch.save(model, self.folder_path + "/model.pth")
        f = open(total_rewards_file + str(hyperparams["num time steps"]) + '.txt', "a")
        # np.savetxt(total_rewards_file + str(hyperparams["num time steps"]) + '.txt', self.rewards, delimiter =",")
        np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)
        f.write(str(self.rewards) + "\n")
        np.save(self.folder_path + '/rewards.npy', np.array(self.rewards))


import numpy
def generate_trajectory(env, model : CombinedNN, render_last_step: bool=True, logger : Logger = None,  ):
    episode = []
    s = env.reset()
    probs, value = model(torch.tensor(s))
    a = probs.sample()
    done = False
    tot_reward = 0
    time_steps = 0
    frames = []
    while not done:
        with torch.no_grad():
            s_p, r, done, info = env.step(a.item())
            a = probs.sample()
            episode.append((s, a, r, probs.log_prob(a), value))
            s = s_p
            probs, value = model(torch.tensor(s_p))
            if logger.reached_max and logger.should_record_frames and not logger.already_recorded:
                frames.append(env.render(mode='rgb_array'))
            if done and render_last_step:
                env.render()
            tot_reward += r
            time_steps += 1
    if len(frames) >= 499 and tot_reward >= 499.0 and logger.should_record_frames and not logger.already_recorded:
        logger.frames = frames
        logger.record_frames()
    logger.add_trajectory_reward(tot_reward, time_steps)
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
import time
def main():

    env = gym.make("CartPole-v1")
    lr = 1e-2 # .01
    gamma = .99
    lmbda = .95
    epsilon = .2

    NUM_TIME_STEPS = 500000
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
        "num time steps": NUM_TIME_STEPS,
        "num epochs": NUM_EPOCHS,
        "num trajectories" : NUM_TRAJECTORIES
    }


    model.train()

    start = time.time()
    logger = Logger(NUM_TRAJECTORIES, NUM_TIME_STEPS, record_frames=True)
    model_optim = torch.optim.Adam(model.parameters(),  lr=lr)
    time_step = 0
    while time_step < NUM_TIME_STEPS:
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
        print(str(time_step) + "/" + str(NUM_TIME_STEPS))
        time_step = logger.time_step
        # logger.output_training_step()
        # logger.new_training()
        logger.average_rewards()
    
    logger.save(hyperparams, model)
    end = time.time()
    print(end - start)



if __name__ == "__main__":
    main()