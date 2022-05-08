import gym
import torch
from nn import ValueNN, PolicyNN, CombinedNN

gym.make("CartPole-v1")

import numpy
def generate_trajectory(env, model : CombinedNN, render_last_step: bool=True ):
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
    
    print(tot_reward)
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
    # assert len(trajectories) == len(adv_tensors)

    size_of_tensors = sum([len(traj) for traj in trajectories])
    new_log_probs = torch.zeros(size_of_tensors)
    new_values = torch.zeros(size_of_tensors)

    tensor_idx = 0
    for i, adv in enumerate(trajectories):
        for j, step in enumerate(adv):
            state = torch.tensor(trajectories[i][j][0])
            new_probs, new_value = model(state)
            new_log_probs[tensor_idx] = new_probs.log_prob(trajectories[i][j][1])
            new_values[tensor_idx] = (new_value)
            tensor_idx += 1

    norm_sampled_adv = (adv_tensors - adv_tensors.mean()) / (adv_tensors.std() + 1e-8)


    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(min=1.0 - epsilon, max=1.0 + epsilon)
    policy_objective = torch.min(ratio * norm_sampled_adv, clipped_ratio * norm_sampled_adv)
    # print(policy_objective)
    policy_objective = policy_objective.mean()
    # entropy_bonus = entropies.mean()

    # calculate value parameter delta

    # convert values to tensors
    # new_values = torch.tensor(new_values)
    # old_values = torch.tensor(old_values)

    clipped_value = old_values + (new_values - old_values).clamp(min=-epsilon, max=epsilon)
    # print(new_values-old_values)
    value_loss = torch.max((new_values - sampled_returns) ** 2, (clipped_value - sampled_returns) ** 2)
    value_loss = .5 * value_loss.mean()

    # cp_value = torch.tensor(value_loss)
    # policy_objective =  -(policy_objective - 0.5 * value_loss)
    # loss = -(policy_objective - 0.5 * value_loss + .01*entropy_bonus)
    loss = -(policy_objective - 0.5 * value_loss)

    return loss

def main():

    env = gym.make("CartPole-v1")
    lr = 1e-4
    N = 10
    gamma = .99
    lmbda = .95
    epsilon = .2

    # pi = PolicyNN(
    #     env.observation_space.shape[0],
    #     2,
    #     32,
    #     env.action_space.n
    # )
    # V = ValueNN(
    #     env.observation_space.shape[0],
    #     2,
    #     32,
    #     1
    # )
    model = CombinedNN(
        env.observation_space.shape[0],
        2,
        32,
        env.action_space.n
    )
    num_epochs = 10000
    # policy_optim = torch.optim.Adam(pi.parameters(), lr=lr)
    # value_optim = torch.optim.Adam(V.parameters(), lr=lr)
    model_optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch_num in range(num_epochs):
        trajectories = []
        advantages = []
        values = []
        # pi.train()
        # V.train()
        for i in range(N):
            traj = generate_trajectory(env, model, True)
            trajectories.append(traj)
            adv, vals = advantage_estimates(traj, gamma, lmbda, model)
            advantages.append(adv)
            values.append(vals)

        size_of_tensors = sum([len(traj) for traj in trajectories])
        # entropies = torch.zeros(size_of_tensors)
        sampled_returns = torch.zeros(size_of_tensors)
        old_values = torch.zeros(size_of_tensors)
        # adv_tensors = torch.zeros(size_of_tensors)
        adv_tensors = torch.empty(0)
        # old_log_probs = torch.empty(0, requires_grad=True)
        old_log_probs = torch.zeros(size_of_tensors)
        tensor_idx = 0
        for i, adv in enumerate(advantages):
            for j, step in enumerate(adv):
                with torch.no_grad():
                    sampled_returns[tensor_idx]=(values[i][j] + adv[j]) 
                    # print(old_log_probs)
                    # print(trajectories[i][j][3])
                    # old_log_probs = torch.cat((old_log_probs,trajectories[i][j][3]))
                    old_log_probs[tensor_idx] = trajectories[i][j][3]
                    # entropies[tensor_idx] = new_probs.entropy()
                    old_values[tensor_idx] = (trajectories[i][j][4])
                    tensor_idx += 1

            adv_tensors = torch.cat((adv_tensors, adv))

        for i in range(3):
            
            loss = calc_loss(trajectories, old_log_probs, adv_tensors, old_values, sampled_returns, epsilon, model)
            # value_loss = value_loss ** 2
            # policy_optim.zero_grad()
            # value_optim.zero_grad()
            model_optim.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # for p 
            # policy_reward.backward()
            # x = policy_reward.grad_fn.next_functions
            # while x:
            #     print(x)
            #     x = x[1][0].next_functions
            # value_loss.backward()
            # for p in model.parameters():
            #     print(p)

            # policy_optim.step()
            # value_optim.step()
            model_optim.step()
            # print(policy_reward)
            print(loss)
            # print("duh")


if __name__ == "__main__":
    main()