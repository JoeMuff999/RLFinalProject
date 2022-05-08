import gym
import torch
from nn import ValueNN, PolicyNN

gym.make("CartPole-v1")

import numpy
def generate_trajectory(env, pi : PolicyNN, V : ValueNN, render_last_step: bool=True ):
    episode = []
    s = env.reset()
    a, probs = pi(torch.tensor(s))
    done = False
    tot_reward = 0
    while not done:
        s_p, r, done, info = env.step(torch.argmax(a).numpy())
        value = V(torch.tensor(s_p))
        episode.append((s, a, r, probs.log_prob(a), value))
        s = s_p
        a, probs = pi(torch.tensor(s_p))
        if done and render_last_step:
            env.render()
        tot_reward += r
    # print(tot_reward)
    return episode

from typing import List
def advantage_estimates(trajectory, gamma : float, lmbda, V):
    advantages = [0 for _ in trajectory]
    values = [0 for _ in trajectory]
    prev_adv = 0.0
    for t in range(len(trajectory)-2, -1, -1):
        s_t, a_t, r_t, _, _ = trajectory[t]
        s_t_1, _, _, _, _ = trajectory[t+1]
        curr_val = V(torch.tensor(s_t)) 
        delta = r_t + gamma * V(torch.tensor(s_t_1)) - curr_val
        prev_adv = delta + gamma * lmbda * prev_adv
        values[t] = curr_val
        advantages[t] = prev_adv
    return advantages, values


def calc_loss(trajectories, advantages, values, epsilon : float, pi : PolicyNN, V : ValueNN):
    assert len(trajectories) == len(advantages)
    # old_log_probs = []
    # new_log_probs = []
    entropies = []
    size_of_tensors = sum([len(traj) for traj in trajectories])
    sampled_returns = torch.zeros(size_of_tensors)
    old_values = torch.zeros(size_of_tensors)
    new_values = torch.zeros(size_of_tensors)
    adv_tensors = torch.zeros(size_of_tensors)
    old_log_probs = torch.zeros(size_of_tensors)
    new_log_probs = torch.zeros(size_of_tensors)
    tensor_idx = 0
    for i, adv in enumerate(advantages):
        for j, step in enumerate(adv):
            sampled_returns[tensor_idx]=(values[i][j] + adv[j]) 
            state = torch.tensor(trajectories[i][j][0])
            old_log_probs[tensor_idx] = trajectories[i][j][3]
            a, probs = pi(state)
            new_log_probs[tensor_idx] = probs.log_prob(a)
            # entropies.append(policy_res[1].entropy())
            old_values[tensor_idx] = (trajectories[i][j][4])
            new_values[tensor_idx] = (V(state))
            adv_tensors[tensor_idx] = (adv[j])
            tensor_idx += 1

    # adv_tensors = torch.tensor(adv_tensors, requires_grad=True)
    # print(adv_tensors)
    # sampled_returns = torch.tensor(sampled_returns, requires_grad=True)
    norm_sampled_adv = (adv_tensors - adv_tensors.mean()) / (adv_tensors.std() + 1e-8)
    # print(state)
    # new_log_probs = torch.tensor(new_log_probs)
    # old_log_probs = torch.tensor(old_log_probs)

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

    value_loss = torch.max((new_values - sampled_returns) ** 2, (clipped_value - sampled_returns) ** 2)
    value_loss = .5 * value_loss.mean()

    # cp_value = torch.tensor(value_loss)
    # policy_objective =  -(policy_objective - 0.5 * value_loss)

    return policy_objective, value_loss

def main():

    env = gym.make("CartPole-v1")
    lr = 1e-4
    N = 10
    gamma = .99
    lmbda = .95
    epsilon = .2

    pi = PolicyNN(
        env.observation_space.shape[0],
        2,
        32,
        env.action_space.n
    )
    V = ValueNN(
        env.observation_space.shape[0],
        2,
        32,
        1
    )
    num_epochs = 10000
    policy_optim = torch.optim.Adam(pi.parameters(), lr=lr)
    value_optim = torch.optim.Adam(V.parameters(), lr=lr)
    for epoch_num in range(num_epochs):
        trajectories = []
        advantages = []
        values = []
        pi.train()
        V.train()
        for i in range(N):
            traj = generate_trajectory(env, pi, V, False)
            trajectories.append(traj)
            adv, vals = advantage_estimates(traj, gamma, lmbda, V)
            advantages.append(adv)
            values.append(vals)

        policy_reward, value_loss = calc_loss(trajectories, advantages, values, epsilon, pi, V)
        # value_loss = value_loss ** 2
        policy_reward = -(policy_reward)
        policy_optim.zero_grad()
        value_optim.zero_grad()


        policy_reward.backward()
        # x = policy_reward.grad_fn.next_functions
        # while x:
        #     print(x)
        #     x = x[1][0].next_functions
        value_loss.backward()
        for p in V.parameters():
            print(p.grad)

        policy_optim.step()
        value_optim.step()
        print(policy_reward)
        print(value_loss)
        print("duh")


if __name__ == "__main__":
    main()