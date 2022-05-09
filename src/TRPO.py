# pseudocode for implementation taken from : https://spinningup.openai.com/en/latest/algorithms/trpo.html
# more pseudocode https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a

'''
What do I do need to do
How can i give this project structure

okay, i kind of figured out what TRPO does
'''

'''
My pseudocode

input: initial policy parameters "theta" and initial value function "V". both will be Torch neural networks

for k = 0, 1, 2, ... num_trajs do
    
    1. generate trajectories T_k using current policy pi_theta_k...parameterized policy
    2. Estimate advantages of the trajectory using the current policy. will just use A(a,s) = reward_t + gamma * V(s_t+1) - V(s_t). **we will need to save the rewards we generate from trajectory**
    3. Form sample estimates for policy gradient g_k using advantage estimates
        3a. g_k = gradient_theta of L_theta_k (theta)
        3b. recall the L is the expected discounted difference in advantage between two policies using importan sampling. use old advantage * importance ratio. question: where does the theta (not theta_k) come from?
    4. Form sample estimates for KL-divergence Hessian-vector product function f(v) = H_k * v
        4a. we estimate the hessian product by computing the second gradient of the KL divergence
        4b. i am not sure how this works in practice
        4c. output = H_k
    5. Use Conjugate Gradient to take the two estimates and compute the inverse H_k * g_k
        5a. We use n steps here, where n is the number of dimensions (I assume this is relative to the number of policy parameters)
    6. We estimate the proposed step (DELTA_k)
        6a. DELTA_k = sqrt((2*delta)/transpose(x_k) * H_k * x_k) * x_k
        6b. can do this with torch operations most likely
    7. We then verify that this step DELTA_k is valid
        7a. easy part. backtracking ine search until we find something that validates:
        7b. KL-divergence(theta || theta_k) <= delta
        7c. L_theta_k (theta) >= 0
        7d. result: theta_k+1 = theta_k + alpha^j * DELTA_k

end for :)

questions:
    - where does theta (not theta_k) come from? how are they different
        -ans: it is just any policy that we modify from theta_k to potentially find the right theta_k+1.
    - how will this work with TF? I need to figure out what the loss is here. I imagine its just literally just the update rule so Loss = alpha^j * delta_k  <- want to minimize this so eventually theta_k+1 == theta_k !
'''




# step 1, generate trajectories
'''
As input, we will need:
    - the gym environment
    - a policy "pi" which takes in the state and outputs an action
As output, we return:
    - the trajectory of (s, a, r, a_prob) tuples
'''
import numpy
def generate_trajectory(env, pi, render_last_step: bool=True ):
    episode = []
    s = env.reset()
    a, a_prob = pi(torch.tensor(s))
    done = False
    
    while not done:
        s_p, r, done, info = env.step(torch.argmax(a).numpy())
        episode.append((s, a, r, a_prob))
        s = s_p
        a, a_prob = pi(torch.tensor(s_p))
        if done and render_last_step:
            env.render()

    return episode

# step 2, estimate the advantage of the trajectory
'''
Note: Advantage = A(a, s) = sum(r_t + gamma^(t+1) * V(s_t+1) - V(s_t)) where t = 0,1,2,...,T-1, and T = len(trajectory)
As input, we will need:
    - the trajectory, list of (s, a, r) tuples
    - gamma (discount)
    - value function V which takes in the state and outputs a scalar value
As output, we return:
    - advantage estimate per time step
'''
from typing import List
def advantage_estimates(trajectory, gamma : float, V) -> List[float]:
    advantages = []
    for t in range(0, len(trajectory)-1):
        s_t, a_t, r_t, _ = trajectory[t]
        s_t_1, _, _, _ = trajectory[t+1]
        advantages.append(r_t + gamma * V(torch.tensor(s_t_1)) - V(torch.tensor(s_t)))
    return advantages



def get_kl(pi, trajectory):
    kl = None
    for step in trajectory:
        s, _, _, action_probs = step
        mean1 = action_probs.mean()
        mean0 = torch.tensor(mean1)
        log_std1 = torch.log(action_probs)
        log_std0 = torch.tensor(log_std1)
        std1 = torch.tensor(torch.exp(log_std1))
        std0 = torch.tensor(std1)
        if kl == None:
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        else:
            kl.cat(log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5)
    return kl.sum(1, keepdim=True)


# step 3, sample estimates for objective L_theta_k and KL-Divergence constraint H using advantage estimates
'''
Single path sample estimates for L_theta_k and H
Equations:
    - g_k = gradient_theta(L_theta_k(theta))
    - L_theta_k = sum(discounted future rewards -> just use advantage), where s_0~p_0, a~q, q(a|s) = policy_theta_k(a|s)
    - H = FIM = sum(KL_divergence(pi_theta_old(*|s_n) || pi_theta(*|s_n)))
Inputs:
    - policy_theta_k
    - DONT need action distribution (for single path, this will be equivalent to our policy)
    - number of samples
    - to pass along
        - env (calling generate trajectory)
        - V (calling advantage estimate)
        - gamma (calling advantage estimate)
Outputs:
    - estimated objective L_theta_k(theta)
    - estimated constraint KL divergence H 
'''
import torch
def single_path_sample_estimator(pi, env, V, gamma, num_samples: int):
    # set of trajectories D_k
    D_k = []
    # set of Advantages
    A_k = []
    for i in range(num_samples):
        trajectory = generate_trajectory(env, pi)
        D_k.append(trajectory)
        A_k.append(advantage_estimates(trajectory, gamma, V))
    # estimate policy gradient and KL_divergence
    g_k = None
    for traj_idx in range(len(D_k)):
        episode = D_k[traj_idx]
        adv = A_k[traj_idx]
        for t in range(len(episode)-1):
            s, a, r, a_prob = episode[t]

            # print(torch.autograd.grad(a_prob, pi.parameters(),create_graph=True))
            # print(a)
            # print(adv[0])
            grads = torch.autograd.grad(a_prob[a], pi.parameters(),create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            if g_k == None:
                g_k = torch.zeros(flat_grad_kl.shape)
            g_k += torch.mul(flat_grad_kl, adv[t].item())
            # print(get_kl(pi, episode))
            # grads = torch.autograd.grad(kl_v, model.parameters())


           
    return g_k/num_samples

'''
Runner
'''
# first, let us create our inputs
import gym 
from nn import PolicyNN, ValueNN

# lets get our environment
env = gym.make("CartPole-v1")

# Policy parameters (NN) = "theta". 2 hidden layers w/ 32 nodes. softmax output for each action
theta = PolicyNN(
    env.observation_space.shape[0],
    2,
    32,
    env.action_space.n
    )

# Value function parameters (NN) = "V". outputs value of state (scalar)
V = ValueNN(
    env.observation_space.shape[0],
    2,
    32,
    1
)
GAMMA = 1
NUM_SAMPLES = 10
for _ in range(1):
    print(single_path_sample_estimator(theta, env, V, GAMMA, NUM_SAMPLES))



