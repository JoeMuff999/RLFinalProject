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



