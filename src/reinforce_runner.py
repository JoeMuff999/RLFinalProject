from email.mime import base
import numpy as np
from matplotlib import pyplot as plt
import gym 
from reinforce import REINFORCE, Baseline, PiApproximationWithNN, VApproximationWithNN

def reinforce_tests():
    num_iter = 5

    # Test REINFORCE without baseline
    without_baseline = []
    for _ in range(num_iter):
        training_progress = run_reinforce(use_baseline=False)
        without_baseline.append(training_progress)
    without_baseline = np.mean(without_baseline,axis=0)

    # Test REINFORCE with baseline
    with_baseline = []
    for _ in range(num_iter):
        training_progress = run_reinforce(use_baseline=True)
        with_baseline.append(training_progress)
    with_baseline = np.mean(with_baseline,axis=0)

    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(without_baseline)),without_baseline, label='without baseline')
    ax.plot(np.arange(len(with_baseline)),with_baseline, label='with baseline')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()

def run_reinforce(use_baseline, gym_env_string : str = "CartPole-v1"):
    env = gym.make(gym_env_string)
    gamma = 1.
    alpha = 3e-4

    pi = PiApproximationWithNN(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha)
    if use_baseline:
        baseline = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        baseline = Baseline(0.)

    return REINFORCE(env,gamma,100,pi,baseline)
