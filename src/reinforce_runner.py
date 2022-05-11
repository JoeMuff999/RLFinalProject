from email.mime import base
import numpy as np
from matplotlib import pyplot as plt
import gym 
import sys
import time
from reinforce import REINFORCE, Baseline, PiApproximationWithNN, VApproximationWithNN

def reinforce_tests():
    NUM_TIME_STEPS = 100000
    print("testing REINFORCE without baseline")
    # # Test REINFORCE without baseline
    # wout_baseline_logger = {
    #     "time_steps": NUM_TIME_STEPS,
    #     "render": True, #last step
    #     "frames": []
    # }
    # start = time.time()
    # rewards_no_B = run_reinforce(use_baseline=False, logger=wout_baseline_logger)
    # end = time.time()
    # print("no B " + str(end - start))
    # Test REINFORCE with baseline
    print("testing REINFORCE with baseline")
    with_baseline_logger = {
        "time_steps": NUM_TIME_STEPS,
        "render": True, # last step
        "frames": []
    }
    start = time.time()
    rewards_with_B = run_reinforce(use_baseline=True, logger=with_baseline_logger)
    end = time.time()

    print("w B " + str(end - start))
    
    total_rewards_file_withB = '../data/total_rewards_RWB-'
    total_rewards_file_withoutB = '../data/total_rewards_RNOB-'



    f = open(total_rewards_file_withoutB + str(NUM_TIME_STEPS) + '.txt', "a")
    np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)
    f.write(str(rewards_no_B) + "\n")
    f.close()

    f = open(total_rewards_file_withB + str(NUM_TIME_STEPS) + '.txt', "a")
    np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)
    f.write(str(rewards_with_B) + "\n")
    f.close()
    

def run_reinforce(use_baseline, logger, gym_env_string : str = "CartPole-v1"):
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
        # print(sum(p.numel() for p in baseline.nn.parameters() if p.requires_grad))
    else:
        baseline = Baseline(0.)
    # print(sum(p.numel() for p in pi.nn.parameters() if p.requires_grad))

    return REINFORCE(env,gamma,logger['time_steps'],pi,baseline, logger)
