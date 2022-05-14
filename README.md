# Comparing Policy Gradient Methods in the CartPole Domain
### Authors: Joseph Muffoletto, Haroon Mushtaq
### CS394R Prof. Stone, Prof. Niekum
---
# Run Instructions

## Prerequisites

We used Python3.8.9. Due to the sheer number of packages required, we recommend that you use the same version. To install the necessary requirements, run the following command:

pip3 install -r requirements.txt

## REINFORCE

Files for REINFORCE:
- src/reinforce.py - core algorithm and models
- src/reinforce_runner.py - wrapper funcs which runs both reinforce and reinforce w/ baseline
- src/run.py - runs entire reinforce code

To run:

cd src
python3 run.py 

Outputs the rewards array to ./data/total_rewards_RNOB-{NUM_TIMESTEPS}.txt or ./data/total_rewards_RWB-{NUM_TIMESTEPS}.txt for REINFORCE or REINFORCE w/ Baseline respectively. NUM_TIMESTEPS is a value set in src/reinforce_runner.py

## TRPO

Files for TRPO implementation can be found in pytorch-trpo

To run:

cd pytorch-trpo
python3 main.py --env-name "CartPole-v1"

Outputs the rewards array to ./data/total_rewards_TRPO-{NUM_TIMESTEPS}.txt . NUM_TIMESTEPS is a value set in pytorch-trpo/main.py or through the --training-steps cmd line argument

**NOTE: IMPLEMENTATION MODIFIED FROM HERE: https://github.com/ikostrikov/pytorch-trpo**

Our unfinished TRPO implementation can be found in src/trpo.py. 

## PPO

The file for PPO is src/ppo.py

To run:

cd src
python3 ppo.py

This will output the rewards array to ./data/total_rewards-{NUM_TIMESTEPS}.txt where NUM_TIMESTEPS is a tunable parameter in ppo.py

## Figures

The code we used for generating graphs can be found in src/graph_lib.py

To reproduce the graph found in our report, simply run the following:

cd src
python3 graph_lib.py

Code for generating gifs from gym environment : https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553 

### Comparison plot of tested policy gradient methods

![](data/figures/combined_plot.png)

### Example max reward run on CartPole using PPO trained model

![Max reward run using PPO](data/videos/PPO.gif)