# %% 
from ray.rllib.algorithms import Algorithm
import gymnasium as gym

# from RLE.envs.RLEEnv import RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)

# path_checkpoint = "./PPO_checkpoint_negative_reward"
path_checkpoint = "ray_results/PPO_2024-01-14_02-42-04/PPO_RLEEnv_expectation_negative_reward_penalty_continue_1058a_00008_8_penalty_abs=0.0834_2024-01-14_02-42-04"
algo = Algorithm.from_checkpoint(path_checkpoint)

env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25, "scenario": "random"}
# env = gym.make(RLEEnv_expectation_negative_reward, env_config)
env = RLEEnv_expectation_negative_reward(env_config)

num_simulations = 100000

DLTs = [0] * (env_config["N_total"] + 1)

for _ in range(num_simulations):
    done = False
    observation, info = env.reset()
    while not done:
        action = algo.compute_single_action(observation)
        observation, reward, done, truncated, info = env.step(action)
    #     print(f"{observation=}")
    #     print(f"{reward=}")
    #     print(f"{done=}")
    #     print(f"{info=}")
    # print(observation)
    num_DLTS = round(observation[-2] * env_config["N_total"])
    DLTs[num_DLTS] = DLTs[num_DLTS] + 1
#%%

import pickle

with open("ndarray_DLTs.pickle", "wb") as f:
    pickle.dump(DLTs, f)

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(DLTs)
plt.savefig("fig_DLTs.png")
