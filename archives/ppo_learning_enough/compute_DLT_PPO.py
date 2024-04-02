# %% 
from ray.rllib.algorithms import Algorithm
import gymnasium as gym

# from RLE.envs.RLEEnv import RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)
from ray.rllib.policy.policy import Policy

# path_checkpoint = "./PPO_checkpoint_negative_reward"
path_checkpoint = "~/ray_results/PPO_2024-01-14_02-42-04/PPO_RLEEnv_expectation_negative_reward_penalty_continue_1058a_00008_8_penalty_abs=0.0834_2024-01-14_02-42-04/checkpoint_000000/policies"
# algo = Algorithm.from_checkpoint(path_checkpoint)
algo = Policy.from_checkpoint(path_checkpoint)

env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25, "scenario": "random"}
# env = gym.make(RLEEnv_expectation_negative_reward, env_config)
env = RLEEnv_expectation_negative_reward(env_config)

num_simulations = 100000

metrics = {"DLT":[0] * (env_config["N_total"] + 1), "penalty_metric":[], "MTD_metric":[]}
# DLTs = [0] * (env_config["N_total"] + 1)

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
    metrics["DLTs"][num_DLTS] = metrics["DLTs"][num_DLTS] + 1
    metrcis["penalty_metric"].append(info["penalty_metric"])
    metrics["MTD_metric"].append(info["MTD_metric"])
#%%

import pickle

with open("dict_DLTs_PPO_2024-01-14_02-42-04.pickle", "wb") as f:
    pickle.dump(DLTs, f)
