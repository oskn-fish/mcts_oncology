import time
import numpy as np
import pandas as pd
import gymnasium as gym
import ray

# from ray.rllib.agents.ppo   import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms.ppo import PPOConfig

# from ray.rllib.algorithms import ppo
# from ppo_mcts.ppo import PPOConfig

import RLE

from ray.tune.registry import register_env

# from RLE.envs.RLEEnv import RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)

from ray.rllib.algorithms.algorithm import Algorithm

# from alpha_zero_bayes.models.custom_torch_models import DenseModel

ENV_NAME = "RLE-v0"
CHECKPOINT_PATH = "PPO_checkpoint_negative_reward"
# register_env(ENV_NAME, lambda config: RLEEnv(config))

# def env_creator(config):
#     # rleenv = gym.make("RLE-v0", config=env_config)
#     return RLEEnv(config)

# register_env(ENV_NAME, env_creator)

ray.init(ignore_reinit_error=True, log_to_driver=False)

agent = Algorithm.from_checkpoint("/home/chinen/esc_mcts/PPO_checkpoint")
env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "scenario": "random"}
env = RLEEnv_expectation_negative_reward(env_config)

for _ in range(15):
    obs, info = env.reset()
    done = False
    while not done:
        # print(f"{obs}=")
        print(env.DLTs / env.Ns)
        action = agent.compute_single_action(obs)
        print(f"{action=}")
        obs, reward, done, truncated, info = env.step(action)
        # print(f"{reward=}")
        # print()
    print(f"{reward=}")


# N = 3000
# results = []
# episode_data = []
# start_time = time.time()

# for n in range(1, N + 1):
#     result = agent.train()
#     results.append(result)
#     episode = {
#         "n": n,
#         "episode_reward_min": result["episode_reward_min"],
#         "episode_reward_mean": result["episode_reward_mean"],
#         "episode_reward_max": result["episode_reward_max"],
#         "episode_len_mean": result["episode_len_mean"],
#     }
#     episode_data.append(episode)
#     print(
#         f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}/{result["episode_len_mean"]:8.4f}'
#     )
#     if n >= 1000 and n % 500 == 0:
#         checkpoint_path = agent.save(CHECKPOINT_PATH)
#         print(checkpoint_path)

# end_time = time.time()
# print("time spent: " + str(end_time - start_time))

# df = pd.DataFrame(data=episode_data)
# df.to_csv("PPO_result_learn_RLE_negative_reward.csv", index=False)

# import matplotlib.pyplot as plt


# ray.shutdown()
