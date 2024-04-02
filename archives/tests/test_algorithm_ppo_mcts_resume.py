import time
import numpy as np
import pandas as pd
import gymnasium as gym
import ray

# from ray.rllib.agents.ppo   import PPOTrainer, DEFAULT_CONFIG
# from ray.rllib.algorithms.ppo   import PPOConfig
# from ray.rllib.algorithms import ppo
from ppo_mcts.ppo import PPOConfig

# from test_ppo_220 import PPOConfig


import RLE

from ray.tune.registry import register_env

# from RLE.envs.RLEEnv import RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)

from alpha_zero_bayes.models.ppo_mcts_custom_torch_models import FullyConnectedNetwork

ENV_NAME = "RLE-v0"
CHECKPOINT_PATH = "PPO_mcts_checkpoint_resume"

from ray.rllib.algorithms import Algorithm

# ray.init(num_rollout_woker=0)
agent = Algorithm.from_checkpoint("PPO_mcts_checkpoint")


N = 3000
results = []
episode_data = []
start_time = time.time()

for n in range(1, N + 1):
    result = agent.train()
    results.append(result)
    episode = {
        "n": n,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"],
        "episode_reward_max": result["episode_reward_max"],
        "episode_len_mean": result["episode_len_mean"],
    }
    episode_data.append(episode)
    print(
        f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}/{result["episode_len_mean"]:8.4f}'
    )
    # if n >= 1000 and n % 500 == 0:
    checkpoint_path = agent.save(CHECKPOINT_PATH)
    print(checkpoint_path)

end_time = time.time()
print("time spent: " + str(end_time - start_time))

df = pd.DataFrame(data=episode_data)
df.to_csv("PPO_result_learn_RLE.csv", index=False)

# import matplotlib.pyplot as plt


ray.shutdown()
