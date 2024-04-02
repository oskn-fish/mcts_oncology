# import time
# import numpy as np
# import pandas as pd
# import gymnasium as gym
# import ray

# # from ray.rllib.agents.ppo   import PPOTrainer, DEFAULT_CONFIG
# # from ray.rllib.algorithms.ppo   import PPOConfig
# # from ray.rllib.algorithms import ppo
# from ppo_mcts.ppo import PPOConfig

# # from test_ppo_220 import PPOConfig


# import RLE

# from ray.tune.registry import register_env

# # from RLE.envs.RLEEnv import RLEEnv
# from RLE.env.RLEEnv_expectation_negative_reward import (
#     RLEEnv_expectation_negative_reward,
# )

# from alpha_zero_bayes.models.ppo_mcts_custom_torch_models import FullyConnectedNetwork

# ENV_NAME = "RLE-v0"
# CHECKPOINT_PATH = "PPO_mcts_checkpoint"
# # register_env(ENV_NAME, lambda config: RLEEnv(config))

# # def env_creator(config):
# #     # rleenv = gym.make("RLE-v0", config=env_config)
# #     return RLEEnv(config)

# # register_env(ENV_NAME, env_creator)

# ray.init(ignore_reinit_error=True, log_to_driver=False)

# # config = DEFAULT_CONFIG.copy()
# config = PPOConfig()
# # config['seed'] = 123
# config = config.debugging(seed=123)
# # config['gamma'] = 1.0
# config = config.training(gamma=1.0)
# # config['framework'] = 'torch'
# config = config.framework("torch")
# # config['num_workers'] = 4
# config = config.rollouts(num_rollout_workers=60)
# # config['num_sgd_iter'] = 20
# config = config.training(num_sgd_iter=20)
# # config['num_cpus_per_worker'] = 1
# config = config.resources(num_cpus_per_worker=1)
# # config['sgd_minibatch_size'] = 200
# config = config.training(sgd_minibatch_size=200)
# # config['train_batch_size'] = 10000
# config = config.training(train_batch_size=10000)
# config = config.rl_module(_enable_rl_module_api=False)
# config = config.training(_enable_learner_api=False)
# config = config.training(model={"custom_model": FullyConnectedNetwork})

# # config = config.training(mcts_on=False)

# # config['env_config'] = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}
# config = config.environment(
#     env=RLEEnv_expectation_negative_reward,
#     env_config={"D": 6, "N_cohort": 3, "N_total": 36, "scenario": "random"},
# )  # env=ENV_NAME,

# mcts_config = {
#     "puct_coefficient": 1.0,
#     "num_simulations": 300,
#     "temperature": 0.4,
#     "dirichlet_epsilon": 0.25,
#     "dirichlet_noise": 0.03,
#     "argmax_tree_policy": True,
#     "add_dirichlet_noise": False,
#     "temp_threshold": float("inf"),
#     "use_Q": False,
#     "mcts_action": False,
# }

# config = config.training(mcts_config=mcts_config)
# config = config.training(mcts_on=True)


# # config = config.reporting()
# # agent = PPOTrainer(config, ENV_NAME)
# # print(config.to_dict())

# # added to avoid ValueError: Your gymnasium.Env's `reset()` method raised an Exception!
# # config = config.environment(disable_env_checking=True)
# # config = config.environment(ENV_NAME, env_config={'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}, disable_env_checking=True)
# agent = config.build()
# # agent = config.build()

# # algo = ppo.PPO(env=ENV_NAME, config={
# #     "env_config": {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'},  # config to pass to env class
# #     "disable_env_checking": True,
# # })

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
#     # if n >= 1000 and n % 500 == 0:
#     checkpoint_path = agent.save(CHECKPOINT_PATH)
#     print(checkpoint_path)

# end_time = time.time()
# print("time spent: " + str(end_time - start_time))

# df = pd.DataFrame(data=episode_data)
# df.to_csv("PPO_result_learn_RLE.csv", index=False)

# # import matplotlib.pyplot as plt


# ray.shutdown()

# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
# %%
from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)
from ppo_mcts.ppo import PPOConfig
from ppo_mcts.ppo import PPO

# from alpha_zero_bayes.models.custom_torch_models import DenseModelWithPrediction
from alpha_zero_bayes.models.ppo_mcts_custom_torch_models import FullyConnectedNetwork

import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import gymnasium as gym
from ray.tune.registry import register_env

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import os
from ray.tune.logger import DEFAULT_LOGGERS

import pickle

# def env_creator(env_config):
#     return RLEEnv_modified(env_config)

# register_env("RLEEnv_modified", env_creator)

# TODO: model specification
# offline dataにより，trajectoryの情報が取得できる
env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25}

"""
バイアス取り除いていることに注意．
"""

mcts_config = {
    # "puct_coefficient": tune.grid_search([0.1, 0.5, 1.0]),
    "puct_coefficient": 1.0,
    "num_simulations": 1000,
    "temperature": 1.0,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": True,
    "add_dirichlet_noise": False,
    "temp_threshold": float("inf"),
    "use_Q": False,
    "mcts_action": True,
}


config = PPOConfig()
# config['seed'] = 123
config = config.debugging(seed=123)
# config['gamma'] = 1.0
config = config.training(gamma=1.0)
# config['framework'] = 'torch'
config = config.framework("torch")
# config['num_workers'] = 4
config = config.rollouts(num_rollout_workers=127)

# config = config.resources(num_gpus=2)

# config['num_sgd_iter'] = 20
config = config.training(num_sgd_iter=20)
# config['num_cpus_per_worker'] = 1
config = config.resources(num_cpus_per_worker=1)
# config['sgd_minibatch_size'] = 200
config = config.training(sgd_minibatch_size=200)
# config['train_batch_size'] = 10000
config = config.training(train_batch_size=10000)
config = config.rl_module(_enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)
config = config.training(model={"custom_model": FullyConnectedNetwork})

config = config.training(mcts_config=mcts_config)
config = config.training(mcts_on=True)
# config = config.training(use_Q=tune.grid_search([True, False]))

config = config.environment(
    env=RLEEnv_expectation_negative_reward,
    env_config={"D": 6, "N_cohort": 3, "N_total": 36, "scenario": "random"},
)

checkpoint_path = "mcts_on=True_use_Q=False_mcts_action=True"
N = 500
results = []
episode_data = []
# start_time = time.time()
agent = config.build()
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
    path = agent.save(checkpoint_path)

# end_time = time.time()
print("time spent: " + str(end_time - start_time))