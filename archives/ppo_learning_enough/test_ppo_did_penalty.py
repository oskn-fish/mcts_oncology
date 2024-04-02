# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
# %%
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import (
    RLEEnv_expectation_negative_reward_penalty_continue,
)

# from ppo_mcts.ppo import PPOConfig
# from ppo_mcts.ppo import PPO
# from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
from ppo_penalty import PPOConfig
from ppo_penalty import PPO

# from ray.rllib.algorithms.ppo import PPOConfig

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
env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": 1,
    "penalty_abs": 100,
}

"""
バイアス取り除いていることに注意．
"""

mcts_config = {
    # "puct_coefficient": tune.grid_search([0.1, 0.5, 1.0]),
    "puct_coefficient": 1.0,
    "num_simulations": 400,
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
config = config.rollouts(num_rollout_workers=0)
# config = config.training(ranked_rewards={"enable": False})

# config = config.resources(num_gpus=2)

# config['num_sgd_iter'] = 20
config = config.training(num_sgd_iter=1)
# config['num_cpus_per_worker'] = 1
config = config.resources(num_cpus_per_worker=1)
# config['sgd_minibatch_size'] = 200
config = config.training(sgd_minibatch_size=200)
# config['train_batch_size'] = 10000
config = config.training(train_batch_size=10000)
config = config.rl_module(_enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)
config = config.training(model={"custom_model": FullyConnectedNetwork})

# config = config.training(mcts_config=mcts_config)
# config = config.training(mcts_on=True)
# config = config.training(use_Q=tune.grid_search([True, False]))

config = config.environment(
    env=RLEEnv_expectation_negative_reward_penalty_continue,
    # env_config={
    #     "D": 6,
    #     "N_cohort": 3,
    #     "N_total": 36,
    #     "scenario": "random",
    #     "DLT_max": 18,
    # }
    # ,
    env_config=env_config,
)

"""ここから"""
# config = config.traing(lr_schedule=[[0, 5e-5], [tune.grid_search([])]])


agent = config.build()

ITERATION = 30
# for _ in range(ITERATION):
result = agent.train()
print(result)

# %%
