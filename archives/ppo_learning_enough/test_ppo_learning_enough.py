import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import (
    RLEEnv_expectation_negative_reward_penalty_continue,
)

from ppo_penalty.ppo import PPO
from ppo_penalty.ppo import PPOConfig


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
import math
from ray import train
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

"""
バイアス取り除いていることに注意．
"""


config = PPOConfig()
config = config.training(gamma=1.0)
config = config.framework("torch")

# NUM_ALL_CPUS = 50
# NUM_ALL_GPUS = 3
# NUM_PARALLEL_EXPERIMENTS = 10

# num_rollout_workers = math.floor(NUM_ALL_CPUS / NUM_PARALLEL_EXPERIMENTS)
# num_gpus = NUM_ALL_GPUS / NUM_PARALLEL_EXPERIMENTS
config = config.rollouts(num_rollout_workers=4)
config = config.resources(num_gpus=1)

config = config.training(num_sgd_iter=20)
config = config.resources(num_cpus_per_worker=1)
config = config.training(sgd_minibatch_size=200)
config = config.training(train_batch_size=10000)
config = config.rl_module(_enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)
config = config.training(model={"custom_model": FullyConnectedNetwork})
config = config.reporting(keep_per_episode_custom_metrics=True)

env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": 10,
    "penalty_abs": 0.06513854285023316
    # "penalty_abs": tune.grid_search([i*1e-3 for i in range(1, 10)])
}

config = config.environment(env=RLEEnv_expectation_negative_reward_penalty_continue, env_config=env_config)


algo = config.build()

N = 1000000
results = []
path_checkpoint = "/home/chinen/esc_mcts/ppo_learning_enough/checkpoints"

for iteration in range(1,N+1):
    result = algo.train()
    results.append(result)
    episode = {'n': iteration,
               'episode_reward_min':  result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max':  result['episode_reward_max'],
               'episode_len_mean':    result['episode_len_mean']}
    # episode_data.append(episode)
    print(f'{iteration:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}/{result["episode_len_mean"]:8.4f}')
    if iteration%500 == 0:
        algo.save(path_checkpoint)
    
    