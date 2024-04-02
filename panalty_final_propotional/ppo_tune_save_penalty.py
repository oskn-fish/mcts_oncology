import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RLE.env.RLEEnv_expectation_negative_reward_penalty_proprtional import (
    RLEEnv_expectation_negative_reward_penalty_propotional,
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

NUM_ALL_CPUS = 50
NUM_ALL_GPUS = 3
NUM_PARALLEL_EXPERIMENTS = 10

num_rollout_workers = math.floor(NUM_ALL_CPUS / NUM_PARALLEL_EXPERIMENTS)
num_gpus = NUM_ALL_GPUS / NUM_PARALLEL_EXPERIMENTS
config = config.rollouts(num_rollout_workers=num_rollout_workers)
config = config.resources(num_gpus=num_gpus)

config = config.training(num_sgd_iter=20)
config = config.resources(num_cpus_per_worker=1)
config = config.training(sgd_minibatch_size=200)
config = config.training(train_batch_size=10000)
config = config.rl_module(_enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)
config = config.training(model={"custom_model": FullyConnectedNetwork})

# new_list = [penalty/10 for penalty in [0.06513854285023316, 0.05721842600035547, 0.04993262476736769, 0.043187026441359, 0.03690702464285426, 0.031032475286063992, 0.02551418439839037, 0.020311401357501223, 0.015389990790009425, 0]]

new_list = [penalty/10 for penalty in [0.2834043767146469, 0.2203114013575012, 0.18340437671464696, 0.15721842600035546, 0.13690702464285426, 0.12031140135750121, 0.10628000179850475, 0.09412545064320973, 0.08340437671464696, 0.0738140492857085]]

env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": 10,
    # "penalty_coefficient_abs": tune.grid_search([0.06513854285023316, 0.05721842600035547, 0.04993262476736769, 0.043187026441359, 0.03690702464285426, 0.031032475286063992, 0.02551418439839037, 0.020311401357501223, 0.015389990790009425, 0])
    "penalty_coefficient_abs": tune.grid_search(new_list)
    # "penalty_abs": tune.grid_search([i*1e-3 for i in range(1, 10)])
}

 
config = config.environment(
    env=RLEEnv_expectation_negative_reward_penalty_propotional,
    env_config=env_config,
)

config.reporting(keep_per_episode_custom_metrics=True)



# ray.init()

scheduler = ASHAScheduler(
    metric="episode_reward_mean",
    mode="max",
    max_t=8000,
    grace_period=8000,
    reduction_factor=2,
)

tuner = tune.Tuner(
    PPO,
    param_space=config,
    run_config=air.RunConfig(
         checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=True
        ),
        callbacks = [WandbLoggerCallback(project="proposional_penalty_ppo", api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"))],
        
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,  # searchalgはデフォルトがgrid search
        # max_concurrent_trials=4
    ),
)

result = tuner.fit()
# df_result = result.get_dataframe()
# print(df_result)
# pickle.dump(df_result, open("df_result.dump", "wb"))
