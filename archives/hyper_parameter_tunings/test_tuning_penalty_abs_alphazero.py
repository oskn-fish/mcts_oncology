# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
# %%
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import (
    RLEEnv_expectation_negative_reward_penalty_continue,
)
from alpha_zero_bayes.bayes_alpha_zero import AlphaZero

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
# from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
# from alpha_zero_bayes.models.custom_torch_models import DenseModel
from alpha_zero_bayes.models.ppo_mcts_custom_torch_models import FullyConnectedNetwork
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig

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


def env_creator(env_config):
    return RLEEnv_expectation_negative_reward_penalty_continue(env_config)


register_env("RLEEnv_expectation_negative_reward_penalty_continue", env_creator)

# env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25, "DLT_max":7, "penalty_abs": tune.grid_search([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])}
env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": 10,
    "penalty_abs": tune.grid_search([0.2834043767146469, 0.2203114013575012, 0.18340437671464696, 0.15721842600035546, 0.13690702464285426, 0.12031140135750121, 0.10628000179850475, 0.09412545064320973, 0.08340437671464696]),
}
mcts_config = {
    "puct_coefficient": 1.0,
    "num_simulations": 400,
    "temperature": 0.4,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": True,
    "add_dirichlet_noise": False,
    "temp_threshold": float("inf"),
    "use_Q": False,
    "mcts_action": True,
}

sgd_minibatch_size = 200
replay_buffer_config = {
    "type": "ReplayBuffer",
    # Size of the replay buffer in batches (not timesteps!).
    "capacity": 163840,
    # "capacity": 4000,
    # Choosing `fragments` here makes it so that the buffer stores entire
    # batches, instead of sequences, pisodes or timesteps.
    # "storage_unit": "fragments",
    "storage_unit": "timesteps",
}

lr = 5e-5
# lr_schedule =
l2_coeff = 0
num_sgd_iter = 20
epsilon = 1e-6
train_batch_size = 10000


# hardware ### CHANGE HERE ####
TOTAL_CPUS = 120
# TOTAL_GPUS = 2

# settings
NUM_PARALLEL_EXPERIMENTS = 7
# NUM_PARALLEL = len(LIST_GRID)


num_rollout_workers = math.floor(
    (TOTAL_CPUS - NUM_PARALLEL_EXPERIMENTS) / NUM_PARALLEL_EXPERIMENTS
)
# num_gpus = TOTAL_GPUS/NUM_PARALLEL_EXPERIMENTS
num_gpus_for_optimization = 2
num_gpus = num_gpus_for_optimization / NUM_PARALLEL_EXPERIMENTS

num_gpus_per_worker = 0  # (TOTAL_GPUS-num_gpus_for_optimization)/(NUM_PARALLEL_EXPERIMENTS*num_rollout_workers)


# TODO: model specification
config = (
    AlphaZeroConfig()
    .training(
        ranked_rewards={"enable": False},
        model={"custom_model": FullyConnectedNetwork},
        l2_coeff=l2_coeff,
        epsilon=epsilon,
        sgd_minibatch_size=sgd_minibatch_size,
        num_sgd_iter=num_sgd_iter,
        replay_buffer_config=replay_buffer_config,
        lr=lr,
        train_batch_size=train_batch_size,
    )
    .resources(num_gpus=num_gpus, num_gpus_per_worker=num_gpus_per_worker)
    .rollouts(num_rollout_workers=num_rollout_workers)
    .environment(
        env=RLEEnv_expectation_negative_reward_penalty_continue,
        env_config=env_config,
        disable_env_checking=False,
    )
    .offline_data(output="./offline_data_expectation")
    .training(mcts_config=mcts_config)
    # .reporting(keep_per_episode_custom_metrics=True)
)
# optuna_search = OptunaSearch()

ray.init()

scheduler = ASHAScheduler(
    metric="episode_reward_mean",
    mode="max",
    max_t=500,
    grace_period=500,
    reduction_factor=2,
)

tuner = tune.Tuner(
    AlphaZero,
    param_space=config,
    run_config=air.RunConfig(
        callbacks=[
            WandbLoggerCallback(
                project=os.path.splitext(os.path.basename(__file__))[0],
                api_key_file=os.path.join(
                    os.path.dirname(__file__), "wandb_api_key.txt"
                ),
            )
        ],
        # stop = {"training_iteration": 100}
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,  # searchalgはデフォルトがgrid search
        # max_concurrent_trials=4
    ),
)
result = tuner.fit()
df_result = result.get_dataframe()
print(df_result)
pickle.dump(df_result, open("df_result.dump", "wb"))
# %%
# print(df_result)
# pickle.dump(result, open("fit.dump", "wb"))
# #%%
# print(result)
# # algo = config.build()
