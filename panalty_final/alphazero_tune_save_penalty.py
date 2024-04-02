
# from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import RLEEnv_expectation_negative_reward_penalty_continue
from ray.tune.registry import  register_env

from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
from alpha_zero_bayes.bayes_alpha_zero import AlphaZero
import gymnasium as gym

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
# from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
from alpha_zero_bayes.models.ppo_mcts_custom_torch_models import FullyConnectedNetwork
import os
from ray.rllib.algorithms.algorithm import Algorithm
import time


from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.air import session

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray import train

env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": 10,
    "penalty_abs": tune.grid_search([0.06513854285023316, 0.05721842600035547, 0.04993262476736769, 0.043187026441359, 0.03690702464285426, 0.031032475286063992, 0.02551418439839037, 0.020311401357501223, 0.015389990790009425, 0])
    # "penalty_abs": tune.grid_search([i*1e-3 for i in range(1, 10)])
}
mcts_config = {
            "puct_coefficient": 1.0,
            "num_simulations": 400,
            "temperature": 1,
            "dirichlet_epsilon": 0.0,
            "dirichlet_noise": 0.01,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": False,
            "temp_threshold": 10000,
            "use_Q": False,
            "mcts_action": True
        }

all_cpus = 100
num_experiments = 10
import math
num_rollout_workes = math.floor(all_cpus/num_experiments)
all_gpus = 2
num_gpus = all_gpus/num_experiments


config = AlphaZeroConfig().training(sgd_minibatch_size=200, ranked_rewards={"enable": False}, model={"custom_model":FullyConnectedNetwork}, lr=5e-5,num_sgd_iter=20, epsilon=0, train_batch_size=10000).resources(num_gpus=num_gpus).rollouts(num_rollout_workers=num_rollout_workes).environment(env=RLEEnv_expectation_negative_reward_penalty_continue, env_config=env_config, disable_env_checking=True).offline_data(output="./offline_data").training(mcts_config = mcts_config)

algo = config.build()


config = config.environment(
    env=RLEEnv_expectation_negative_reward_penalty_continue,
    env_config=env_config,
)

config.reporting(keep_per_episode_custom_metrics=True)

scheduler = ASHAScheduler(
    metric="episode_reward_mean",
    mode="max",
    max_t=200,
    grace_period=200,
    reduction_factor=2,
)

tuner = tune.Tuner(
     AlphaZero,
    param_space=config,
    run_config=air.RunConfig(
         checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=True
        )
        
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,  # searchalgはデフォルトがgrid search
        # max_concurrent_trials=4
    ),
)

result = tuner.fit()


    