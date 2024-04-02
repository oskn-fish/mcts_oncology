from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import (
    RLEEnv_expectation_negative_reward_penalty_continue,
)

# from alpha_zero_bayes.bayes_alpha_zero import AlphaZero
# from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
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

"""
バイアス取り除いていることに注意．
"""

# mcts_config = {
# "puct_coefficient": tune.grid_search([0.1, 0.5, 1.0]),
#     "puct_coefficient": 1.0,
#     "num_simulations": 400,
#     "temperature": 1.0,
#     "dirichlet_epsilon": 0.25,
#     "dirichlet_noise": 0.03,
#     "argmax_tree_policy": True,
#     "add_dirichlet_noise": False,
#     "temp_threshold": float("inf"),
#     "use_Q": False,
#     "mcts_action": True,
# }


config = PPOConfig()
# config['seed'] = 123
config = config.debugging(seed=123)
# config['gamma'] = 1.0
config = config.training(gamma=1.0)
# config['framework'] = 'torch'
config = config.framework("torch")
# config['num_workers'] = 4

NUM_ALL_CPUS = 60
NUM_ALL_GPUS = 0
NUM_PARALLEL_EXPERIMENTS = 9
num_rollout_workers = math.floor(NUM_ALL_CPUS / NUM_PARALLEL_EXPERIMENTS)
num_gpus = NUM_ALL_GPUS / NUM_PARALLEL_EXPERIMENTS

config = config.rollouts(num_rollout_workers=num_rollout_workers)

config = config.resources(num_gpus=num_gpus)

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

# config = config.training(mcts_config=mcts_config)
# config = config.training(ranked_rewards={"enable": False})
# config = config.training(mcts_on=True)
# config = config.training(use_Q=tune.grid_search([True, False]))

env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": tune.grid_search([10, 11, 12, 13, 14, 15]),
    "penalty_abs": 500,
}


config = config.environment(
    env=RLEEnv_expectation_negative_reward_penalty_continue,
    env_config=env_config,
)

config.reporting(keep_per_episode_custom_metrics=True)

"""ここから"""
# config = config.traing(lr_schedule=[[0, 5e-5], [tune.grid_search([])]])


ray.init()

scheduler = ASHAScheduler(
    metric="episode_reward_mean",
    mode="max",
    max_t=5000,
    grace_period=5000,
    reduction_factor=2,
)

tuner = tune.Tuner(
    # AlphaZeroConfig,
    PPO,
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
# ray.init()

# scheduler = ASHAScheduler(
#     metric="episode_reward_mean",
#     mode="max",
#     max_t=10000,
#     grace_period=10000,
#     reduction_factor=2,
# )

# tuner = tune.Tuner(
#     PPO,
#     param_space=config,
#     run_config=air.RunConfig(
#         callbacks=[
#             WandbLoggerCallback(
#                 project=os.path.splitext(os.path.basename(__file__))[0],
#                 api_key_file=os.path.join(
#                     os.path.dirname(__file__), "wandb_api_key.txt"
#                 ),
#             )
#         ],
#         # stop = {"training_iteration": 100}
#     ),
#     tune_config=tune.TuneConfig(
#         scheduler=scheduler,  # searchalgはデフォルトがgrid search
#         # max_concurrent_trials=4
#     ),
# )
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
