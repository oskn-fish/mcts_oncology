# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
#%%
from RLE.env.RLEEnv_modified import RLEEnv_modified
from alpha_zero_bayes.bayes_alpha_zero import AlphaZero

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
# from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
from alpha_zero_bayes.models.custom_torch_models import DenseModelWithPrediction
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
import torch

def env_creator(env_config):
    return RLEEnv_modified(env_config)

register_env("RLEEnv_modified", env_creator)

# TODO: model specification
# offline dataにより，trajectoryの情報が取得できる
env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25, "early_stop": False}

"""
バイアス取り除いていることに注意．
"""

mcts_config = {
    # "puct_coefficient": tune.grid_search([0.1, 0.5, 1.0]),
    "puct_coefficient": 1.0,
    "num_simulations":50,
    "temperature": 1.0,
    "dirichlet_epsilon": 0,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": True,
    "add_dirichlet_noise": False
}
# sgd_minibatch_size=128, lr=0.00001, num_sgd_iter=1, 

replay_buffer_config = {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 1000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, episodes or timesteps.
            "storage_unit": "fragments",
        }

epsilon = tune.grid_search([0, 1e-8, 1e-12, torch.finfo(torch.float32).tiny])

config = AlphaZeroConfig().training(ranked_rewards={"enable": False}, model={"custom_model":DenseModelWithPrediction}, l2_coeff=0.0001, replay_buffer_config=replay_buffer_config, epsilon=epsilon).resources(num_gpus=1).rollouts(num_rollout_workers=6).environment(env=RLEEnv_modified, env_config=env_config, disable_env_checking=True).offline_data(output="./offline_data_epsilon").training(mcts_config = mcts_config)
# optuna_search = OptunaSearch()

ray.init()

tuner = tune.Tuner(
    AlphaZero,  
    param_space = config,
    run_config = air.RunConfig(
        callbacks = [WandbLoggerCallback(project="no_early_stop_with_prediction_on_matsuura_epsilon", api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"))],
        stop = {"episode_reward_mean": 1}
    )
)
result = tuner.fit()
df_result = result.get_dataframe()
print(df_result)
pickle.dump(df_result, open("df_result.dump", "wb"))
#%%
# print(df_result)
# pickle.dump(result, open("fit.dump", "wb"))
# #%%
# print(result)
# # algo = config.build()

