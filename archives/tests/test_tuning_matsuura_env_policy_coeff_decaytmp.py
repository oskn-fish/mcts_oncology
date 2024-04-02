# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
#%%
from RLE.env.RLEEnv_expectation import RLEEnv_expectation
from alpha_zero_bayes.bayes_alpha_zero import AlphaZero

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
# from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
from alpha_zero_bayes.models.custom_torch_models import DenseModel#WithPrediction
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

def env_creator(env_config):
    return RLEEnv_expectation(env_config)

register_env("RLEEnv_expectation", env_creator)

# TODO: model specification
# offline dataにより，trajectoryの情報が取得できる
env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25, "early_stop": False}

"""
バイアス取り除いていることに注意．
"""

mcts_config = {
    # "puct_coefficient": tune.grid_search([0.1, 0.5, 1.0]),
    "puct_coefficient": 1.0,
    "num_simulations": 100,
    "temperature": 1.0,
    "dirichlet_epsilon": 0,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": True,
    "add_dirichlet_noise": tune.grid_search([True, False]),
    "temp_threshold": float("inf"),
    # "decay_temp":True
}
# sgd_minibatch_size=128, lr=0.00001, num_sgd_iter=1, 

replay_buffer_config = {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 163840,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, episodes or timesteps.
            "storage_unit": "timesteps",
        }

num_sgd_iter=6
epsilon=1e-12
num_steps_sampled_before_learning_starts=10240
sgd_minibatch_size=128
train_batch_size=10240
min_sample_timesteps_per_iteration=10240
policy_coeff = tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5])

config = AlphaZeroConfig().training(ranked_rewards={"enable": False}, model={"custom_model":DenseModel}, l2_coeff=0.001,  num_sgd_iter=num_sgd_iter, epsilon=epsilon, num_steps_sampled_before_learning_starts=num_steps_sampled_before_learning_starts,sgd_minibatch_size=sgd_minibatch_size,replay_buffer_config=replay_buffer_config,train_batch_size=train_batch_size, policy_coeff=policy_coeff).resources(num_gpus=0).rollouts(num_rollout_workers=10).environment(env=RLEEnv_expectation, env_config=env_config, disable_env_checking=True).offline_data(output="./offline_data").training(mcts_config = mcts_config).reporting(min_sample_timesteps_per_iteration=min_sample_timesteps_per_iteration)
# optuna_search = OptunaSearch()

ray.init()

scheduler = ASHAScheduler(
    metric='episode_reward_mean', mode='max', max_t=500,
    grace_period=500, reduction_factor=2)

tuner = tune.Tuner(
    AlphaZero,  
    param_space = config,
    run_config = air.RunConfig(
        callbacks = [WandbLoggerCallback(project=os.path.splitext(os.path.basename(__file__))[0], api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"))],
        # stop = {"training_iteration": 100}
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler, # searchalgはデフォルトがgrid search
        # max_concurrent_trials=4
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

