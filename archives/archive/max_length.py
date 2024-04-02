# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
#%%
from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from alpha_zero_bayes.bayes_alpha_zero import AlphaZero

import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models

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

# from logging import getLogger, DEBUG, StreamHandler
# # logger = getLogger()
# logger = getLogger("pystan")
# logger.propagate=False
# handler = StreamHandler()
# logger.addHandler(handler)

def env_creator(env_config):
    # rleenv = gym.make("RLE-v0", config=env_config)
    return Bayes_RLEEnv(env_config)

register_env("Bayes_RLE", env_creator)

# TODO: model specification
# offline dataにより，trajectoryの情報が取得できる
env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
# config = (AlphaZeroConfig().
#           training(sgd_minibatch_size=256, 
#                    ranked_rewards={"enable": False}, 
#                    model={"custom_model":models.DenseModel},
#                    mcts_config = {
#                         "puct_coefficient": 1.0,
#                         "num_simulations": 100,
#                         "temperature": 0.5,
#                         "dirichlet_epsilon": 0.25,
#                         "dirichlet_noise": 0.03,
#                         "argmax_tree_policy": False,
#                         "add_dirichlet_noise": True,
#                     })
#           .resources(num_gpus=0)
#           .rollouts(num_rollout_workers=0)
#           .environment(env=Bayes_RLEEnv, 
#                        env_config=env_config, 
#                        disable_env_checking=True
#                        )
#           .offline_data(output="./offline_data"))

config = {
    "env": Bayes_RLEEnv,
    "sgd_minibatch_size": 64,
    "ranked_rewards": {"enable": False},
    "model": {"custom_model":models.DenseModel},
    "mcts_config": {
        # "puct_coefficient": tune.grid_search([0.1, 0.5, 1.0]),
        "puct_coefficient": 1.0,
        "num_simulations":50,
        "temperature": tune.grid_search([0.1, 0.5, 1.0]),
        "dirichlet_epsilon": tune.grid_search([0.1, 0.2, 0.3]),
        "dirichlet_noise": tune.grid_search([0.01, 0.03, 0.05]),
        "argmax_tree_policy": False,
        "add_dirichlet_noise": True
    },
    "num_gpus": 0,
    "num_cpus": 1,
    "num_rollout_workers": 4,
    "env": Bayes_RLEEnv,
    "env_config": env_config,
    "disable_env_checking": True,
    # "wandb":{
    #     "project": "RLE-bayes",
    #     "api_key_file": os.path.join(__file__, "wandb_api_key.txt"),
    #     "log_config": True
    # }
    "log_level": "INFO",
    "training_batch_size": 256
}

asha_scheduler = ASHAScheduler(
    time_attr = "training_iteration", # number of train() https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm.Algorithm.training_iteration.html
    metric = "episode_len_mean",
    mode = "max",
    max_t = 50
)

# optuna_search = OptunaSearch()

ray.init()

tuner = tune.Tuner(
    AlphaZero,
    param_space = config,
    tune_config = tune.TuneConfig(
        # num_samples = -1 # do infinite hyper parameter samplings 
        # metric = "episode_reward_mean",
        # mode = "max",
        scheduler = asha_scheduler,
        # search_alg = optuna_search
    ),
    run_config = air.RunConfig(
        callbacks = [WandbLoggerCallback(project="RLEE_Bayes", api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"))]
    )
)
result = tuner.fit()
df_result = result.get_dataframe()
print(df_result)
pickle.dump(df_result, open("df.dump", "wb"))
#%%
print(df_result)
pickle.dump(result, open("fit.dump", "wb"))
#%%
print(result)
# algo = config.build()

