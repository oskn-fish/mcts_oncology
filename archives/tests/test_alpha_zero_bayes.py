from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
from ray.tune.registry import  register_env
# from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
import gymnasium as gym

import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
import os
from ray.rllib.algorithms.algorithm import Algorithm
# from ray.rllib.models.catalog import ModelCatalog

# ModelCatalog.register_custom_model("dense_model", models.DenseModel)    

def env_creator(env_config):
    rleenv = gym.make("RLE-v0", config=env_config)
    return RLEEnv4Alphazero(rleenv)

register_env("RLEEnv4AlphaZero", env_creator)

default_checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint")
env_config = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}

if os.path.isdir(default_checkpoint_path):
    algo = Algorithm.from_checkpoint(default_checkpoint_path)
else:
    # TODO: model specification
    config = AlphaZeroConfig().training(sgd_minibatch_size=256, ranked_rewards={"enable": False}, model={"custom_model":models.DenseModel}).resources(num_gpus=0).rollouts(num_rollout_workers=1).environment("RLEEnv4AlphaZero", env_config=env_config, disable_env_checking=False).offline_data(output="./offline_data").training(mcts_config = {
            "puct_coefficient": 1.0,
            "num_simulations": 5,
            "temperature": 1.5,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        })
    algo = config.build()

for i in range(1, 1001):
    algo.train()
    print(f"train#:{i}")
    if i%100==0:
        algo.save(default_checkpoint_path)