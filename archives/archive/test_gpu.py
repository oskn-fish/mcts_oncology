from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from ray.tune.registry import  register_env
# from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
import gymnasium as gym

from alpha_zero_bayes.models.custom_torch_models import DenseModelWithPrediction
import os
from ray.rllib.algorithms.algorithm import Algorithm
import time

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.air import session

env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25, "early_stop":False}
mcts_config = {
            "puct_coefficient": 1.0,
            "num_simulations": 50,
            "temperature": 1,
            "dirichlet_epsilon": 0.0,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        }


config = AlphaZeroConfig().training(sgd_minibatch_size=32, ranked_rewards={"enable": False}, model={"custom_model":DenseModelWithPrediction}).resources(num_gpus=1).rollouts(num_rollout_workers=0).environment(env=Bayes_RLEEnv, env_config=env_config, disable_env_checking=True).offline_data(output="./offline_data").training(mcts_config = mcts_config)

algo = config.build()


import pandas as pd
# print(f"checkpointpath: {default_checkpoint_path}")
N = 3000
results = []
episode_data = []
start_time = time.time()

for i in range(1,N+1):
    result = algo.train()
    results.append(result)
    episode = {'n': i,
               'episode_reward_min':  result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max':  result['episode_reward_max'],
               'episode_len_mean':    result['episode_len_mean']}
    episode_data.append(episode)
    
    print(f'{i:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
    
end_time = time.time()

print('time spent: ' + str(end_time - start_time))

df = pd.DataFrame(data=episode_data)
df.to_csv('result_learn_RLE.csv', index=False)
    