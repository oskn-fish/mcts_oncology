# from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from RLE.env.RLEEnv_expectation import RLEEnv_expectation
from ray.tune.registry import  register_env
# from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
import gymnasium as gym

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
from alpha_zero_bayes.models.custom_torch_models import DenseModel
# from alpha_zero_bayes.models.custom_torch_models import DenseModel
import os
from ray.rllib.algorithms.algorithm import Algorithm
import time

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.air import session


env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
mcts_config = {
            "puct_coefficient": 1.0,
            "num_simulations": 300,
            "temperature": .4,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": True,
            "add_dirichlet_noise": False,
            "temp_threshold": float("inf"),
        }

sgd_minibatch_size = 128
# num_sgd_iter = 6
replay_buffer_config = {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 163840,
            # "capacity": 4000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, pisodes or timesteps.
            # "storage_unit": "fragments",
            "storage_unit": "timesteps"
        }
# num_steps_sampled_before_learning_starts = 10240
# lr = 5e-5
lr = 0.001
# train_batch_size = 4096
# epsilon = 1e-12
l2_coeff = 0.0001
# policy_coeff = 0.5

num_sgd_iter=6
# epsilon=1e-12
epsilon = 0.0
num_steps_sampled_before_learning_starts=10240
sgd_minibatch_size=128
train_batch_size=10240
min_sample_timesteps_per_iteration=10240

# TODO: model specification
config = AlphaZeroConfig().training(ranked_rewards={"enable": False}, model={"custom_model":DenseModel}, l2_coeff=l2_coeff, epsilon=epsilon,sgd_minibatch_size=sgd_minibatch_size, num_sgd_iter=num_sgd_iter, replay_buffer_config=replay_buffer_config, num_steps_sampled_before_learning_starts=num_steps_sampled_before_learning_starts, lr=lr, train_batch_size=train_batch_size).resources(num_gpus=1).rollouts(num_rollout_workers=0).environment(env=RLEEnv_expectation, env_config=env_config, disable_env_checking=False).offline_data(output="./offline_data_expectation").training(mcts_config = mcts_config)
# from ray.rllib.examples.env.cartpole_sparse_rewards import CartPoleSparseRewards
# config = AlphaZeroConfig().training(sgd_minibatch_size=256, ranked_rewards={"enable": False}, model={"custom_model":models.DenseModel}).resources(num_gpus=0).rollouts(num_rollout_workers=1).environment(env=CartPoleSparseRewards, disable_env_checking=True).offline_data(output="./offline_data").training(mcts_config = {
#         "puct_coefficient": 1.0,
#         "num_simulations": 100,
#         "temperature": 1.5,
#         "dirichlet_epsilon": 0.25,
#         "dirichlet_noise": 0.03,
#         "argmax_tree_policy": False,
#         "add_dirichlet_noise": True,
#     })
algo = config.build()

import pandas as pd
N = 3000
results = []
episode_data = []
start_time = time.time()
checkpoint_path = "alphazero_checkpoint"

# wandb setup 
# wandb = setup_wandb(config=mcts_config, api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"))
for i in range(1,N+1):
    result = algo.train()
    results.append(result)
    episode = {'n': i,
               'episode_reward_min':  result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max':  result['episode_reward_max'],
               'episode_len_mean':    result['episode_len_mean']}
    episode_data.append(episode)
    
    print(f'{i:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}   ///  {result["episodes_this_iter"]:8.4f}   {result["episode_len_mean"]}')
    
    # if i >= 1000 and i % 500 == 0:
    algo.save(checkpoint_path)
    # print(checkpoint_path)
        
    # wandb
    # wandb.log(result)
end_time = time.time()

print('time spent: ' + str(end_time - start_time))
print(episode_data)

df = pd.DataFrame(data=episode_data)
df.to_csv('result_learn_RLE.csv', index=False)