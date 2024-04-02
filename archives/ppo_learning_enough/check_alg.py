from RLE.env.RLEEnv_expectation import RLEEnv_expectation
from ray.tune.registry import  register_env
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
import gymnasium as gym

from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
import os
from ray.rllib.algorithms.algorithm import Algorithm
import time

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.air import session

env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
mcts_config = {
            "puct_coefficient": 1.0,
            "num_simulations": 100,
            "temperature": 1,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": True,
            "add_dirichlet_noise": True,
        }

sgd_minibatch_size = 128
num_sgd_iter = 6
replay_buffer_config = {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 4096,
            # "capacity": 4000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, pisodes or timesteps.
            # "storage_unit": "fragments",
            "storage_unit": "timesteps"
        }
num_steps_sampled_before_learning_starts = 4096
lr = 5e-5
train_batch_size = 4096
epsilon = 1e-12
l2_coeff = 0.0001


# TODO: model specification
# config = AlphaZeroConfig().training(ranked_rewards={"enable": False}, model={"custom_model":DenseModelWithOneBias}, l2_coeff=l2_coeff, epsilon=epsilon,sgd_minibatch_size=sgd_minibatch_size, num_sgd_iter=num_sgd_iter, replay_buffer_config=replay_buffer_config, num_steps_sampled_before_learning_starts=num_steps_sampled_before_learning_starts, lr=lr, train_batch_size=train_batch_size).resources(num_gpus=1).rollouts(num_rollout_workers=10).environment(env=RLEEnv_expectation, env_config=env_config, disable_env_checking=False).offline_data(output="./offline_data").training(mcts_config = mcts_config)

# algo = Algorithm.from_checkpoint("with_posterior")
from ray.rllib.policy.policy import Policy
policy = Policy.from_checkpoint("with_posterior")


env = gym.make("RLEEnv_expectation", config=env_config)
obs, info =env.reset()
for i in range(10):
    action = policy.compute_single_actions(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    
