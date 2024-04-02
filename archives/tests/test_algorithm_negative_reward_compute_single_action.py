# from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)
from ray.tune.registry import register_env

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


from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import Episode

# agent = Algorithm.from_checkpoint(
#     "/home/chinen/ray_results/AlphaZero_2023-11-23_03-25-33/AlphaZero_RLEEnv_expectation_negative_reward_864e7_00009_9_puct_coefficient=1.0000_2023-11-23_03-25-33/checkpoint_000000"
# )
# agent = Algorithm.from_checkpoint(
#     "/home/chinen/ray_results/AlphaZero_2023-11-26_06-28-05/AlphaZero_RLEEnv_expectation_negative_reward_85345_00002_2_add_dirichlet_noise=True,policy_coeff=0.2000_2023-11-26_06-28-05"
# )

agent = Algorithm.from_checkpoint("/home/chinen/esc_mcts/alphazero_checkpoint")

policy = agent.get_policy(DEFAULT_POLICY_ID)

env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25}
env = RLEEnv_expectation_negative_reward(env_config)

obs, info = env.reset()

episode = Episode(
    # PolicyMap(0, 0),
    PolicyMap(capacity=0),
    lambda _, __: DEFAULT_POLICY_ID,
    lambda: None,
    lambda _: None,
    0,
)
episode.user_data["initial_state"] = env.get_state()

N = 10
# for i in range(1, N + 1):
done = False
while not done:
    print(obs)
    action, _, _ = policy.compute_single_action(obs, episode=episode)
    print(f"{policy.model.compute_priors_and_value(obs)}")
    print(f"{action=}")
    obs, reward, done, truncated, info = env.step(action)
    episode.user_data["on_episode_step_last_obs"] = obs
    episode.length += 1

print(f"{reward=}")
