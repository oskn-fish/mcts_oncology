#%%
from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
from ray.tune.registry import  register_env
# from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.rllib.algorithms.ppo import PPOConfig
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
    # config = PPOConfig().training(sgd_minibatch_size=256, ranked_rewards={"enable": False}, model={"custom_model":models.DenseModel}).resources(num_gpus=0).rollouts(num_rollout_workers=1).environment("RLEEnv4AlphaZero", env_config=env_config, disable_env_checking=False)
    config = PPOConfig().rollouts(num_rollout_workers=1).environment("RLEEnv4AlphaZero", env_config=env_config)
    algo = config.build()
    

# simulate actions
# env = env_creator(env_config)
# terminated = truncated = False
# for i in range(10):
#     obs, info = env.reset()
#     while not terminated and not truncated:
#         action = algo.compute_single_action(obs)
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"obs:{obs}, reward:{reward}, terminated:{terminated}, truncated:{truncated}, info:{info}")
    
# for i in range(1, 1001):
#     algo.train()
#     print(f"train#:{i}")
#     if i%100==0:
#         algo.save(default_checkpoint_path)
    
# confirm that the env works fine
import random
# breakpoint()
env = env_creator(env_config)

# rewardは環境上は特に問題なさそう．
for i in range(10):
    obs, info = env.reset()
    terminated = truncated = False
    print(f"\n\nnew episode with info:{info}")
    while not terminated and not truncated:
        action = random.randint(0,3+env_config["D"]-1)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"action:{action}")
        print(f"obs:{obs}, reward:{reward}, terminated:{terminated}, truncated:{truncated}, info:{info}")
# #%%
# obs, info = env.reset()
# #%%
# action = 1
# observation_before, reward, terminated, truncated, info = env.step(action)
# state = env.get_state()

# env2 = env_creator(env_config)
# observation_after = env2.set_state(state)


# # %%
# print(observation_after)
# # %%
