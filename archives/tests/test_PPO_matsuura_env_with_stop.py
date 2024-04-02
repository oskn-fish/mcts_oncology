import time
import numpy as np
import pandas as pd
import gymnasium as gym
import ray
# from ray.rllib.agents.ppo   import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms.ppo   import PPOConfig
from ray.rllib.algorithms import ppo
import RLE

from ray.tune.registry import register_env
# from RLE.env.RLEEnv_modified import RLEEnv_modified
from RLE.env.
# from alpha_zero_bayes.models.action_mask_ppo import TorchActionMaskModel

from ray import tune, air
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import os

def env_creator(env_config):
    return RLEEnv_modified(env_config)

register_env("RLEE_modified", env_creator)

# register_env(ENV_NAME, lambda config: RLEEnv(config))

# def env_creator(config):
#     # rleenv = gym.make("RLE-v0", config=env_config)
#     return RLEEnv(config)

# register_env(ENV_NAME, env_creator)

checkpoint_path = "~/PPO_with_stop"

# config = DEFAULT_CONFIG.copy()
config = PPOConfig()
# config['seed'] = 123
config = config.debugging(seed=123)
# config['gamma'] = 1.0
config = config.training(gamma=1.0, model={"custom_model":TorchActionMaskModel})
# config['framework'] = 'torch'
config = config.framework("torch")
# config['num_workers'] = 4
config = config.rollouts(num_rollout_workers=4)
# config['num_sgd_iter'] = 20
config = config.training(num_sgd_iter=20)
# config['num_cpus_per_worker'] = 1 
config = config.resources(num_cpus_per_worker=1)
# config['sgd_minibatch_size'] = 200
config = config.training(sgd_minibatch_size=200)
# config['train_batch_size'] = 10000
config = config.training(train_batch_size=10000)
# config['env_config'] = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}
# config = config.environment(env=Bayes_RLEEnv, env_config={'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'})
config = config.environment(env=RLEEnv_modified, env_config={'D':6, 'N_cohort':3, 'N_total':36, "phi":0.25, "early_stop": True})
# config = config.environment(env_config={'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'})

# ValueError: Cannot use `custom_model` option with RLModule API. `custom_model` is part of the ModelV2 API and Policy API, which are not compatible with the RLModule API 対策
config = config.rl_module( _enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)
# config = config.reporting()


# agent = PPOTrainer(config, ENV_NAME)
# print(config.to_dict())

# added to avoid ValueError: Your gymnasium.Env's `reset()` method raised an Exception!
# config = config.environment(disable_env_checking=True)
# config = config.environment(ENV_NAME, env_config={'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}, disable_env_checking=True)
# agent = config.build()
agent = config.build()

# algo = ppo.PPO(env=ENV_NAME, config={
#     "env_config": {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'},  # config to pass to env class
#     "disable_env_checking": True,
# })

# ray.init(ignore_reinit_error=True, log_to_driver=False)
# tuner = tune.Tuner(
#     # AlphaZero,  
#     "PPO",
#     param_space = config,
#     run_config = air.RunConfig(
#         callbacks = [WandbLoggerCallback(project="PPO_with_matsuura_env", api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"))],
#         stop = {"episode_reward_mean": 1}
#     )
# )
# result = tuner.fit()
# df_result = result.get_dataframe()
# print(df_result)


N = 3000
results = []
episode_data = []
start_time = time.time()

for n in range(1, N+1):
    result = agent.train()
    results.append(result)
    episode = {'n': n,
               'episode_reward_min':  result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max':  result['episode_reward_max'],
               'episode_len_mean':    result['episode_len_mean']}
    episode_data.append(episode)
    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
    if n % 100 == 0:
        agent.save(checkpoint_path)
        print(checkpoint_path)

end_time = time.time()
print('time spent: ' + str(end_time - start_time))

df = pd.DataFrame(data=episode_data)
df.to_csv('result_learn_RLE.csv', index=False)

ray.shutdown()