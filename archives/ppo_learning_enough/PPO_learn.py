
# from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from RLE.env.RLEEnv_expectation_negative_reward import RLEEnv_expectation_negative_reward
from ray.tune.registry import  register_env

from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
import gymnasium as gym

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
# from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
from alpha_zero_bayes.models.ppo_mcts_custom_torch_models import FullyConnectedNetwork
import os
from ray.rllib.algorithms.algorithm import Algorithm
import time


from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.air import session

from ray.rllib.algorithms.ppo import PPOConfig


# env_config = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}
env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
# mcts_config = {
#             "puct_coefficient": 1.0,
#             "num_simulations": 400,
#             "temperature": 1,
#             "dirichlet_epsilon": 0.0,
#             "dirichlet_noise": 0.01,
#             "argmax_tree_policy": False,
#             "add_dirichlet_noise": False,
#             "temp_threshold": 10000,
#             "use_Q": False,
#             "mcts_action": True
#         }


# if os.path.isdir(default_checkpoint_path):
#     algo = Algorithm.from_checkpoint(default_checkpoint_path)
# else:
# TODO: model specification
config = PPOConfig()
# config['seed'] = 123
config = config.debugging(seed=123)
# config['gamma'] = 1.0
config = config.training(gamma=1.0)
# config['framework'] = 'torch'
config = config.framework("torch")
# config['num_workers'] = 4
config = config.rollouts(num_rollout_workers=10)
# config['num_sgd_iter'] = 20
config = config.training(num_sgd_iter=20)
# config['num_cpus_per_worker'] = 1
config = config.resources(num_cpus_per_worker=1)
# config['sgd_minibatch_size'] = 200
config = config.training(sgd_minibatch_size=200)
# config['train_batch_size'] = 10000
config = config.training(train_batch_size=10000)
config = config.training(model={"custom_model": FullyConnectedNetwork})
# config = config.training(model={"custom_model": DenseModel})
config = config.rl_module(_enable_rl_module_api=False)
config = config.training(_enable_learner_api=False)

# config['env_config'] = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}
config = config.environment(
    env=RLEEnv_expectation_negative_reward,
    env_config={"D": 6, "N_cohort": 3, "N_total": 36, "scenario": "random"},
)  # env=ENV_NAME,

config = config.resources(num_gpus=1)

# config = config.reporting()
# agent = PPOTrainer(config, ENV_NAME)
# print(config.to_dict())

# added to avoid ValueError: Your gymnasium.Env's `reset()` method raised an Exception!
# config = config.environment(disable_env_checking=True)
# config = config.environment(ENV_NAME, env_config={'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}, disable_env_checking=True)
agent = config.build()
    


import pandas as pd
# print(f"checkpointpath: {default_checkpoint_path}")
N = 3000
results = []
# episode_data = []
start_time = time.time()
import datetime
import csv
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"ppo_negative_reward")

if not os.path.isfile(os.path.join(CHECKPOINT_PATH, "resutls.csv")):
    with open(os.path.join(CHECKPOINT_PATH, "resutls.csv"), 'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['created '+datetime.datetime.now().strftime('%m月%d日 %H:%M:%S')])
        writer.writerow(['date', 'time', 'content', 'min', 'mean', 'max'])
        

# wandb = setup_wandb(config=mcts_config, api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"), project="alphazero_learn")
for i in range(1,N+1):
    result = agent.train()
    results.append(result)

    # print(datetime.datetime.now().strftime('%m月%d日 %H:%M:%S')+f'{i:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
    with open(os.path.join(CHECKPOINT_PATH, "resutls.csv"), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([str(i), datetime.datetime.now().strftime('%m月%d日 %H:%M:%S'), "Min/Mean/Max", f'{result["episode_reward_min"]:8.4f}', f'{result["episode_reward_mean"]:8.4f}', f'{result["episode_reward_max"]:8.4f}'])
    # if i >= 1000 and i % 500 == 0:
    # checkpoint_path = algo.save(default_checkpoint_path)
    agent.save(os.path.join(CHECKPOINT_PATH, str(i)))
    # print(checkpoint_path)
    # wandb.log(result)
    # df = pd.DataFrame(data=episode_data)
    # df.to_csv('result_learn_RLE.csv', index=False, mode='x') # overwriting
    


end_time = time.time()
print('time spent: ' + str(end_time - start_time))


    