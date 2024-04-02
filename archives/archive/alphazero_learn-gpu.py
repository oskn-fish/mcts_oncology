# from RLE.wrapper.RLEEnv4AlphaZero import RLEEnv4Alphazero
from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from ray.tune.registry import  register_env
# from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from alpha_zero_bayes.bayes_alpha_zero import AlphaZeroConfig
import gymnasium as gym

# import ray.rllib.algorithms.alpha_zero.models.custom_torch_models as models
from alpha_zero_bayes.models.custom_torch_models import DenseModelWithOneBias
import os
from ray.rllib.algorithms.algorithm import Algorithm
import time


from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.air import session
# from ray.rllib.models.catalog import ModelCatalog

# ModelCatalog.register_custom_model("dense_model", models.DenseModel)    

# def env_creator(env_config):
#     rleenv = gym.make("RLE-v0", config=env_config)
#     return RLEEnv4Alphazero(rleenv)

# register_env("Bayes_RLE", env_creator)


# from logging import getLogger, DEBUG, StreamHandler
# logger = getLogger()
# logger.setLevel(DEBUG)
# handler = StreamHandler()
# logger.addHandler(handler)

# default_checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint")
"""
self.D        = config["D"]
self.N_cohort = config["N_cohort"]
self.N_total  = config["N_total"]
self.phi = config["phi"]
"""
# env_config = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}
env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
mcts_config = {
            # "puct_coefficient": 1.0,
            # "num_simulations": 100,
            # "temperature": 0.5,
            # "dirichlet_epsilon": 0.25,
            # "dirichlet_noise": 0.03,
            # "argmax_tree_policy": False,
            # "add_dirichlet_noise": True,
            "puct_coefficient": 1.0,
            "num_simulations": 100,
            "temperature": 1,
            "dirichlet_epsilon": 0.0,
            "dirichlet_noise": 0.01,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": False,
        }


# if os.path.isdir(default_checkpoint_path):
#     algo = Algorithm.from_checkpoint(default_checkpoint_path)
# else:
# TODO: model specification
config = AlphaZeroConfig().training(sgd_minibatch_size=32, ranked_rewards={"enable": False}, model={"custom_model":DenseModelWithOneBias}).resources(num_gpus=4).rollouts(num_rollout_workers=4).environment(env=Bayes_RLEEnv, env_config=env_config, disable_env_checking=True).offline_data(output="./offline_data").training(mcts_config = mcts_config)
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
    
"""
(.venv) root@0490d98cffba:~/esc_mcts# /root/esc_mcts/.venv/bin/python /root/esc_mcts/alphazero_learn.py
2023-09-13 15:06:45,587 WARNING deprecation.py:50 -- DeprecationWarning: `DirectStepOptimizer` has been deprecated. This will raise an error in the future!
2023-09-13 15:06:45,872 WARNING deprecation.py:50 -- DeprecationWarning: `rllib/algorithms/alpha_star/` has been deprecated. Use `rllib_contrib/alpha_star/` instead. This will raise an error in the future!
/root/esc_mcts/.venv/lib/python3.10/site-packages/ray/rllib/algorithms/algorithm.py:484: RayDeprecationWarning: This API is deprecated and may be removed in future Ray releases. You could suppress this warning by setting env variable PYTHONWARNINGS="ignore::DeprecationWarning"
`UnifiedLogger` will be removed in Ray 2.7.
return UnifiedLogger(config, logdir, loggers=None)
/root/esc_mcts/.venv/lib/python3.10/site-packages/ray/tune/logger/unified.py:53: RayDeprecationWarning: This API is deprecated and may be removed in future Ray releases. You could suppress this warning by setting env variable PYTHONWARNINGS="ignore::DeprecationWarning"
The `JsonLogger interface is deprecated in favor of the `ray.tune.json.JsonLoggerCallback` interface and will be removed in Ray 2.7.
self._loggers.append(cls(self.config, self.logdir, self.trial))
/root/esc_mcts/.venv/lib/python3.10/site-packages/ray/tune/logger/unified.py:53: RayDeprecationWarning: This API is deprecated and may be removed in future Ray releases. You could suppress this warning by setting env variable PYTHONWARNINGS="ignore::DeprecationWarning"
The `CSVLogger interface is deprecated in favor of the `ray.tune.csv.CSVLoggerCallback` interface and will be removed in Ray 2.7.
self._loggers.append(cls(self.config, self.logdir, self.trial))
/root/esc_mcts/.venv/lib/python3.10/site-packages/ray/tune/logger/unified.py:53: RayDeprecationWarning: This API is deprecated and may be removed in future Ray releases. You could suppress this warning by setting env variable PYTHONWARNINGS="ignore::DeprecationWarning"
The `TBXLogger interface is deprecated in favor of the `ray.tune.tensorboardx.TBXLoggerCallback` interface and will be removed in Ray 2.7.
self._loggers.append(cls(self.config, self.logdir, self.trial))
2023-09-13 15:06:51,488 WARNING services.py:1832 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 54468608 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=10.24gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
2023-09-13 15:06:51,657 INFO worker.py:1621 -- Started a local Ray instance.
"""

import pandas as pd
# print(f"checkpointpath: {default_checkpoint_path}")
N = 30
results = []
# episode_data = []
start_time = time.time()

# wandb = setup_wandb(config=mcts_config, api_key_file=os.path.join(os.path.dirname(__file__), "wandb_api_key.txt"), project="alphazero_learn_gpu")
for i in range(1,N+1):
    result = algo.train()
    results.append(result)
    # episode = {'n': i,
              #  'episode_reward_min':  result['episode_reward_min'],
              #  'episode_reward_mean': result['episode_reward_mean'],
              #  'episode_reward_max':  result['episode_reward_max'],
              #  'episode_len_mean':    result['episode_len_mean']}
    # episode_data.append(episode)
    print(f'{i:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
    # if i >= 1000 and i % 500 == 0:
    # checkpoint_path = algo.save(default_checkpoint_path)
    checkpoint_path = algo.save()
    print(checkpoint_path)
    # wandb.log(result)
    # df = pd.DataFrame(data=episode_data)
    # df.to_csv('result_learn_RLE.csv', index=False, mode='x') # overwriting
    

end_time = time.time()
print('time spent: ' + str(end_time - start_time))


    