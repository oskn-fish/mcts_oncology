import time
import numpy as np
import pandas as pd
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig

import RLE

from ray.tune.registry import register_env


from RLE.env.RLEEnv_expectation_negative_reward import (
    RLEEnv_expectation_negative_reward,
)


from ray.tune.schedulers import ASHAScheduler
import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import os

# ENV_NAME = "RLE-v0"
# CHECKPOINT_PATH = "PPO_checkpoint"


ray.init(ignore_reinit_error=True, log_to_driver=False)


config = PPOConfig()
config = config.debugging(seed=1234)
config = config.training(gamma=1.0)
config = config.framework("torch")
config = config.rollouts(num_rollout_workers=1)
config = config.training(num_sgd_iter=20)
config = config.resources(num_cpus_per_worker=1)
config = config.training(sgd_minibatch_size=200)
config = config.training(train_batch_size=10000)
# config = config.training(model={"custom_model": DenseModel})
# config = config.rl_module(_enable_rl_module_api=False)
# config = config.training(_enable_learner_api=False)


config = config.environment(
    env=RLEEnv_expectation_negative_reward,
    env_config={
        "D": 6,
        "N_cohort": 3,
        "N_total": 36,
        "scenario": "random",
    },
)  # env=ENV_NAME,

config = config.reporting()

scheduler = ASHAScheduler(
    metric="episode_reward_mean",
    mode="max",
    max_t=50000,
    grace_period=50000,
    reduction_factor=2,
)

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=air.RunConfig(
        callbacks=[
            WandbLoggerCallback(
                project=os.path.splitext(os.path.basename(__file__))[0],
                api_key_file=os.path.join(
                    os.path.dirname(__file__), "wandb_api_key.txt"
                ),
            )
        ],
        # stop = {"training_iteration": 100}
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,  # searchalgはデフォルトがgrid search
        # max_concurrent_trials=4
    ),
)

results = tuner.fit()
# ray.shutdown()
