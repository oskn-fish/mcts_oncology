
# %%
import numpy as np
import pandas as pd
from scipy.special import softmax

import ray
from ray.rllib.policy import Policy
# from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate

import tqdm

checkpoint_path = "/home/chinen/esc_mcts/PPO_checkpoint_negative_reward/policies/default_policy"
result = {}


def evaluate(scenarioID):
    agent = Policy.from_checkpoint(checkpoint_path)
    sim_config = {'D':6, 'N_cohort':3, 'N_total':36, 'phi':0.25, 'DLT_max':100, 'early_stop': False, 'penalty_abs': 0,  'scenario':scenarioID}
    env = RLEEnv_expectation_negative_reward_penalty_continue_evaluate(sim_config)

    simID = 123
    env.seed(simID)

    
    DLTs = []
    rewards = []
    penalties = []
    print(f"{scenarioID}=")
    for _ in tqdm.tqdm(range(10000)):
        state, info = env.reset()
        done = False
        while not done:
            action = agent.compute_single_action(state, full_fetch = True)
            probs = softmax(action[2]['action_dist_inputs'])
            action = np.argmax(probs)
            state, reward, done, truncated, info = env.step(action)
          
        assert info["DLTs"] != np.nan
        DLTs.append(info["DLTs"])
        rewards.append(reward)
        penalties.append(info["penalty_metric"])
        assert type(info["DLTs"]) == int
    # assert not np.isnan(np.array(DLTs)).any() and  not np.isnan(np.array(rewards)).any() and not np.isnan(np.array(penalties)).any()
    result = {"DLTs": DLTs, "rewards": rewards, "penalties":penalties}
    return result


for scenario_ID in range(10):
    print(evaluate(scenario_ID))