
# %%
import numpy as np
import pandas as pd
from scipy.special import softmax

import ray
from ray.rllib.policy import Policy
# from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate

import tqdm


ray.init(ignore_reinit_error=True, log_to_driver=False)


# SAFE_MODE = True

# checkpoint_path = 'checkpoint/checkpoint_003000'
checkpoint_path = "/home/chinen/esc_mcts/PPO_checkpoint_negative_reward/policies/default_policy"
# measure_names = ['MTD', 'reward']
# state_names = ['state' + str(num).zfill(2) for num in range(1+6+6+2)]
# prob_names = ['prob' + str(i) for i in range(3+6)]

from ray.rllib.algorithms import Algorithm
agent = Algorithm.from_checkpoint("/home/chinen/esc_mcts/PPO_checkpoint_negative_reward/")

# results_score = []
# results_cohort = []
result = {}

@ray.remote
def evaluate(scenarioID):
    # for scenarioID in range(10):
    # sim_config = {'D':6, 'N_cohort':3, 'N_total':36, 'phi':0.25, 'scenario':str(scenarioID)}
    agent = Policy.from_checkpoint(checkpoint_path)
    sim_config = {'D':6, 'N_cohort':3, 'N_total':36, 'phi':0.25, 'DLT_max':100, 'early_stop': False, 'penalty_abs': 0,  'scenario':scenarioID}
    env = RLEEnv_expectation_negative_reward_penalty_continue_evaluate(sim_config)

    simID = 123
    env.seed(simID)
    # state = env.reset()
    # done = False
    # cohortID = 1
    
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

            # if SAFE_MODE:
            #     current_dose = (np.rint(state[0] * (sim_config['D']-1))).astype(np.int)
            #     Ns = np.round(state[1:(sim_config['D']+1)] * sim_config['N_total'])
            #     DLTs = np.round(state[(sim_config['D']+1):(2*sim_config['D']+1)] * sim_config['N_total'])
            #     ratios = DLTs / Ns
            #     if ratios[current_dose] > sim_config['phi'] and current_dose <= sim_config['D'] - 2 and action == 2:
            #         action = 1
            #     elif ratios[current_dose] > 2*sim_config['phi'] and current_dose >= 1 and 1 <= action <= 2:
            #         action = 0
            #     elif ratios[current_dose] < sim_config['phi'] and action == 0:
            #         action = 1

            # results_cohort.append([scenarioID, simID, cohortID, *probs, action, *state])
            state, reward, done, truncated, info = env.step(action)
            # cohortID += 1
            # if done:
                # measures = [info['MTD'], reward]
                # results_score.append([scenarioID, simID, *measures, *state])
        assert info["DLTs"] != np.nan
        DLTs.append(info["DLTs"])
        rewards.append(reward)
        penalties.append(info["penalty_metric"])
    
    # result["scenario_"+str(scenarioID)] = {"DLTs": DLTs, "rewards": rewards, "penalties":penalties}
    result = {"DLTs": DLTs, "rewards": rewards, "penalties":penalties}
    return result
    
promises = [evaluate.remote(scenario_ID) for scenario_ID in range(10)]
# results = {}
# for scenario_ID in len(promises):
#     results["senario_ID"+scenario_ID] = ray.get(promises[scenario_ID])

# %%
import pickle
with open("evalulation.pickle", 'wb') as f:
    # pickle.dump(result, f)
    results = {"scenario_"+str(scenario_ID): ray.get(promises[scenario_ID]) for scenario_ID in range(len(promises))}
    pickle.dump(results, f)

    # df_score = pd.DataFrame(results_score, columns=['scenarioID', 'simID', *measure_names, *state_names])
    # df_cohort = pd.DataFrame(results_cohort, columns=['scenarioID', 'simID', 'cohortID', *prob_names, 'action', *state_names])
    # df_score.to_csv('evaluation_score.csv', index=False)
    # df_cohort.to_csv('evaluation_cohort.csv', index=False)

ray.shutdown()
#%%
# import statistics
# for scenario_ID in range(10):
#     print(statistics.mean(result["scenario_"+str(scenario_ID)]["DLTs"]))
# # %%
# result
# # %%
import pickle
with open('evalulation.pickle', 'rb') as f:
    results = pickle.load(f)
    
results

# %%
import matplotlib.pyplot as plt
all = []
for ID in range(10):
    fig, ax = plt.subplots()
    x = results["scenario_"+str(ID)]["DLTs"]
    ax.hist(x, rwidth=1, bins=[1/2 + i for i in range(min(x)-1, max(x)+1)])
    plt.show()
    all = all+x
    print(np.mean(np.array(x)))
#%%
np.mean(all)

# %%
import numpy as np
np.nanmean(np.array(x))
# %%
np.set_printoptions(threshold=np.inf)
np.array(x)
# %%
np.argwhere(np.isnan(x))
# %%

# %%
