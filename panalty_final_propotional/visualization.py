# %%
# simulate_for_evaluation.pyにより，[ppo_panalty_final/result_10_scenario.pickle, ppo_panalty_final/result_20_scenario.pickle]を取得
import pickle 
import numpy as np
import matplotlib.pyplot as plt


with open('/home/chinen/esc_mcts/panalty_final/all_agents_all_scenarios_ppo.pickle', 'rb') as f:
    all_agents_all_scenarios_ppo = pickle.load(f)
all_agents_20_scenarios_ppo = all_agents_all_scenarios_ppo["all_agents_20_scenarios"]
all_agents_10_scenarios_ppo = all_agents_all_scenarios_ppo["all_agents_10_scenarios"]

# with open('/home/chinen/esc_mcts/panalty_final/all_agents_all_scenarios_alphazero_few_0.pickle', 'rb') as f:
#     all_agents_all_scenarios_alphazero_few_0 = pickle.load(f)
# all_agents_20_scenarios_alphazero = all_agents_all_scenarios_alphazero_few_0["all_agents_20_scenarios"]
# all_agents_10_scenarios_alphazero = all_agents_all_scenarios_alphazero_few_0["all_agents_10_scenarios"]
with open('/home/chinen/esc_mcts/panalty_final/all_agents_all_scenarios_alphazero.pickle', 'rb') as f:
    all_agents_all_scenarios_alphazero = pickle.load(f)
all_agents_20_scenarios_alphazero = all_agents_all_scenarios_alphazero["all_agents_20_scenarios"]
all_agents_10_scenarios_alphazero = all_agents_all_scenarios_alphazero["all_agents_10_scenarios"]
# %%

# シナリオを結合
def merge_scenarios(results):
    total = {}
    for scenario in results.keys():
        for key, value in results[scenario].items():
            if key not in total: total[key] = []
            total[key].extend(value)
    return total

    
# 100episodeごとの平均を算出
def plot_averaged(x_legend, ax, data):
    SPLIT_LEN = 100
    splitted = {}
    for key, value in data.items():
        splitted[key] = [np.mean(value[i*SPLIT_LEN:(i+1)*SPLIT_LEN]) for i in range(len(value)//SPLIT_LEN)]
    ax.scatter(splitted[x_legend], splitted["rewards"])

# def shrink_data(data, num):
#     result = {}
#     for key_agent, value_scenarios in data.items():
#         scenarios = {}
#         for key_scenario, value_metrics in value_scenarios.items():
#             metrics = {}
#             for key_metric, values in value_metrics.items():
#                 metrics[key_metric] = values[0:num]
#             scenarios[key_scenario] = metrics
#         result[key_agent] = scenarios
#     return result
            
# all_agents_20_scenarios_ppo = shrink_data(all_agents_20_scenarios_ppo, 1000)
# all_agents_10_scenarios_ppo = shrink_data(all_agents_10_scenarios_ppo, 1000)

#%%
# dataのデータサイズをnumに圧縮（最初のnum episodesだけ取る）
def compare_ppo_and_alphazero(agents_scenarios_ppo, agents_scenarios_alphazero, metric="rewards"):
    print("ppo:")
    for key, value in agents_scenarios_ppo.items():
        print(f"ppo: {key=}, mean {metric}={np.mean(merge_scenarios(value)[metric])}")
    # print("alphazero:")
    # for key, value in agents_scenarios_alphazero.items():
        print(f"alphazero: {key=}, mean {metric}={np.mean(merge_scenarios(agents_scenarios_alphazero[key])[metric])}")
        print(f"gain={np.mean(merge_scenarios(agents_scenarios_alphazero[key])[metric])-np.mean(merge_scenarios(value)[metric])}")

compare_ppo_and_alphazero(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero,"DLTs")

# %%
# ppo 20scenarioで縦軸DLTで描画
# fig, ax = plt.subplots()
# # ax.set_xlim([-1, 1])
# ax.set_xlim([0, 13])
# ax.set_ylim([-1, 1])
# for penalty_abs, scenarios in all_agents_20_scenarios_ppo.items():
#     plot_averaged("DLTs", ax, merge_scenarios(scenarios))
#%% 

# ppo 20scenarioで縦軸DLTで描画
fig, ax = plt.subplots()
# ax.set_xlim([-1, 1])
ax.set_xlim([0, 13])
ax.set_ylim([-1, 1])
for penalty_abs, scenarios in all_agents_20_scenarios_ppo.items():
    plot_averaged("DLTs", ax, merge_scenarios(scenarios)) 
    
    
# %%
## 100ごとに分けずにplot

from matplotlib.collections import LineCollection
import matplotlib.cm as cm
N = 10
colors = [cm.viridis(i/N) for i in range(N)]

fig, ax = plt.subplots()
penaltyies = ([np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["penalty_metric"]) for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])
rewards = ([np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["rewards"]) for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])

penaltyies_ = ([np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["penalty_metric"]) for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])
rewards_ = ([np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["rewards"]) for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])

lines = [[[penalty_ppo, reward_ppo], [penalty_alphazero, reward_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lc =  LineCollection(lines, colors=colors, linewidth=2)
ax.add_collection(lc)
ax.scatter(penaltyies, rewards, c='red', label="prior work")
ax.scatter(penaltyies_, rewards_, c='blue', label="proposed method")
plt.legend(loc=4)

#%%
# alphazero 20scenarioで縦軸DLTで描画
fig, ax = plt.subplots()
# ax.set_xlim([-1, 1])
ax.set_xlim([0, 13])
ax.set_ylim([-1, 1])
for penalty_abs, scenarios in all_agents_20_scenarios_alphazero.items():
    plot_averaged("DLTs", ax, merge_scenarios(scenarios))  
# %%
# ppo 20scenarioで縦軸penaltyで描画
fig, ax = plt.subplots()
# ax.set_xlim([-1, 1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
for penalty_abs, scenarios in all_agents_20_scenarios_ppo.items():
    plot_averaged("penalty_metric", ax, merge_scenarios(scenarios))
#%%
# alphazero 20scenarioで縦軸penaltyで描画
fig, ax = plt.subplots()
# ax.set_xlim([-1, 1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
for penalty_abs, scenarios in all_agents_20_scenarios_alphazero.items():
    plot_averaged("penalty_metric", ax, merge_scenarios(scenarios))  
    
# %% 
# DLT人数をヒストグラムで表示
# fig, ax = plt.subplots()
# plt.hist(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["DLTs"], [i-1/2 for i in range(min(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["DLTs"]), max(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["DLTs"])+2)])
# plt.hist(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["DLTs"], [i-1/2 for i in range(min(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["DLTs"]), max(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["DLTs"])+2)])
#%%
def compare_DLT_hist(first_data, second_data, penalty_key, second_penalty_key=None, legend=None):
    fig, ax = plt.subplots()
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([0,50000])
    if not second_penalty_key:
        second_penalty_key = penalty_key
    print("first data mean:")
    print(np.mean(merge_scenarios(first_data[penalty_key])["DLTs"]))
    print("second data mean:")
    print(np.mean(merge_scenarios(second_data[second_penalty_key])["DLTs"]))
    x = [merge_scenarios(first_data[penalty_key])["DLTs"],merge_scenarios(second_data[second_penalty_key])["DLTs"]]
    plt.hist(x, [i-1/2 for i in [i-1/2 for i in range(min(merge_scenarios(second_data[penalty_key])["DLTs"]), max(merge_scenarios(second_data[penalty_key])["DLTs"])+2)]], label=legend)
    plt.legend()
#%% 
# abs=0.0651 でppoとalphazeroをヒストグラムで比較
compare_DLT_hist(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero, "penalty_abs=0.0651", legend=["prior work", "proposed method"])
#%%
# abs=0.0651 でppoとalphazeroをヒストグラムで比較（evaluate用10 scenario）
compare_DLT_hist(all_agents_10_scenarios_ppo, all_agents_10_scenarios_alphazero, "penalty_abs=0.0651")
#%%
# alphazeroで，penalty_abs=0とpenalty_abs=0.0651をヒストグラムで比較
compare_DLT_hist(all_agents_20_scenarios_alphazero, all_agents_20_scenarios_alphazero, "penalty_abs=0", "penalty_abs=0.0651")
#%%
compare_DLT_hist(all_agents_20_scenarios_ppo, all_agents_20_scenarios_ppo, "penalty_abs=0", "penalty_abs=0.0651")
#%%
# abs=0 でppoとalphazeroをヒストグラムで比較
compare_DLT_hist(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero, "penalty_abs=0")
#%%
# # ppoで，penalty_abs=0とpenalty_abs=0.0651をヒストグラムで比較
# x = [merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["DLTs"],merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["DLTs"]]
# #%%
# # ppo, penalty
# fig, ax = plt.subplots()
# plt.hist(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"])["DLTs"], [i-1/2 for i in range(min(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"])["DLTs"]), max(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"])["DLTs"])+2)])

#%% 
# エージェント固定，シナリオ結合済みのデータに関して，penalty換算したtrue rewardを算出
def true_reward(scenario_merged_data, penalty_abs):
    true_reward = []
    for i in range(len(scenario_merged_data["DLTs"])):
        penalty = -penalty_abs*(scenario_merged_data["penalty_metric"][i]+1)/2
        reward = scenario_merged_data["rewards"][i]
        # print(f"{penalty+reward=}")
        true_reward.append(penalty+reward)
    return true_reward
#%%

true_reward_ppo = true_reward(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"]), 0.0651)
true_reward_alphazero = true_reward(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"]), 0.0651)
print(np.mean(true_reward_ppo))
print(np.mean(true_reward_alphazero))
#%%

def compare_reward_each_scenarios(agent_specified_data_first, agent_specified_data_second, key='rewards', legend=None):
    for key_scenario, value_metrics in agent_specified_data_first.items():
        # print(key_scenario+"---------------------")
        # print(f"first, {np.mean(value_metrics[key])}")
        # print(f"second, {np.mean(agent_specified_data_second[key_scenario][key])}")
        # print(f"second - first = {np.mean(agent_specified_data_second[key_scenario][key])-np.mean(value_metrics[key])}")
        # key = 'rewards'

        if np.mean(agent_specified_data_second[key_scenario][key])-np.mean(value_metrics[key])>0:
            print(f"{int(key_scenario[9:])+1} & {np.mean(value_metrics[key]):.3f} & $\\bm{{{np.mean(agent_specified_data_second[key_scenario][key]):.3f}}}$ & {np.mean(agent_specified_data_second[key_scenario][key])-np.mean(value_metrics[key]):.3f} \\\\")
        elif np.mean(agent_specified_data_second[key_scenario][key])-np.mean(value_metrics[key])<0:
            print(f"{int(key_scenario[9:])+1} & $\\bm{{{np.mean(value_metrics[key]):.3f}}}$ & {np.mean(agent_specified_data_second[key_scenario][key]):.3f} & {np.mean(agent_specified_data_second[key_scenario][key])-np.mean(value_metrics[key]):.3f} \\\\")
        else:
            print(f"{int(key_scenario[9:])+1} & {np.mean(value_metrics[key]):.3f} & {np.mean(agent_specified_data_second[key_scenario][key]):.3f} & {np.mean(agent_specified_data_second[key_scenario][key])-np.mean(value_metrics[key]):.3f} \\\\")
        if (int(key_scenario[9:])+1)%2==1:
            print(r"\rowcolor[rgb]{0.9, 0.9, 0.9}")
    if np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key])>0:
        print(f"average & {np.mean(merge_scenarios(agent_specified_data_first)[key]):.3f} & $\\bm{{{np.mean(merge_scenarios(agent_specified_data_second)[key]):.3f}}}$ & {np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key]):.3f} \\\\")
    elif np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key])<0:
        print(f"average & $\\bm{{{np.mean(merge_scenarios(agent_specified_data_first)[key]):.3f}}}$ & {np.mean(merge_scenarios(agent_specified_data_second)[key]):.3f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key]):.3f} \\\\")
    else:
        print(f"average & {np.mean(merge_scenarios(agent_specified_data_first)[key]):.3f} & {np.mean(merge_scenarios(agent_specified_data_second)[key]):.3f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key]):.3f} \\\\")
    
    x = [[np.mean(value_scenario["rewards"]) for value_scenario in agent_specified_data_first.values()], [np.mean(value_scenario["rewards"]) for value_scenario in agent_specified_data_second.values()]]
    print(x)
    # fig, ax = plt.subplots()
    if legend:
        plt.bar([i for i in range(1, 20+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_first.values()], align="center", width=0.3, label=legend[0])
        plt.bar([i+0.3 for i in range(1, 20+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_second.values()], align="center", width=0.3, label=legend[1])
    else:
        plt.bar([i for i in range(1, 20+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_first.values()], align="center", width=0.3)
        plt.bar([i+0.3 for i in range(1, 20+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_second.values()], align="center", width=0.3)
    label_x = ["#"+str(i) for i in range(1, 20+1)]
    # plt.ylim(-1, 1)
    plt.xticks([i for i in range(1, 20+1)], label_x)
    if legend:
        plt.legend(loc=4)

#%% 
# ppoとalphazeroのpenaltyなしでreward比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0"], legend=["prior work", "proposed method"])
#%% 
# penaltyありでreward比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], all_agents_20_scenarios_alphazero["penalty_abs=0.0651"], legend=["prior work", "proposed method"])
#%%
# ppoとalphazeroのpenaltyなしでpenalty metric比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0"], "penalty_metric")
#%%
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], all_agents_20_scenarios_alphazero["penalty_abs=0.0651"],"penalty_metric")


######### 10 scenarioでも実行 #########
#%%
# ppoとalphazeroのpenaltyなしでreward比較
compare_reward_each_scenarios(all_agents_10_scenarios_ppo["penalty_abs=0"], all_agents_10_scenarios_alphazero["penalty_abs=0"])
#%% 
# penaltyありでreward比較
compare_reward_each_scenarios(all_agents_10_scenarios_ppo["penalty_abs=0.0651"], all_agents_10_scenarios_alphazero["penalty_abs=0.0651"])

#%%
# penalty加えたrewardについて比較
def add_true_reward_each_scenario(agent_fixed_data, penalty_abs):
    result = {}
    for key_scenario, value_metrics in agent_fixed_data.items():
        result[key_scenario] = value_metrics|{"true_rewards": true_reward(value_metrics, penalty_abs)}
    return result
# add_true_reward_each_scenario(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], 0.0651)
compare_reward_each_scenarios(add_true_reward_each_scenario(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], 0.0651), add_true_reward_each_scenario(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"], 0.0651),"true_rewards", legend=["prior work", "proposed method"])

######## 10 scenarioでも実行 #######
# %%
compare_reward_each_scenarios(add_true_reward_each_scenario(all_agents_10_scenarios_ppo["penalty_abs=0.0651"], 0.0651), add_true_reward_each_scenario(all_agents_10_scenarios_alphazero["penalty_abs=0.0651"], 0.0651),"true_rewards")

#%% 
def compare_rewards_with_sd(first_agent_fixed_data, second_agent_fixed_data):
    labels = first_agent_fixed_data.keys()
    firt_agent_rewards_means = [np.mean(metrics["rewards"]) for metrics in first_agent_fixed_data.values()]
    second_agent_rewards_means = [np.mean(metrics["rewards"]) for metrics in second_agent_fixed_data.values()]
    firt_agent_rewards_stds = [np.std(metrics["rewards"]) for metrics in first_agent_fixed_data.values()]
    second_agent_rewards_stds = [np.std(metrics["rewards"]) for metrics in second_agent_fixed_data.values()]
    fig, ax = plt.subplots()
    ax.errorbar([labels, labels], [firt_agent_rewards_means, second_agent_rewards_means], [firt_agent_rewards_stds, second_agent_rewards_stds])
compare_rewards_with_sd(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0"])

# def 
# %%
# ppo 10scenarioで縦軸DLTで描画
# fig, ax = plt.subplots()
# # ax.set_xlim([-1, 1])
# ax.set_xlim([0, 13])
# ax.set_ylim([-1, 1])
# for penalty_abs, scenarios in all_agents_10_scenarios_ppo.items():
#     plot_averaged("DLTs", ax, merge_scenarios(scenarios))
# %%
