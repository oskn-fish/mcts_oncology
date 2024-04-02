# %%
# simulate_for_evaluation.pyにより，[ppo_panalty_final/result_10_scenario.pickle, ppo_panalty_final/result_20_scenario.pickle]を取得
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


with open('/home/chinen/esc_mcts/panalty_final_propotional/all_agents_all_scenarios_ppo.pickle', 'rb') as f:
    all_agents_all_scenarios_ppo = pickle.load(f)
all_agents_20_scenarios_ppo = all_agents_all_scenarios_ppo["all_agents_20_scenarios"]
all_agents_10_scenarios_ppo = all_agents_all_scenarios_ppo["all_agents_10_scenarios"]

# with open('/home/chinen/esc_mcts/panalty_final/all_agents_all_scenarios_alphazero_few_0.pickle', 'rb') as f:
#     all_agents_all_scenarios_alphazero_few_0 = pickle.load(f)
# all_agents_20_scenarios_alphazero = all_agents_all_scenarios_alphazero_few_0["all_agents_20_scenarios"]
# all_agents_10_scenarios_alphazero = all_agents_all_scenarios_alphazero_few_0["all_agents_10_scenarios"]
with open('/home/chinen/esc_mcts/panalty_final_propotional/all_agents_all_scenarios_alphazero.pickle', 'rb') as f:
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
# def plot_averaged(x_legend, ax, data):
#     SPLIT_LEN = 100
#     splitted = {}
#     for key, value in data.items():
#         splitted[key] = [np.mean(value[i*SPLIT_LEN:(i+1)*SPLIT_LEN]) for i in range(len(value)//SPLIT_LEN)]
#     ax.scatter(splitted[x_legend], splitted["rewards"])

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
# def compare_ppo_and_alphazero(agents_scenarios_ppo, agents_scenarios_alphazero, metric="rewards"):
#     print("ppo:")
#     for key, value in agents_scenarios_ppo.items():
#         print(f"ppo: {key=}, mean {metric}={np.mean(merge_scenarios(value)[metric])}")
#     # print("alphazero:")
#     # for key, value in agents_scenarios_alphazero.items():
#         print(f"alphazero: {key=}, mean {metric}={np.mean(merge_scenarios(agents_scenarios_alphazero[key])[metric])}")
#         print(f"gain={np.mean(merge_scenarios(agents_scenarios_alphazero[key])[metric])-np.mean(merge_scenarios(value)[metric])}")

# compare_ppo_and_alphazero(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero,"DLTs")

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
# fig, ax = plt.subplots()
# # ax.set_xlim([-1, 1])
# ax.set_xlim([0, 13])
# ax.set_ylim([-1, 1])
# for penalty_abs, scenarios in all_agents_20_scenarios_ppo.items():
#     plot_averaged("DLTs", ax, merge_scenarios(scenarios)) 
    
    
# %%
## 20シナリオで横軸penalty発生率，縦軸正解率でplot

from matplotlib.collections import LineCollection
import matplotlib.cm as cm
N = 10
colors = [cm.viridis(i/N) for i in range(N)]

fig, ax = plt.subplots()
penaltyies = ([(np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["penalty_metric"])/2+1/2)*100 for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])
rewards = ([(np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["rewards"])/2+1/2)*100 for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])


a, b = np.polyfit(penaltyies, rewards, 1)
y = a*np.linspace(15, 55,len(penaltyies))+b

penaltyies_ = ([(np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["penalty_metric"])/2+1/2)*100 for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])
rewards_ = ([(np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["rewards"])/2+1/2)*100 for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])

a_, b_ = np.polyfit(penaltyies_, rewards_, 1)
y_ = a_*np.linspace(15, 55,len(penaltyies))+b_

lines = [[[penalty_ppo, reward_ppo], [penalty_alphazero, reward_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lc =  LineCollection(lines, colors=colors, linewidth=2)

ax.plot(np.linspace(15, 55,len(penaltyies)), y, color='lightcoral')
ax.plot(np.linspace(15, 55,len(penaltyies)), y_, color='cornflowerblue')
ax.add_collection(lc)
ax.scatter(penaltyies, rewards, c='red', label="RLE")
ax.scatter(penaltyies_, rewards_, c='blue', label="提案手法")

fontsize=20
ax.set_xlabel("用量制限毒性上限超過率 (%)", fontsize=fontsize)
ax.set_ylabel("最大耐用量推定精度 (%)", fontsize=fontsize)

# ax.text(43, 45*a+b-0.5, "y="+str(a)[:6]+"x+"+str(b)[:6], color="red",fontsize=14)
# ax.text(20, 20*a_+b_+0.5, "y="+str(a_)[:6]+"x+"+str(b_)[:6], color="blue",fontsize=14)
ax.text(43, 45*a+b-0.5, "既存手法RLE", color="red",fontsize=18)
ax.text(15, 20*a_+b_+0.4, "提案手法", color="blue",fontsize=18)
# plt.legend(loc=4, fontsize=14)
plt.tick_params(labelsize=15)

#
    # ax.set_xlabel("DLT発現人数", fontsize=19)
    # ax.set_ylabel("頻度", fontsize=19)

    # # plt.legend(merge_scenarios(first_data[penalty_key])["DLTs"])
    # plt.xticks([i for i in range(1, 19)])
    # # plt.rcParams["font.size"] = 10
    # plt.legend(fontsize=17)
    # plt.tick_params(labelsize=15)
# %%
## 20シナリオで横軸penalty発生率，縦軸DLT人数でplot

from matplotlib.collections import LineCollection
import matplotlib.cm as cm
N = 10
colors = [cm.viridis(i/N) for i in range(N)]

fig, ax = plt.subplots()
penaltyies = ([(np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["penalty_metric"])/2+1/2)*100 for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])
rewards = ([np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["DLTs"]) for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])


a, b = np.polyfit(penaltyies, rewards, 1)
y = a*np.linspace(15, 55,len(penaltyies))+b

penaltyies_ = ([(np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["penalty_metric"])/2+1/2)*100 for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])
rewards_ = ([np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["DLTs"]) for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])

a_, b_ = np.polyfit(penaltyies_, rewards_, 1)
y_ = a_*np.linspace(15, 55,len(penaltyies))+b_

lines = [[[penalty_ppo, reward_ppo], [penalty_alphazero, reward_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lc =  LineCollection(lines, colors=colors, linewidth=2)

ax.plot(np.linspace(15, 55,len(penaltyies)), y, color='lightcoral')
ax.plot(np.linspace(15, 55,len(penaltyies)), y_, color='cornflowerblue')
ax.add_collection(lc)
ax.scatter(penaltyies, rewards, c='red', label="RLE")
ax.scatter(penaltyies_, rewards_, c='blue', label="提案手法")
fontsize=16
ax.set_xlabel("DLT上限超過率 (%)", fontsize=fontsize)
ax.set_ylabel("DLT発現人数", fontsize=fontsize)
plt.tick_params(labelsize=15)

# ax.text(43, 45*a+b-0.5, "y="+str(a)[:6]+"x+"+str(b)[:6], color="red")
# ax.text(20, 20*a_+b_+0.7, "y="+str(a_)[:6]+"x+"+str(b_)[:6], color="blue")
plt.legend(loc=4, fontsize=14)
#%%
def cal_std(data):
    num_negative = np.unique(data, return_counts=True)[1][0]
    num_positive = np.unique(data, return_counts=True)[1][1]
    alpha = num_negative+1
    beta = num_positive+1
    return np.sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1)))*100
set_test = set()
for key_agent in sorted(all_agents_20_scenarios_ppo.keys()):
    assert not cal_std(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])['rewards']) in set_test 
    set_test.add(cal_std(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])['rewards']))
    assert not cal_std(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])['rewards']) in set_test 
    set_test.add(cal_std(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])['rewards']))
    assert not cal_std(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])['penalty_metric']) in set_test 
    set_test.add(cal_std(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])['penalty_metric']))
    assert not cal_std(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])['penalty_metric']) in set_test 
    set_test.add(cal_std(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])['penalty_metric']))

print(set_test)
#%%

#%%
# rewardとpenalty_metric with std計算
def table_maker(first_data, second_data):
    i = 1
    for key_agent in sorted(first_data.keys()):
        if i%2==0:
            print("\\rowcolor[rgb]{0.9, 0.9, 0.9}")
        if (np.mean(merge_scenarios(first_data[key_agent])['rewards'])/2+1/2)*100>(np.mean(merge_scenarios(second_data[key_agent])['rewards'])/2+1/2)*100:
            print(f"{float(key_agent[12:]):.4f} & $\\bm{{{(np.mean(merge_scenarios(first_data[key_agent])['rewards'])/2+1/2)*100:.2f}\pm{cal_std(merge_scenarios(first_data[key_agent])['rewards']):.2f}}}$ & {(np.mean(merge_scenarios(second_data[key_agent])['rewards'])/2+1/2)*100:.2f} $\\pm${cal_std(merge_scenarios(second_data[key_agent])['rewards']):.2f}", end="")
        else:
            print(f"{float(key_agent[12:]):.4f} & {(np.mean(merge_scenarios(first_data[key_agent])['rewards'])/2+1/2)*100:.2f}$\\pm${cal_std(merge_scenarios(first_data[key_agent])['rewards']):.2f} & $\\bm{{{(np.mean(merge_scenarios(second_data[key_agent])['rewards'])/2+1/2)*100:.2f}\\pm{cal_std(merge_scenarios(second_data[key_agent])['rewards']):.2f}}}$", end="")
            
        print(f" & {(np.mean(merge_scenarios(second_data[key_agent])['rewards'])/2+1/2)*100-(np.mean(merge_scenarios(first_data[key_agent])['rewards'])/2+1/2)*100:.2f}", end="")

        if (np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100<(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:
            print(f" & $\\bm{{{(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}\pm{cal_std(merge_scenarios(first_data[key_agent])['penalty_metric']):.2f}}}$ & {(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}$\\pm${cal_std(merge_scenarios(second_data[key_agent])['penalty_metric']):.2f}", end="")
        else:
            print(f" & {(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}$\\pm${cal_std(merge_scenarios(first_data[key_agent])['penalty_metric']):.2f}& $\\bm{{{(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}\\pm{cal_std(merge_scenarios(second_data[key_agent])['penalty_metric']):.2f}}}$ ", end="")
        print(f" & {(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100-(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f} \\\\")
        i += 1
table_maker(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero)
#%%
def honda_std(list_data):
    # print(np.mean(list_data))
    # print(len(list_data))
    # print(np.mean(list_data)*(1-np.mean(list_data))/len(list_data))
    return np.sqrt(np.mean(list_data)*(1-np.mean(list_data))/len(list_data))
# %%
def DLT_PCR_table(first_data, second_data):
    i = 1
    for key_agent in sorted(first_data.keys()):
        if i%2==0:
            print("\\rowcolor[rgb]{0.9, 0.9, 0.9}")
        if np.mean(merge_scenarios(first_data[key_agent])['DLTs'])<np.mean(merge_scenarios(second_data[key_agent])['DLTs']):
            print(f"{float(key_agent[12:]):.4f} & $\\bm{{{np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}\pm{np.std(merge_scenarios(first_data[key_agent])['DLTs']):.2f}}} $ & {np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f} $\\pm${np.std(merge_scenarios(second_data[key_agent])['DLTs']):.2f}", end="")
        else:
            print(f"{float(key_agent[12:]):.4f} & {np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}$\\pm${np.std(merge_scenarios(first_data[key_agent])['DLTs']):.2f} & $\\bm{{{np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f}\\pm{np.std(merge_scenarios(second_data[key_agent])['DLTs']):.2f}}}$", end="")
            
        print(f" & {np.mean(merge_scenarios(second_data[key_agent])['DLTs'])-np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}", end="")
        print(merge_scenarios(first_data[key_agent])['penalty_metric'])
        if (np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100<(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:
            print(f" & $\\bm{{{(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}\pm{honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}}}$ & {(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}$\\pm${honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}", end="")
        else:
            print(f" & {(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}$\\pm${honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}& $\\bm{{{(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}\\pm{honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}}}$", end="")
        print(f" & {(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100-(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f} \\\\")
        i += 1
DLT_PCR_table(all_agents_20_scenarios_ppo,all_agents_20_scenarios_alphazero)
#%%
def DLT_PCR_table(first_data, second_data):
    i = 1
    for key_agent in sorted(first_data.keys()):
        if i%2==0:
            print("\\rowcolor[rgb]{0.9, 0.9, 0.9}")
        if np.mean(merge_scenarios(first_data[key_agent])['DLTs'])<np.mean(merge_scenarios(second_data[key_agent])['DLTs']):
            print(f"{float(key_agent[12:]):.4f} & $\\bm{{{np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}\pm{np.std(merge_scenarios(first_data[key_agent])['DLTs']):.2f}}} $ & {np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f} $\\pm${np.std(merge_scenarios(second_data[key_agent])['DLTs']):.2f}", end="")
        else:
            print(f"{float(key_agent[12:]):.4f} & {np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}$\\pm${np.std(merge_scenarios(first_data[key_agent])['DLTs']):.2f} & $\\bm{{{np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f}\\pm{np.std(merge_scenarios(second_data[key_agent])['DLTs']):.2f}}}$", end="")
            
        print(f" & {np.mean(merge_scenarios(second_data[key_agent])['DLTs'])-np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}", end="")
        print(merge_scenarios(first_data[key_agent])['penalty_metric'])
        if (np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100<(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:
            print(f" & $\\bm{{{(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}\pm{honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}}}$ & {(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}$\\pm${honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}", end="")
        else:
            print(f" & {(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}$\\pm${honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}& $\\bm{{{(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}\\pm{honda_std(np.array(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f}}}$", end="")
        print(f" & {(np.mean(merge_scenarios(second_data[key_agent])['penalty_metric'])/2+1/2)*100-(np.mean(merge_scenarios(first_data[key_agent])['penalty_metric'])/2+1/2)*100:.2f} \\\\")
        i += 1
DLT_PCR_table(all_agents_20_scenarios_ppo,all_agents_20_scenarios_alphazero)
#%%
# ペナルティ加算
def penalty_added_reward_comparison(first_data, second_data):
    i = 1
    for key_agent in sorted(first_data.keys()):
        if i%2==0:
            print("\\rowcolor[rgb]{0.9, 0.9, 0.9}")
        penalty_added_rewards_first = np.array(merge_scenarios(first_data[key_agent])["rewards"])+np.array(merge_scenarios(first_data[key_agent])["penalty_metric"])*float(key_agent[12:])
        penalty_added_rewards_second = np.array(merge_scenarios(second_data[key_agent])["rewards"])+np.array(merge_scenarios(second_data[key_agent])["penalty_metric"])*float(key_agent[12:])
        # print(penalty_added_rewards_first)
        if np.mean(penalty_added_rewards_first)>np.mean(penalty_added_rewards_second):
            print(f"{float(key_agent[12:]):.4f} & $\\bm{{{np.mean(penalty_added_rewards_first):.4f}}}$ & {np.mean(penalty_added_rewards_second):.4f} & {np.mean(penalty_added_rewards_second)-np.mean(penalty_added_rewards_first):.4f} ", end="")
        else:
            print(f"{float(key_agent[12:]):.4f} & {np.mean(penalty_added_rewards_first):.4f}& $\\bm{{{np.mean(penalty_added_rewards_second):.4f}}}$ & {np.mean(penalty_added_rewards_second)-np.mean(penalty_added_rewards_first):.4f} ",end="")
        # print("")
            
        if np.mean(merge_scenarios(first_data[key_agent])['DLTs'])<np.mean(merge_scenarios(second_data[key_agent])['DLTs']):
            # print(f"& $\\bm{{{np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}\pm{np.std(merge_scenarios(first_data[key_agent])['DLTs']):.2f}}} $ & {np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f} $\\pm${np.std(merge_scenarios(second_data[key_agent])['DLTs']):.2f} ")
            print(f"& $\\bm{{{np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f}}} $ & {np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f}", end="")
        else:
            print(f"& {np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f} & $\\bm{{{np.mean(merge_scenarios(second_data[key_agent])['DLTs']):.2f}}}$ ", end="")
        print(f" & {np.mean(merge_scenarios(second_data[key_agent])['DLTs'])-np.mean(merge_scenarios(first_data[key_agent])['DLTs']):.2f} \\\\")
        i += 1
penalty_added_reward_comparison(all_agents_20_scenarios_ppo,all_agents_20_scenarios_alphazero)

#%%
# 横軸DLT人数にする
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
N = 10
colors = [cm.viridis(i/N) for i in range(N)]

fig, ax = plt.subplots()
penaltyies = ([np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["DLTs"]) for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])
rewards = ([np.mean(merge_scenarios(all_agents_20_scenarios_ppo[key_agent])["rewards"])/2+1/2 for key_agent in sorted(all_agents_20_scenarios_ppo.keys())])

penaltyies_ = ([np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["DLTs"]) for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])
rewards_ = ([np.mean(merge_scenarios(all_agents_20_scenarios_alphazero[key_agent])["rewards"])/2+1/2 for key_agent in sorted(all_agents_20_scenarios_alphazero.keys())])

# lines = [[[penalty_ppo, reward_ppo], [penalty_alphazero, reward_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lines = [[[reward_ppo, penalty_ppo], [reward_alphazero, penalty_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lc =  LineCollection(lines, colors=colors, linewidth=2)
ax.add_collection(lc)
# ax.scatter(penaltyies, rewards, c='red', label="prior work")
# ax.scatter(penaltyies_, rewards_, c='blue', label="proposed method")
ax.scatter(rewards, penaltyies, c='red', label="prior work")
ax.scatter(rewards_, penaltyies_, c='blue', label="proposed method")
plt.legend(loc=4)

#%%
# 横軸penaltyで10シナリオ
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
N = 10
colors = [cm.viridis(i/N) for i in range(N)]

fig, ax = plt.subplots()
penaltyies = ([np.mean(merge_scenarios(all_agents_10_scenarios_ppo[key_agent])["penalty_metric"])/2+1/2 for key_agent in sorted(all_agents_10_scenarios_ppo.keys())])
rewards = ([np.mean(merge_scenarios(all_agents_10_scenarios_ppo[key_agent])["rewards"])/2+1/2 for key_agent in sorted(all_agents_10_scenarios_ppo.keys())])

penaltyies_ = ([np.mean(merge_scenarios(all_agents_10_scenarios_alphazero[key_agent])["penalty_metric"])/2+1/2 for key_agent in sorted(all_agents_10_scenarios_alphazero.keys())])
rewards_ = ([np.mean(merge_scenarios(all_agents_10_scenarios_alphazero[key_agent])["rewards"])/2+1/2 for key_agent in sorted(all_agents_10_scenarios_alphazero.keys())])

# lines = [[[penalty_ppo, reward_ppo], [penalty_alphazero, reward_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lines = [[[reward_ppo, penalty_ppo], [reward_alphazero, penalty_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lc =  LineCollection(lines, colors=colors, linewidth=2)
ax.add_collection(lc)
# ax.scatter(penaltyies, rewards, c='red', label="prior work")
# ax.scatter(penaltyies_, rewards_, c='blue', label="proposed method")
ax.scatter(rewards, penaltyies, c='red', label="prior work")
ax.scatter(rewards_, penaltyies_, c='blue', label="proposed method")
plt.legend(loc=4)

#%%
# 横軸DLT人数にして10シナリオ
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
N = 10
colors = [cm.viridis(i/N) for i in range(N)]

fig, ax = plt.subplots()
penaltyies = ([np.mean(merge_scenarios(all_agents_10_scenarios_ppo[key_agent])["DLTs"]) for key_agent in sorted(all_agents_10_scenarios_ppo.keys())])
rewards = ([np.mean(merge_scenarios(all_agents_10_scenarios_ppo[key_agent])["rewards"])/2+1/2 for key_agent in sorted(all_agents_10_scenarios_ppo.keys())])

penaltyies_ = ([np.mean(merge_scenarios(all_agents_10_scenarios_alphazero[key_agent])["DLTs"]) for key_agent in sorted(all_agents_10_scenarios_alphazero.keys())])
rewards_ = ([np.mean(merge_scenarios(all_agents_10_scenarios_alphazero[key_agent])["rewards"])/2+1/2 for key_agent in sorted(all_agents_10_scenarios_alphazero.keys())])

lines = [[[penalty_ppo, reward_ppo], [penalty_alphazero, reward_alphazero]] for penalty_ppo, reward_ppo, penalty_alphazero, reward_alphazero in zip(penaltyies, rewards, penaltyies_, rewards_)]
lc =  LineCollection(lines, colors=colors, linewidth=2)
ax.add_collection(lc)
ax.scatter(penaltyies, rewards, c='red', label="prior work")
ax.scatter(penaltyies_, rewards_, c='blue', label="proposed method")
plt.legend(loc=4)

#%%
# alphazero 20scenarioで縦軸DLTで描画
# fig, ax = plt.subplots()
# # ax.set_xlim([-1, 1])
# ax.set_xlim([0, 13])
# ax.set_ylim([-1, 1])
# for penalty_abs, scenarios in all_agents_20_scenarios_alphazero.items():
#     plot_averaged("DLTs", ax, merge_scenarios(scenarios))  
# %%
# ppo 20scenarioで縦軸penaltyで描画
# fig, ax = plt.subplots()
# # ax.set_xlim([-1, 1])
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# for penalty_abs, scenarios in all_agents_20_scenarios_ppo.items():
#     plot_averaged("penalty_metric", ax, merge_scenarios(scenarios))
#%%
# alphazero 20scenarioで縦軸penaltyで描画
# fig, ax = plt.subplots()
# # ax.set_xlim([-1, 1])
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# for penalty_abs, scenarios in all_agents_20_scenarios_alphazero.items():
#     plot_averaged("penalty_metric", ax, merge_scenarios(scenarios))  
    
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

    # ax.hist(x, [i-1/2 for i in [i-1/2 for i in range(min(merge_scenarios(second_data[penalty_key])["DLTs"]), max(merge_scenarios(second_data[penalty_key])["DLTs"])+2)]], label=legend, color=["crimson","dodgerblue"])
    ax.hist(merge_scenarios(first_data[penalty_key])["DLTs"], [i-1/2 for i in range(1,19)], color="red", alpha=0.5, label="RLE")
    ax.hist(merge_scenarios(second_data[penalty_key])["DLTs"], [i-1/2 for i in range(1,19)], color="blue", alpha=0.5, label="提案手法")
    ax.vlines(9.5, 0, 50000, colors="purple", linestyle="dashed", linewidth=3, alpha=0.7)    
    ax.set_xlabel("DLT発現人数", fontsize=19)
    ax.set_ylabel("頻度", fontsize=19)

    # plt.legend(merge_scenarios(first_data[penalty_key])["DLTs"])
    plt.xticks([i for i in range(1, 19)])
    # plt.rcParams["font.size"] = 10
    plt.legend(fontsize=17)
    plt.tick_params(labelsize=15)
    # plt.legend()
    plt.show()
#%% 
# abs=0.0651 でppoとalphazeroをヒストグラムで比較
compare_DLT_hist(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero, "penalty_abs=0.0651", legend=["RLE", "提案手法"])
#%%
# abs=0.0651 でppoとalphazeroをヒストグラムで比較（evaluate用10 scenario）
compare_DLT_hist(all_agents_10_scenarios_ppo, all_agents_10_scenarios_alphazero, "penalty_abs=0.0651")
#%%
# alphazeroで，penalty_abs=0とpenalty_abs=0.0651をヒストグラムで比較
# compare_DLT_hist(all_agents_20_scenarios_alphazero, all_agents_20_scenarios_alphazero, "penalty_abs=0", "penalty_abs=0.0651")
#%%
# compare_DLT_hist(all_agents_20_scenarios_ppo, all_agents_20_scenarios_ppo, "penalty_abs=0", "penalty_abs=0.0651")
#%%
# abs=0 でppoとalphazeroをヒストグラムで比較
# compare_DLT_hist(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero, "penalty_abs=0")
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

def compare_reward_each_scenarios(agent_specified_data_first, agent_specified_data_second, key='rewards', legend=None, rescale=False):
    for key_scenario, value_metrics in agent_specified_data_first.items():
        # print(key_scenario+"---------------------")
        # print(f"first, {np.mean(value_metrics[key])/2+1/2}")
        # print(f"second, {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2}")
        # print(f"second - first = {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2-np.mean(value_metrics[key])}")
        # key = 'rewards'

        if np.mean(agent_specified_data_second[key_scenario][key])/2+1/2-np.mean(value_metrics[key])/2+1/2>0:
            print(f"{int(key_scenario[9:])+1} & {np.mean(value_metrics[key])/2+1/2:.4f} & $\\bm{{{np.mean(agent_specified_data_second[key_scenario][key])/2+1/2:.4f}}}$ & {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2-(np.mean(value_metrics[key])/2+1/2):.4f} \\\\")
        elif np.mean(agent_specified_data_second[key_scenario][key])/2+1/2-np.mean(value_metrics[key])/2+1/2<0:
            print(f"{int(key_scenario[9:])+1} & $\\bm{{{np.mean(value_metrics[key])/2+1/2:.4f}}}$ & {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2:.4f} & {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2-(np.mean(value_metrics[key])/2+1/2):.4f} \\\\")
        else:
            print(f"{int(key_scenario[9:])+1} & {np.mean(value_metrics[key])/2+1/2:.4f} & {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2:.4f} & {np.mean(agent_specified_data_second[key_scenario][key])/2+1/2-(np.mean(value_metrics[key])/2+1/2):.4f} \\\\")
        if (int(key_scenario[9:])+1)%2==1:
            print(r"\rowcolor[rgb]{0.9, 0.9, 0.9}")
    if np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key])>0:
        print(f"average & {np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2:.4f} & $\\bm{{{np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2:.4f}}}$ & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2-(np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2):.4f} \\\\")
    elif np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key])<0:
        print(f"average & $\\bm{{{np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2:.4f}}}$ & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2:.4f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2-(np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2):.4f} \\\\")
    else:
        print(f"average & {np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2:.4f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2:.4f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2-(np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2):.4f} \\\\")
    
    # x = [[np.mean(value_scenario["rewards"]) for value_scenario in agent_specified_data_first.values()], [np.mean(value_scenario["rewards"]) for value_scenario in agent_specified_data_second.values()]]

    if legend:
        if rescale:
            plt.bar([i for i in range(1, len(agent_specified_data_first)+1)], [np.mean(value_scenario[key])/2+1/2 for value_scenario in agent_specified_data_first.values()], align="center", width=0.3, label=legend[0])
            plt.bar([i+0.3 for i in range(1, len(agent_specified_data_second)+1)], [np.mean(value_scenario[key])/2+1/2 for value_scenario in agent_specified_data_second.values()], align="center", width=0.3, label=legend[1])
        else:
            plt.bar([i for i in range(1, len(agent_specified_data_first)+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_first.values()], align="center", width=0.3, label=legend[0])
            plt.bar([i+0.3 for i in range(1, len(agent_specified_data_second)+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_second.values()], align="center", width=0.3, label=legend[1])
    else:
        if rescale:
            plt.bar([i for i in range(1, len(agent_specified_data_first)+1)], [np.mean(value_scenario[key])/2+1/2 for value_scenario in agent_specified_data_first.values()], align="center", width=0.3)
            plt.bar([i+0.3 for i in range(1, len(agent_specified_data_second)+1)], [np.mean(value_scenario[key])/2+1/2 for value_scenario in agent_specified_data_second.values()], align="center", width=0.3)
        else:
            plt.bar([i for i in range(1, len(agent_specified_data_first)+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_first.values()], align="center", width=0.3)
            plt.bar([i+0.3 for i in range(1, len(agent_specified_data_second)+1)], [np.mean(value_scenario[key]) for value_scenario in agent_specified_data_second.values()], align="center", width=0.3)
    label_x = ["#"+str(i) for i in range(1, 20+1)]
    # plt.ylim(-1, 1)
    plt.xticks([i for i in range(1, 20+1)], label_x)
    # if rescale:
    #     if key=="rewards":
    #         plt.ylim([0.5, 0.6])
    #     elif key=="penalty_metric":
    #         plt.ylim([0.35, 0.45])
    if legend:
        plt.legend(loc=4)

#%% 
# 20シナリオpenaltyなしでaccuracy
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0"], legend=["prior work", "proposed method"],rescale=True)
#%% 
# 20シナリオpenalty0.0651の加算なしでaccuraty比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], all_agents_20_scenarios_alphazero["penalty_abs=0.0651"], legend=["prior work", "proposed method"],rescale=True)
#%%
# 20シナリオのpenaltyなしでpenalty metric比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0"], "penalty_metric")
#%%
# 20シナリオのpenalty0.0651でpenalty metric比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], all_agents_20_scenarios_alphazero["penalty_abs=0.0651"],"penalty_metric")
#%%
# 20シナリオのpenalty0.0369でpenalty metric比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0369"], all_agents_20_scenarios_alphazero["penalty_abs=0.0369"],"penalty_metric", rescale=True)

#%%
# 20シナリオのpenalty0.0369でaccuracy比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0369"], all_agents_20_scenarios_alphazero["penalty_abs=0.0369"],"rewards", rescale=True)

#%%
# 20シナリオのpenalty0のppoとpenaltya0.0651のalphazeroのrewards比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0.0651"],"rewards", rescale=True)

#%%
# 20シナリオのpenalty0のppoとpenaltya0.0651のalphazeroのpenalty比較
compare_reward_each_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0.0651"],"penalty_metric", rescale=True)


######### 10 scenarioでも実行 #########
#%%
# 10シナリオpenaltyなしでreward比較DLT
compare_reward_each_scenarios(all_agents_10_scenarios_ppo["penalty_abs=0"], all_agents_10_scenarios_alphazero["penalty_abs=0"])
#%% 
# 10シナリオpenalty0.0651ありで加算なしreward比較
compare_reward_each_scenarios(all_agents_10_scenarios_ppo["penalty_abs=0.0651"], all_agents_10_scenarios_alphazero["penalty_abs=0.0651"])

#%%
def add_true_reward_each_scenario(agent_fixed_data, penalty_abs):
    result = {}
    for key_scenario, value_metrics in agent_fixed_data.items():
        result[key_scenario] = value_metrics|{"true_rewards": true_reward(value_metrics, penalty_abs)}
    return result
# add_true_reward_each_scenario(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], 0.0651)
#%% 
# 20シナリオでpanelty0.0651加算でreward比較
compare_reward_each_scenarios(add_true_reward_each_scenario(all_agents_20_scenarios_ppo["penalty_abs=0.0651"], 0.0651), add_true_reward_each_scenario(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"], 0.0651),"true_rewards", legend=["prior work", "proposed method"])

######## 10 scenarioでも実行 #######
# %%
# 10シナリオでpanelty0.0651加算でreward比較
compare_reward_each_scenarios(add_true_reward_each_scenario(all_agents_10_scenarios_ppo["penalty_abs=0.0651"], 0.0651), add_true_reward_each_scenario(all_agents_10_scenarios_alphazero["penalty_abs=0.0651"], 0.0651),"true_rewards", legend=["prior work", "proposed method"])

#%% 
# def compare_rewards_with_sd(first_agent_fixed_data, second_agent_fixed_data):
#     labels = first_agent_fixed_data.keys()
#     firt_agent_rewards_means = [np.mean(metrics["rewards"]) for metrics in first_agent_fixed_data.values()]
#     second_agent_rewards_means = [np.mean(metrics["rewards"]) for metrics in second_agent_fixed_data.values()]
#     firt_agent_rewards_stds = [np.std(metrics["rewards"]) for metrics in first_agent_fixed_data.values()]
#     second_agent_rewards_stds = [np.std(metrics["rewards"]) for metrics in second_agent_fixed_data.values()]
#     fig, ax = plt.subplots()
#     ax.errorbar([labels, labels], [firt_agent_rewards_means, second_agent_rewards_means], [firt_agent_rewards_stds, second_agent_rewards_stds])
# compare_rewards_with_sd(all_agents_20_scenarios_ppo["penalty_abs=0"], all_agents_20_scenarios_alphazero["penalty_abs=0"])

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
# 正解率(PCS)の標準誤差を出力
zero_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0"])["rewards"])/2+1/2, return_counts=True)[1][0]
one_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0"])["rewards"])/2+1/2, return_counts=True)[1][1]
alpha = one_num+1
beta = zero_num+1
np.sqrt((beta*alpha)/(((beta+alpha)**2)*(beta+alpha+1)))
# %%
zero_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"])["rewards"])/2+1/2, return_counts=True)[1][0]
one_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0"])["rewards"])/2+1/2, return_counts=True)[1][1]
alpha = one_num+1
beta = zero_num+1
np.sqrt((beta*alpha)/(((beta+alpha)**2)*(beta+alpha+1)))
# %%
np.mean(np.array(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["rewards"])/2+1/2)
# %%
# 正解率(PCS)の標準誤差を出力
zero_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["rewards"])/2+1/2, return_counts=True)[1][0]
one_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_alphazero["penalty_abs=0.0651"])["rewards"])/2+1/2, return_counts=True)[1][1]
alpha = one_num+1
beta = zero_num+1
np.sqrt((beta*alpha)/(((beta+alpha)**2)*(beta+alpha+1)))
# %%
np.mean(np.array(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["rewards"])/2+1/2)
#%%
zero_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["rewards"])/2+1/2, return_counts=True)[1][0]
one_num = np.unique(np.array(merge_scenarios(all_agents_20_scenarios_ppo["penalty_abs=0.0651"])["rewards"])/2+1/2, return_counts=True)[1][1]
alpha = one_num+1
beta = zero_num+1
np.sqrt((beta*alpha)/(((beta+alpha)**2)*(beta+alpha+1)))
# %%
# それぞれのペナルティで，平均DLTsを計算
def compare_DLTs_each_penalties(agent_first, agent_second, key='DLTs', legend=None, rescale=False):
    i=1
    for key_agent in sorted(agent_first.keys()):
        # print(-np.mean(merge_scenarios(agent_first[key_agent])[key]))
        # print(np.mean(merge_scenarios(agent_second[key_agent])[key]))
        if i%2==0:
            print(r"\rowcolor[rgb]{0.9, 0.9, 0.9}")

        if -np.mean(merge_scenarios(agent_first[key_agent])[key])+np.mean(merge_scenarios(agent_second[key_agent])[key])>0:
            print(f"{float(key_agent[12:]):.4f} & {np.mean(merge_scenarios(agent_first[key_agent])[key]):.2f} & $\\bm{{{np.mean(merge_scenarios(agent_second[key_agent])[key]):.2f}}}$ & {np.mean(merge_scenarios(agent_second[key_agent])[key])-np.mean(merge_scenarios(agent_first[key_agent])[key]):.2f} \\\\")
        elif -np.mean(merge_scenarios(agent_first[key_agent])[key])+np.mean(merge_scenarios(agent_second[key_agent])[key])<0:
            print(f"{float(key_agent[12:]):.4f} & $\\bm{{{np.mean(merge_scenarios(agent_first[key_agent])[key]):.2f}}}$ & {np.mean(merge_scenarios(agent_second[key_agent])[key]):.2f} & {np.mean(merge_scenarios(agent_second[key_agent])[key])-np.mean(merge_scenarios(agent_first[key_agent])[key]):.2f} \\\\")
        else:
            print(f"{float(key_agent[12:]):.4f} & {np.mean(merge_scenarios(agent_first)[key_agent]):.4f} & {np.mean(merge_scenarios(agent_second)[key_agent]):.4f} & {np.mean(merge_scenarios(agent_second)[key_agent])-np.mean(merge_scenarios(agent_first)[key_agent]):.4f} \\\\")

        i += 1
            
    # if np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key])>0:
    #     print(f"average & {np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2:.4f} & $\\bm{{{np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2:.4f}}}$ & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2-(np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2):.4f} \\\\")
    # elif np.mean(merge_scenarios(agent_specified_data_second)[key])-np.mean(merge_scenarios(agent_specified_data_first)[key])<0:
    #     print(f"average & $\\bm{{{np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2:.4f}}}$ & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2:.4f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2-(np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2):.4f} \\\\")
    # else:
    #     print(f"average & {np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2:.4f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2:.4f} & {np.mean(merge_scenarios(agent_specified_data_second)[key])/2+1/2-(np.mean(merge_scenarios(agent_specified_data_first)[key])/2+1/2):.4f} \\\\")
#%%
compare_DLTs_each_penalties(all_agents_20_scenarios_ppo, all_agents_20_scenarios_alphazero)
# %%
