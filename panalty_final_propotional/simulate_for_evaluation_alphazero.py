# %%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
# from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import RLEEnv_expectation_negative_reward_penalty_continue
# from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate
from RLE.env.RLEEnv_expectation_negative_reward_penalty_proprtional import RLEEnv_expectation_negative_reward_penalty_propotional
from RLE.env.RLEEnv_expectation_negative_reward_penalty_proprtional_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_propotional
from ray.rllib.policy.policy import Policy

from torch.nn import Softmax
import torch
import numpy as np
from tqdm import tqdm
import re
import pickle

from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap

# %%

# 各シナリオについてシミュレーション（Matsuuraと同じ設定）
# reward, DLT, penatlty_metric の辞書を返す
@ray.remote
def remote_sample_simulations(agent_path, env_func, env_config, num_simulations):
    
    algo = Policy.from_checkpoint(agent_path)
    metrics = {"DLTs":[], "penalty":[], "MTD_metric":[], "rewards":[]}
    env = env_func(env_config)
    for _ in range(num_simulations):
        done = False
        observation, info = env.reset()
        
        if "alphazero" in agent_path.lower():
            episode = Episode(
                PolicyMap(),
                lambda _, __: DEFAULT_POLICY_ID,
                lambda: None,
                lambda _: None,
                0,
            )
            episode.user_data["initial_state"] = env.get_state()
        else:
            assert "ppo" in agent_path.lower()
        
        while not done:
            if "alphazero" in agent_path.lower():
                action, _, _ = algo.compute_single_action(observation, episode=episode)
                observation, reward, done, truncated, info = env.step(action)
                episode.length += 1
                episode.user_data["on_episode_step_last_obs"] = observation
            else:
                infos = algo.compute_single_action(observation)
                tensor_action_dist = Softmax(dim=0)(torch.from_numpy(infos[2]["action_dist_inputs"]).clone())
                nd_array_action_dist = tensor_action_dist.to('cpu').detach().numpy().copy()
                action = np.random.choice(10, 1, p=nd_array_action_dist)[0]
                observation, reward, done, truncated, info = env.step(action)
        metrics["rewards"].append(reward)
        metrics["DLTs"].append(info["DLTs"])
        metrics["penalty"].append(info["penalty"])
        metrics["MTD_metric"].append(info["MTD_metric"])
    assert len(metrics["rewards"])==num_simulations
    return metrics

# 10000シミュレーションを分散処理によってサンプリング
# reward, DLT, penatlty_metric の辞書を返す
def sample_env_simulations(agent_path, env_func, env_config):
    NUM_SIMULATIONS = 10000
    NUM_WORKERS = 125
    simulations_per_worker = NUM_SIMULATIONS//NUM_WORKERS
    
    result_parallel = [remote_sample_simulations.remote(agent_path, env_func, env_config, simulations_per_worker) for _ in range(NUM_WORKERS)]
    result_parallel = ray.get(result_parallel) # rayはリストも受け取れる
    final_result = {}
    for result in result_parallel:
        for key, value in result.items():
            if key not in final_result: final_result[key] = []
            final_result[key].extend(value)
    return final_result

# 学習用の20シナリオについて，それぞれシミュレーションサンプリング
def evaluate_on_20_senarios(agent_path):
    results = {}
    
    for scenario in tqdm(range(20), leave=False):
        env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25, "scenario": scenario, "DLT_max": 10, "penalty_coefficient_abs": 0}
        result = sample_env_simulations(agent_path, RLEEnv_expectation_negative_reward_penalty_propotional, env_config)
        results["scenario_"+str(scenario)] = result
    return results
    
# 評価用の10シナリオについて，それぞれシミュレーションサンプリング  
def evaluate_on_10_senarios(agent_path):
    results = {}
    
    for scenario in tqdm(range(10), leave=False):
        env_config = {"D": 6, "N_cohort": 3, "N_total": 36, "phi": 0.25, "scenario": scenario, "DLT_max": 10, "penalty_coefficient_abs": 0}
        # RLEEnv_expectation_negative_reward_penalty_continue_evaluate
        result = sample_env_simulations(agent_path, RLEEnv_expectation_negative_reward_penalty_continue_propotional, env_config)
        results["scenario_"+str(scenario)] = result
    return results    


# checkpointのフォルダ名からpenalty_abs部分を抽出して，キーに入れる
def sample(list_checkpoint_path, FILE_NAME):
    pattern = ".*(penalty_coefficient_abs=0\.?\d*).*"

    all_agents_20_scenarios = {}
    for agent_path in tqdm(list_checkpoint_path):
        all_agents_20_scenarios[re.match(pattern, agent_path).group(1)] = evaluate_on_20_senarios(agent_path)

    all_agents_10_scenarios = {}
    for agent_path in tqdm(list_checkpoint_path):
        all_agents_10_scenarios[re.match(pattern, agent_path).group(1)] = evaluate_on_10_senarios(agent_path)

    # pickleで保存
    with open(os.path.join(os.path.dirname(__file__),FILE_NAME), 'wb') as f:
        pickle.dump({"all_agents_20_scenarios":all_agents_20_scenarios, "all_agents_10_scenarios":all_agents_10_scenarios}, f)


# path_checkpoint_root_ppo = "/home/chinen/esc_mcts/panalty_final_propotional/PPO_2024-01-27_20-20-13/"
# list_checkpoint_path_ppo = [path_checkpoint_root_ppo+os.path.join(path,"checkpoint_000000/policies/default_policy") for path in os.listdir(path_checkpoint_root_ppo) if os.path.isdir(os.path.join(path_checkpoint_root_ppo, path))]

# if not os.path.isfile(os.path.join(os.path.dirname(__file__),'all_agents_all_scenarios_ppo.pickle')):
#     FILE_NAME = "all_agents_all_scenarios_ppo.pickle"
#     sample(list_checkpoint_path_ppo, FILE_NAME)
    
path_checkpoint_root_alphazero = "/home/chinen/esc_mcts/panalty_final_propotional/AlphaZero_2024-01-27_20-04-45/"
list_checkpoint_path_alphazero = [path_checkpoint_root_alphazero+os.path.join(path,"checkpoint_000000/policies/default_policy") for path in os.listdir(path_checkpoint_root_alphazero) if os.path.isdir(os.path.join(path_checkpoint_root_alphazero, path))]

if not os.path.isfile(os.path.join(os.path.dirname(__file__),'all_agents_all_scenarios_alphazero.pickle')):
    FILE_NAME = "all_agents_all_scenarios_alphazero.pickle"
    sample(list_checkpoint_path_alphazero, FILE_NAME)