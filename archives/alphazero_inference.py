# %%
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import ray
from ray.rllib.evaluation.episode import Episode
# from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue import (
#     RLEEnv_expectation_negative_reward_penalty_continue,
# )
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate

from ray.rllib.algorithms import Algorithm


@ray.remote
def excecute_multi_trial(num_trials, scenarioID):
    # policy = Policy.from_checkpoint(
    #     "/home/chinen/esc_mcts/checkpoint_test_alphazero_learn_for_inference/checkpoint_554/policies/default_policy"
    # )
    policy = Policy.from_checkpoint(
        "/home/chinen/esc_mcts/alphazero_best_config_negative_reward/1151/policies/default_policy"
    )

    env_config = {
        "D": 6,
        "N_cohort": 3,
        "N_total": 36,
        "phi": 0.25,
        # "DLT_max": 1,  
        # "penalty_abs": 100,  
        "DLT_max": 1000,
        "penalty_abs": 0,
        "scenario": scenarioID
    }
    # policy.env = RLEEnv_expectation_negative_reward_penalty_continue(env_config)
    # env = RLEEnv_expectation_negative_reward_penalty_continue(env_config)
    env = RLEEnv_expectation_negative_reward_penalty_continue_evaluate(env_config)
    rewards = []

    for _ in range(num_trials):
        obs, info = env.reset()
        done = False
        episode = Episode(
            PolicyMap(),
            lambda _, __: DEFAULT_POLICY_ID,
            lambda: None,
            lambda _: None,
            0,
        )
        episode.user_data["initial_state"] = env.get_state()
        while not done:
            print(f"{obs=}")
            action, _, _ = policy.compute_single_action(obs, episode=episode)
            print(f"{action=}")
            obs, reward, done, truncated, info = env.step(action)
            episode.length += 1
            episode.user_data["on_episode_step_last_obs"] = obs
        print(f"{reward=}")
        rewards.append(reward)
    return rewards


ress = [excecute_multi_trial.remote(1000) for _ in range(110)]
results = []
for res in ress:
    results += ray.get(res)
print(f"{results=}")
print(f"{sum(results) / len(results)}")

# print(sum(rewards) / len(rewards))
# -73.53833333333333
