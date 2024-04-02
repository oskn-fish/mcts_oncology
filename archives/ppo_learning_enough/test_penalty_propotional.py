from RLE.env.RLEEnv_expectation_negative_reward_penalty_proprtional import RLEEnv_expectation_negative_reward_penalty_propotional
# from alpha_zero_bayes.mcts import MCTS


from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap

from ray.rllib.policy.policy import Policy

env_config = {
    "D": 6,
    "N_cohort": 3,
    "N_total": 36,
    "phi": 0.25,
    "DLT_max": 10,
    # "penalty_coefficient_abs": tune.grid_search([0.06513854285023316, 0.05721842600035547, 0.04993262476736769, 0.043187026441359, 0.03690702464285426, 0.031032475286063992, 0.02551418439839037, 0.020311401357501223, 0.015389990790009425, 0])
    "penalty_coefficient_abs": 0.02834043767146469
    # "penalty_abs": tune.grid_search([i*1e-3 for i in range(1, 10)])
}
env = RLEEnv_expectation_negative_reward_penalty_propotional(env_config)

policy_path = "/home/chinen/esc_mcts/panalty_final_propotional/AlphaZero_2024-01-19_20-34-15/AlphaZero_RLEEnv_expectation_negative_reward_penalty_continue_acb5b_00000_0_penalty_abs=0.0651_2024-01-19_20-34-15/checkpoint_000000/policies/default_policy"

for i in range(10):
    algo = Policy.from_checkpoint(policy_path)
    observation, info = env.reset()
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
        action, _, _ = algo.compute_single_action(observation, episode=episode)
        observation, reward, done, truncated, info = env.step(action)
        episode.length += 1
        episode.user_data["on_episode_step_last_obs"] = observation
        
    # print(obs)
    print(reward)
    print(reward +obs[-2]*36*0.02834043767146469)
