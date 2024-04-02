import numpy as np

# from ray.rllib.algorithms.alpha_zero.mcts import Node, RootParentNode
from alpha_zero_bayes.mcts import StateNode, RootParentNode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY

torch, _ = try_import_torch()

import alpha_zero_bayes.constants as constants

# from alpha_zero_bayes.mcts_bayes import tuple_to_posterior

from ray.rllib.models.modelv2 import restore_original_dimensions

import os
import pickle
import copy


class AlphaZeroPolicy(TorchPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        config,
        model,
        loss,
        action_distribution_class,
        mcts_creator,
        env_creator,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            config,
            model=model,
            loss=loss,
            action_distribution_class=action_distribution_class,
        )
        # we maintain an env copy in the policy that is used during mcts
        # simulations
        self.env_creator = env_creator
        self.mcts = mcts_creator()
        self.env = self.env_creator()
        self.env.reset()
        self.obs_space = observation_space

    @override(TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        input_dict = {"obs": obs_batch}
        if prev_action_batch is not None:
            input_dict["prev_actions"] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict["prev_rewards"] = prev_reward_batch

        return self.compute_actions_from_input_dict(
            input_dict=input_dict,
            episodes=episodes,
            state_batches=state_batches,
        )

    @override(Policy)
    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, **kwargs
    ):
        with torch.no_grad():
            actions = []

            """
            基本，前回のtreeを引き継いで使用するために，action nodeを記録しておく．
            
            前回のtree引き継がんときは新しくnodeを作る
            """
            for i, episode in enumerate(episodes):
                if episode.length == 0:
                    # if first time step of episode, get initial env state
                    env_state = copy.deepcopy(episode.user_data["initial_state"])
                    obs = self.env.set_state(env_state)
                    tree_node = StateNode(
                        obs=obs,
                        done=False,
                        parent=RootParentNode(env=self.env),
                        mcts=self.mcts,
                        reward=0,
                    )

                else:
                    # otherwise get last root node from previous time step

                    action_node = episode.user_data["action_node"]
                    last_obs = episode.user_data["on_episode_step_last_obs"]
                    tupled_obs = tuple(last_obs)

                    if action_node and tupled_obs in action_node.children:
                        tree_node = action_node.children[tupled_obs]
                    else:
                        tree_node = StateNode(
                            obs=last_obs,
                            done=False,
                            parent=RootParentNode(env=self.env),
                            mcts=self.mcts,
                            reward=0,
                        )

                # run monte carlo simulations to compute the actions
                # and record the tree mcts_policy, action, tree_node = self.mcts.compute_action(tree_node)
                mcts_policy, action = self.mcts.compute_action(tree_node)
                # record action
                actions.append(action)
                # store new node when not final time step
                if action in tree_node.children:
                    episode.user_data["action_node"] = tree_node.children[action]
                else:
                    episode.user_data["action_node"] = None

                # store mcts policies vectors and current tree root node
                if episode.length == 0:
                    episode.user_data["mcts_policies"] = [mcts_policy]
                else:
                    episode.user_data["mcts_policies"].append(mcts_policy)

            return (
                np.array(actions),
                [],
                self.extra_action_out(
                    input_dict, kwargs.get("state_batches", []), self.model, None
                ),
            )

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # add mcts policies to sample batch
        sample_batch["mcts_policies"] = np.array(episode.user_data["mcts_policies"])[
            sample_batch["t"]
        ]
        # final episode reward corresponds to the value (if not discounted)
        # for all transitions in episode
        final_reward = sample_batch["rewards"][-1]
        # if r2 is enabled, then add the reward to the buffer and normalize it
        if self.env.__class__.__name__ == "RankedRewardsEnvWrapper":
            self.env.r2_buffer.add_reward(final_reward)
            final_reward = self.env.r2_buffer.normalize(final_reward)
        sample_batch["value_label"] = final_reward * np.ones_like(
            sample_batch["t"]
        )  # single-player用っぽい？？
        if "penalty_metric" in episode.user_data:
            sample_batch["penalty_metric"] = np.array(episode.user_data["penalty_metric"])
        if "penalty" in episode.user_data:
            sample_batch["penalty"] = np.array(episode.user_data["penalty"])
        if "MTD_metric" in episode.user_data:
            sample_batch["MTD_metric"] = np.array(episode.user_data["MTD_metric"])
        if "DLTs" in episode.user_data:
            sample_batch["DLTs"] = np.array(episode.user_data["DLTs"])

        # assert False
        # add one hotted actions (actions>=3 -> 0)
        # actions = sample_batch["actions"]
        # actions_onehot = np.identity(self.env.action_space.n)[actions]
        # actions_onehot = np.delete(actions_onehot, np.s_[3:], 1)
        # sample_batch["actions_onehot"] = actions_onehot

        """
        sample_batch["obs"] = np.array([[action_mask+obs] for t in timestep])
        
        dim = 28 = action_mask(13) + current_dose(1) + Ns(6) + DLTs(6) + other(2)
        """
        # new_obses = sample_batch["new_obs"]
        # obses = sample_batch["obs"]

        # len_action_mask = self.env.action_space.n
        # len_current_dose = 1
        # start = len_action_mask+len_current_dose+self.env.D
        # stop = start+self.env.D

        # new_DLTs = np.array([obs[start:stop]*self.env.N_total for obs in new_obses], dtype=int)
        # DLTs = np.array([obs[start:stop]*self.env.N_total for obs in obses], dtype=int)
        # diff_DLTs = new_DLTs-DLTs

        # assert len(actions)==len(diff_DLTs)
        # actionした結果のDLT人数
        # action_DLTs = [0, 2, 1, ...]
        # action_DLTs = np.hstack([np.array([diff_DLTs[i][actions[i]] for i in range(len(actions)-1)]),np.array([0])]) # 最終のactionは普通と違って3種類じゃない，

        # action_DLTs_onehot = np.identity(self.env.N_cohort+1)[action_DLTs]
        # sample_batch["action_DLTs_onehot"] = action_DLTs_onehot

        return sample_batch

    # @override(TorchPolicy)
    # def learn_on_batch(self, postprocessed_batch):
    #     train_batch = self._lazy_tensor_dict(postprocessed_batch)

    #     loss_out = self._loss(self, self.model, self.dist_class, train_batch)

    #     self._optimizers[0].zero_grad()
    #     loss_out.backward()

    #     grad_process_info = self.extra_grad_process(self._optimizers[0], loss_out)
    #     self._optimizers[0].step()

    #     grad_info = self.extra_grad_info(train_batch)
    #     grad_info.update(grad_process_info)
    #     grad_info.update(
    #         {
    #             "total_loss": loss_out.detach().cpu().numpy(),
    #         }
    #     )

    #     return {LEARNER_STATS_KEY: grad_info}
