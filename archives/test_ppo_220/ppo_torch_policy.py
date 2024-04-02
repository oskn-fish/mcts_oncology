import logging
from typing import Dict, List, Type, Union

import ray
from test_ppo_220.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)
import test_ppo_220
import ppo_mcts
from ray.rllib.policy.policy import Policy
from alpha_zero_bayes.mcts import StateNode, RootParentNode
import copy
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from alpha_zero_bayes.mcts import MCTS
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    GradInfoDict,
    ModelGradients,
    ModelWeights,
    PolicyState,
    TensorStructType,
    TensorType,
)

if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode  # noqa


class PPOTorchPolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(test_ppo_220.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # TODO: Don't require users to call this manually.
        # self._initialize_loss_from_dummy_batch()
        """added"""

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

        _, env_creator = Algorithm._get_env_id_and_creator(config["env"], config)

        def _env_creator():
            return env_creator(config["env_config"])

        def mcts_creator():
            return MCTS(
                self.model, config["mcts_config"]
            )  # 上でTorchPolicyV2.__init__してるからself.modelがあるはず

        self.env_creator = _env_creator
        self.env = self.env_creator()
        self.env.reset()
        self.mcts_creator = mcts_creator
        self.mcts = mcts_creator()

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

    @override(Policy)
    # @DeveloperAPI
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            # input_dict = {"obs": obs_batch}
            input_dict = self._lazy_tensor_dict(
                {
                    SampleBatch.CUR_OBS: obs_batch,
                    "is_training": False,
                }
            )
            # if prev_action_batch is not None:
            #     input_dict["prev_actions"] = prev_action_batch
            # if prev_reward_batch is not None:
            #     input_dict["prev_rewards"] = prev_reward_batch
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = np.asarray(prev_action_batch)
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = np.asarray(prev_reward_batch)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            with torch.no_grad():
                input_dict = self._lazy_tensor_dict(input_dict)
                input_dict.set_training(True)
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
                # self.extra_action_outで，model.value_function()をする．
                # valu_functionでは前回のforwardでvalueにのみ用いられる部分の演算のみを行うから，前もってforwardしておかなくてはいけない．
                # TODO: もっといい感じの実装にする（結果を捨てている）
                # lossにつかわれるかもなので，nogradからは外す
        logits, state = self.model.forward(input_dict, None, None)

        return self._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )

    def compute_actions_from_input_dict(
        self,
        input_dict: Dict[str, TensorType],
        explore: bool = None,
        timestep: Optional[int] = None,
        episodes=None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        with torch.no_grad():
            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            # Pack internal state inputs into (separate) list.
            state_batches = [
                input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
            ]
            # Calculate RNN sequence lengths.
            seq_lens = (
                torch.tensor(
                    [1] * len(state_batches[0]),
                    dtype=torch.long,
                    device=state_batches[0].device,
                )
                if state_batches
                else None
            )

            # input_dict = self._lazy_tensor_dict(input_dict)
            # input_dict.set_training(True)
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
            # self.extra_action_outで，model.value_function()をする．
            # valu_functionでは前回のforwardでvalueにのみ用いられる部分の演算のみを行うから，前もってforwardしておかなくてはいけない．
            # TODO: もっといい感じの実装にする（結果を捨てている）
            # lossにつかわれるかもなので，nogradからは外す
        logits, state = self.model.forward(input_dict, None, None)

        return self._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )
