import logging
from typing import List, Optional, Type, Union

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    multi_gpu_train_one_step,
    train_one_step,
)
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import override

# from ray.rllib.utils.deprecation import (
#     DEPRECATED_VALUE,
#     Deprecated,
#     ALGO_DEPRECATION_WARNING,
# )
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import validate_buffer_config
from ray.rllib.utils.typing import PolicyID, ResultDict

# from ray.rllib.algorithms.alpha_zero.alpha_zero_policy import AlphaZeroPolicy
from alpha_zero_bayes.bayes_alpha_zero_policy import AlphaZeroPolicy

# from ray.rllib.algorithms.alpha_zero.mcts import MCTS
from alpha_zero_bayes.mcts import MCTS

# from ray.rllib.algorithms.alpha_zero.ranked_rewards import get_r2_env_wrapper
from alpha_zero_bayes.ranked_rewards import get_r2_env_wrapper

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

import os
import pickle
import numpy as np

# import torch.autograd.profiler as profiler


class AlphaZeroDefaultCallbacks(DefaultCallbacks):
    """AlphaZero callbacks.

    If you use custom callbacks, you must extend this class and call super()
    for on_episode_start.
    """

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        # Save environment's state when an episode starts.
        env = base_env.get_sub_environments()[0]
        # state = env.get_state()
        state = env.get_state()
        episode.user_data["initial_state"] = state
        # initial_state = env.get_state()
        # state_for_mcts = {key:value for key, value in initial_state.items() if key in ["last_obs", "current_dose", "N_done_patients", "last_reward", "D", "phi","N_cohort"]}
        # episode.user_data["initial_state"] = state_for_mcts

    def on_episode_step(self, worker, base_env, policies, episode, env_index, **kwargs):
        # Save environment's state when an episode starts.
        env = base_env.get_sub_environments()[0]
        episode.user_data[
            "on_episode_step_last_obs"
        ] = env.get_state()  # env.last_obs #
        if hasattr(env, "DLT_max"):
            if episode.length == 1:
                if "penalty_metric" in episode._last_infos["agent0"].keys():
                    episode.user_data["penalty_metric"] = [
                        episode._last_infos["agent0"]["penalty_metric"]
                    ]
                if "penalty" in episode._last_infos["agent0"].keys():
                    episode.user_data["penalty"] = [
                        episode._last_infos["agent0"]["penalty"]
                    ]
                episode.user_data["MTD_metric"] = [
                    episode._last_infos["agent0"]["MTD_metric"]
                ]
                episode.user_data["DLTs"] = [
                    episode._last_infos["agent0"]["DLTs"]
                ]
            else:
                if "penalty_metric" in episode._last_infos["agent0"].keys():
                    episode.user_data["penalty_metric"].append(
                        episode._last_infos["agent0"]["penalty_metric"]
                    )
                if "penalty" in episode._last_infos["agent0"].keys():
                    episode.user_data["penalty"].append(
                        episode._last_infos["agent0"]["penalty"]
                    )
                episode.user_data["MTD_metric"].append(
                    episode._last_infos["agent0"]["MTD_metric"]
                )
                episode.user_data["DLTs"].append(
                    episode._last_infos["agent0"]["DLTs"]
                )
            # if len(episode.user_data["MTD_policies"]) == 1:
            #     episode.user_data["MTD_metric"] = [
            #         episode._last_infos["agent0"]["MTD_metric"]
            #     ]
            # else:
            #     episode.user_data["MTD_metric"].append(
            #         episode._last_infos["agent0"]["MTD_metric"]
            #     )

        # episode.user_data["penalty_metric"] = 0

    # def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
    #     pass
    # env = base_env.get_sub_environments()[0]
    # if hasattr(env, "DLT_max"):
    #     episode.user_data["penalty_metric"] = env.get_penalty_metric()

    # def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
    #     # database 読み込み & set to mcts
    #     data_path = os.path.join(os.path.dirname(__file__), "database_posterior.pickle")
    #     if os.path.isfile(data_path):
    #         with open(data_path, "rb") as f:
    #             policy.mcts.database_posterior = pickle.load(f)
    #     else:
    #         policy.mcts.database_posterior = {}

    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
    #     data_path = os.path.join(os.path.dirname(__file__), "database_posterior.pickle")
    #     with open(data_path, "wb") as f:
    #             pickle.dump(self.mcts.database_posterior, f)


class AlphaZeroConfig(AlgorithmConfig):
    """Defines a configuration class from which an AlphaZero Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
        >>> config = AlphaZeroConfig()   # doctest: +SKIP
        >>> config = config.training(sgd_minibatch_size=256)   # doctest: +SKIP
        >>> config = config..resources(num_gpus=0)   # doctest: +SKIP
        >>> config = config..rollouts(num_rollout_workers=4)   # doctest: +SKIP
        >>> print(config.to_dict()) # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")  # doctest: +SKIP
        >>> algo.train() # doctest: +SKIP

    Example:
        >>> from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = AlphaZeroConfig()
        >>> # Print out some default values.
        >>> print(config.shuffle_sequences) # doctest: +SKIP
        >>> # Update the config object.
        >>> config.training(lr=tune.grid_search([0.001, 0.0001]))  # doctest: +SKIP
        >>> # Set the config object's env.
        >>> config.environment(env="CartPole-v1")   # doctest: +SKIP
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner( # doctest: +SKIP
        ...     "AlphaZero",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or AlphaZero)

        # fmt: off
        # __sphinx_doc_begin__
        # AlphaZero specific config settings:
        self.sgd_minibatch_size = 128
        self.shuffle_sequences = True
        # self.num_sgd_iter = 6
        self.num_sgd_iter = 30
        self.replay_buffer_config = {
            "type": "ReplayBuffer",
            # Size of the replay buffer in batches (not timesteps!).
            "capacity": 1024,
            # "capacity": 4000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, pisodes or timesteps.
            # "storage_unit": "fragments",
            "storage_unit": "timesteps"
        }
        # Number of timesteps to collect from rollout workers before we start
        # sampling from replay buffers for learning. Whether we count this in agent
        # steps  or environment steps depends on config.multi_agent(count_steps_by=..).
        # self.num_steps_sampled_before_learning_starts = 1000
        self.num_steps_sampled_before_learning_starts = 1024
        self.lr_schedule = None
        self.vf_share_layers = True
        self.mcts_config = {
            "puct_coefficient": 1.0,
            "num_simulations": 50,
            "temperature": 1.5,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
            "temp_threshold": 2,
            "use_Q": False
        }
        self.ranked_rewards = {
            "enable": True,
            "percentile": 75,
            "buffer_max_length": 1000,
            # add rewards obtained from random policy to
            # "warm start" the buffer
            "initialize_buffer": True,
            "num_init_rewards": 100,
        }

        # Override some of AlgorithmConfig's default values with AlphaZero-specific
        # values.
        self.framework_str = "torch"
        self.callbacks_class = AlphaZeroDefaultCallbacks
        self.lr = 5e-5
        self.num_rollout_workers = 6
        self.rollout_fragment_length = 200
        # self.train_batch_size = 4000
        self.train_batch_size = 4096
        self.batch_mode = "complete_episodes"
        # Extra configuration that disables exploration.
        self.evaluation(evaluation_config={
            "mcts_config": {
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
            },
        })
        self.exploration_config = {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "StochasticSampling",
            # Add constructor kwargs here (if any).
        }
        # __sphinx_doc_end__
        # fmt: on

        self.buffer_size = DEPRECATED_VALUE
        self.l2_coeff = 0.0
        self.epsilon = 1e-12
        self.policy_coeff = 0.5

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        sgd_minibatch_size: Optional[int] = NotProvided,
        shuffle_sequences: Optional[bool] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
        replay_buffer_config: Optional[dict] = NotProvided,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        vf_share_layers: Optional[bool] = NotProvided,
        mcts_config: Optional[dict] = NotProvided,
        ranked_rewards: Optional[dict] = NotProvided,
        num_steps_sampled_before_learning_starts: Optional[int] = NotProvided,
        l2_coeff: Optional[float] = NotProvided,
        epsilon: Optional[float] = NotProvided,
        train_batch_size: Optional[float] = NotProvided,
        policy_coeff: Optional[float] = NotProvided,
        **kwargs,
    ) -> "AlphaZeroConfig":
        """Sets the training related configuration.

        Args:
            sgd_minibatch_size: Total SGD batch size across all devices for SGD.
            shuffle_sequences: Whether to shuffle sequences in the batch when training
                (recommended).
            num_sgd_iter: Number of SGD iterations in each outer loop.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "learning_starts": 1000,
                "capacity": 50000,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            vf_share_layers: Share layers for value function. If you set this to True,
                it's important to tune vf_loss_coeff.
            mcts_config: MCTS specific settings.
            ranked_rewards: Settings for the ranked reward (r2) algorithm
                from: https://arxiv.org/pdf/1807.01672.pdf
            num_steps_sampled_before_learning_starts: Number of timesteps to collect
                from rollout workers before we start sampling from replay buffers for
                learning. Whether we count this in agent steps  or environment steps
                depends on config.multi_agent(count_steps_by=..).

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if shuffle_sequences is not NotProvided:
            self.shuffle_sequences = shuffle_sequences
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if replay_buffer_config is not NotProvided:
            self.replay_buffer_config = replay_buffer_config
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if vf_share_layers is not NotProvided:
            self.vf_share_layers = vf_share_layers
        if mcts_config is not NotProvided:
            self.mcts_config = mcts_config
        if ranked_rewards is not NotProvided:
            self.ranked_rewards.update(ranked_rewards)
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
            # self.num_steps_sampled_before_learning_starts = (
            #     replay_buffer_config["capacity"]
            # )
        if l2_coeff is not NotProvided:
            self.l2_coeff = l2_coeff
        if epsilon is not NotProvided:
            self.epsilon = epsilon
        if train_batch_size is not NotProvided:
            self.train_batch_size = train_batch_size
        if policy_coeff is not NotProvided:
            self.policy_coeff = policy_coeff
        return self

    @override(AlgorithmConfig)
    def update_from_dict(self, config_dict) -> "AlphaZeroConfig":
        config_dict = config_dict.copy()

        if "ranked_rewards" in config_dict:
            value = config_dict.pop("ranked_rewards")
            self.training(ranked_rewards=value)

        return super().update_from_dict(config_dict)

    @override(AlgorithmConfig)
    def validate(self) -> None:
        """Checks and updates the config based on settings."""
        # Call super's validation method.
        super().validate()
        validate_buffer_config(self)


# def alpha_zero_loss(policy, model, dist_class, train_batch):
#     # get inputs unflattened inputs
#     input_dict = restore_original_dimensions(
#         train_batch["obs"], policy.observation_space, "torch"
#     )
#     # forward pass in model
#     model_out = model.forward(input_dict, None, [1])
#     logits, _ = model_out
#     values = model.value_function()
#     logits, values = torch.squeeze(logits), torch.squeeze(values)
#     priors = nn.Softmax(dim=-1)(logits)
#     # compute actor and critic losses
#     policy_loss = torch.mean(
#         -torch.sum(train_batch["mcts_policies"] * torch.log(priors), dim=-1)
#     )
#     value_loss = torch.mean(torch.pow(values - train_batch["value_label"], 2))
#     # compute total loss
#     total_loss = (policy_loss + value_loss) / 2
#     return total_loss, policy_loss, value_loss


def alpha_zero_loss(policy, model, dist_class, train_batch):
    # get inputs unflattened inputs
    input_dict = restore_original_dimensions(
        train_batch["obs"], policy.observation_space, "torch"
    )
    # forward pass in model
    model_out = model.forward(input_dict, None, [1])
    logits, _ = model_out
    values = model.value_function()
    # prediction_logits = model.prediction_logits_function()
    # logits, values, prediction_logits = torch.squeeze(logits), torch.squeeze(values), torch.squeeze(prediction_logits)
    logits, values = torch.squeeze(logits), torch.squeeze(values)
    priors = nn.Softmax(dim=-1)(logits)
    # prediction = nn.Softmax(dim=1)(prediction_logits)
    # values_sig = nn.Sigmoid()(values)
    # compute actor and critic losses
    """
    tensorの*はhadamard productであることに注意．
    """
    policy_loss = torch.mean(
        # -torch.sum(train_batch["mcts_policies"] * torch.log(priors+torch.finfo(priors.dtype).tiny), dim=-1)
        -torch.sum(
            train_batch["mcts_policies"] * torch.log(priors + policy.config["epsilon"]),
            dim=-1,
        )
    )
    # value_loss = torch.mean(torch.pow(values_sig - train_batch["value_label"], 2))
    value_loss = torch.mean(torch.pow(values - train_batch["value_label"], 2))
    # prediction_loss = torch.mean(
    #     # -torch.sum(train_batch["actions_onehot"].unsqueeze(1) @ torch.log(prediction+torch.finfo(prediction.dtype).tiny) @ train_batch["action_DLTs_onehot"].unsqueeze(2))
    #     -torch.sum(train_batch["actions_onehot"].unsqueeze(1) @ torch.log(prediction+policy.config["epsilon"]) @ train_batch["action_DLTs_onehot"].unsqueeze(2))
    # )

    # reguralization https://qiita.com/tabintone/items/790729a89ed84bb21b74#l2%E6%AD%A3%E5%89%87%E5%8C%96ridge%E5%9B%9E%E5%B8%B0
    # compute total loss
    # total_loss = (policy_loss + value_loss) / 2
    policy_and_value_loss = 2 * (
        policy.config["policy_coeff"] * policy_loss
        + (1 - policy.config["policy_coeff"]) * value_loss
    )
    if policy.config["l2_coeff"] > 0.0:
        l2 = torch.tensor(0.0, requires_grad=True)
        for w in model.parameters():
            l2 = l2 + torch.norm(w) ** 2
        reguralization_loss = policy.config["l2_coeff"] * l2
        # total_loss = (policy_loss + value_loss + prediction_loss + reguralization_loss) / 4
        total_loss = (policy_and_value_loss + reguralization_loss) / 3
    else:
        total_loss = policy_and_value_loss / 2
    assert not torch.isnan(total_loss).any()

    return total_loss  # , policy_loss, value_loss


class AlphaZeroPolicyWrapperClass(AlphaZeroPolicy):
    def __init__(self, obs_space, action_space, config):
        model = ModelCatalog.get_model_v2(
            obs_space, action_space, action_space.n, config["model"], "torch"
        )
        _, env_creator = Algorithm._get_env_id_and_creator(
            config["env"], config
        )  # env_creatorはcustom envをregister剃るときにわたすやつ．単純に，env classのイニシャライザ．
        if config["ranked_rewards"]["enable"]:
            # if r2 is enabled, tne env is wrapped to include a rewards buffer
            # used to normalize rewards
            env_cls = get_r2_env_wrapper(env_creator, config["ranked_rewards"])

            # the wrapped env is used only in the mcts, not in the
            # rollout workers
            def _env_creator():
                return env_cls(config["env_config"])

        else:

            def _env_creator():
                return env_creator(config["env_config"])

        def mcts_creator():
            return MCTS(model, config["mcts_config"])

        super().__init__(
            obs_space,
            action_space,
            config,
            model,
            alpha_zero_loss,
            TorchCategorical,
            mcts_creator,
            _env_creator,
        )


# @Deprecated(
#     old="rllib/algorithms/alpha_star/",
#     new="rllib_contrib/alpha_star/",
#     help=ALGO_DEPRECATION_WARNING,
#     error=False,
# )
class AlphaZero(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return AlphaZeroConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        return AlphaZeroPolicyWrapperClass

    # @override(Algorithm)
    # def training_step(self) -> ResultDict:
    #     """Default single iteration logic of an algorithm.

    #     - Collect on-policy samples (SampleBatches) in parallel using the
    #       Algorithm's RolloutWorkers (@ray.remote).
    #     - Concatenate collected SampleBatches into one train batch.
    #     - Note that we may have more than one policy in the multi-agent case:
    #       Call the different policies' `learn_on_batch` (simple optimizer) OR
    #       `load_batch_into_buffer` + `learn_on_loaded_batch` (multi-GPU
    #       optimizer) methods to calculate loss and update the model(s).
    #     - Return all collected metrics for the iteration.

    #     Returns:
    #         The results dict from executing the training iteration.
    #     """
    #     """TODO:

    #     Returns:
    #         The results dict from executing the training iteration.
    #     """
    #     # Sample n MultiAgentBatches from n workers.
    #     with self._timers[SAMPLE_TIMER]:
    #         new_sample_batches = synchronous_parallel_sample(
    #             worker_set=self.workers, concat=False
    #         )

    #     for batch in new_sample_batches:
    #         # Update sampling step counters.
    #         self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
    #         self._counters["new_samples"] += batch.env_steps()
    #         self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
    #         # Store new samples in the replay buffer
    #         if self.local_replay_buffer is not None:
    #             self.local_replay_buffer.add(batch)

    #     if self.local_replay_buffer is not None:
    #         # Update target network every `target_network_update_freq` sample steps.
    #         # cur_ts = self._counters[
    #         #     NUM_AGENT_STEPS_SAMPLED
    #         #     if self.config.count_steps_by == "agent_steps"
    #         #     else NUM_ENV_STEPS_SAMPLED
    #         # ]
    #         cur_ts = self._counters["new_samples"]

    #         if cur_ts > self.config.num_steps_sampled_before_learning_starts:
    #             train_batch = self.local_replay_buffer.sample(
    #                 self.config.train_batch_size
    #             )
    #         else:
    #             train_batch = None

    #         # if cur_ts > self.config.replay_buffer_config["capacity"]:
    #         #     train_batch = self.local_replay_buffer.sample(
    #         #         self.config.train_batch_size
    #         #     )
    #         # else:
    #         #     train_batch = None
    #     else:
    #         train_batch = concat_samples(new_sample_batches)

    #     # Learn on the training batch.
    #     # Use simple optimizer (only for multi-agent or tf-eager; all other
    #     # cases should use the multi-GPU optimizer, even if only using 1 GPU)
    #     train_results = {}
    #     if train_batch is not None:
    #         if self.config.get("simple_optimizer") is True:
    #             train_results = train_one_step(self, train_batch)
    #             self._counters["new_samples"] = 0
    #         else:
    #             # with profiler.record_function("MULTI_GPU_TRAIN_ONE_STEP"):
    #             print("training...")
    #             train_results = multi_gpu_train_one_step(self, train_batch)
    #             # カウントをリセットしてreplay bufferを総入れ替えする
    #             self._counters["new_samples"] = 0

    #     # TODO: Move training steps counter update outside of `train_one_step()` method.
    #     # # Update train step counters.
    #     # self._counters[NUM_ENV_STEPS_TRAINED] += train_batch.env_steps()
    #     # self._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

    #     # Update weights and global_vars - after learning on the local worker - on all
    #     # remote workers.
    #     global_vars = {
    #         "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
    #     }
    #     with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
    #         self.workers.sync_weights(global_vars=global_vars)

    #     # Return all collected metrics for the iteration.
    #     return train_results
    @override(Algorithm)
    def training_step(self) -> ResultDict:
        """Default single iteration logic of an algorithm.

        - Collect on-policy samples (SampleBatches) in parallel using the
          Algorithm's RolloutWorkers (@ray.remote).
        - Concatenate collected SampleBatches into one train batch.
        - Note that we may have more than one policy in the multi-agent case:
          Call the different policies' `learn_on_batch` (simple optimizer) OR
          `load_batch_into_buffer` + `learn_on_loaded_batch` (multi-GPU
          optimizer) methods to calculate loss and update the model(s).
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        """TODO:

        Returns:
            The results dict from executing the training iteration.
        """
        # Sample n MultiAgentBatches from n workers.
        # with self._timers[SAMPLE_TIMER]:
        #     new_sample_batches = synchronous_parallel_sample(
        #         worker_set=self.workers, concat=False
        #     )
        if self.config.count_steps_by == "agent_steps":
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size,  concat=False
            )
        else:
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size, concat=False
            )

        for batch in new_sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters["new_samples"] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # Store new samples in the replay buffer
            # if self.local_replay_buffer is not None:
            #     self.local_replay_buffer.add(batch)

        train_batch = concat_samples(new_sample_batches)

        # Learn on the training batch.
        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU)
        train_results = {}
        if train_batch is not None:
            if self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
                self._counters["new_samples"] = 0
            else:
                # with profiler.record_function("MULTI_GPU_TRAIN_ONE_STEP"):
                print("training...")
                train_results = multi_gpu_train_one_step(self, train_batch)
                # カウントをリセットしてreplay bufferを総入れ替えする
                self._counters["new_samples"] = 0

        # TODO: Move training steps counter update outside of `train_one_step()` method.
        # # Update train step counters.
        # self._counters[NUM_ENV_STEPS_TRAINED] += train_batch.env_steps()
        # self._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        if "penalty_metric" in train_batch.policy_batches["default_policy"]:
            penalty_metric = np.sum(
            train_batch.policy_batches["default_policy"]["penalty_metric"]
        ) / np.count_nonzero(
            train_batch.policy_batches["default_policy"]["penalty_metric"]
        )
        

            train_results["default_policy"]["learner_stats"][
            "penalty_metric"
        ] = penalty_metric
            assert not np.isnan(
            penalty_metric
        ), f'{np.sum(train_batch.policy_batches["default_policy"]["penalty_metric"])=}, {np.count_nonzero(train_batch.policy_batches["default_policy"]["penalty_metric"])=}'
            
        if "penalty" in train_batch.policy_batches["default_policy"]:
            penalty = np.sum(
            train_batch.policy_batches["default_policy"]["penalty"]
        ) / np.count_nonzero(
            train_batch.policy_batches["default_policy"]["penalty"]
        )
        

            train_results["default_policy"]["learner_stats"][
            "penalty"
        ] = penalty
            assert not np.isnan(
            penalty
        ), f'{np.sum(train_batch.policy_batches["default_policy"]["penalty"])=}, {np.count_nonzero(train_batch.policy_batches["default_policy"]["penalty"])=}'
            
        if "MTD_metric" in train_batch.policy_batches["default_policy"]:

            MTD_metric = np.sum(
            train_batch.policy_batches["default_policy"]["MTD_metric"]
        ) / np.count_nonzero(train_batch.policy_batches["default_policy"]["MTD_metric"])
            train_results["default_policy"]["learner_stats"]["MTD_metric"] = MTD_metric
        if "DLTs" in train_batch.policy_batches["default_policy"]:
            train_results["default_policy"]["learner_stats"]["DLTs"] = np.nanmean(train_batch.policy_batches["default_policy"]["DLTs"])
            assert not np.isnan(
            MTD_metric
        ), f'{np.sum(train_batch.policy_batches["default_policy"]["MTD_metric"])=}, {np.count_nonzero(train_batch.policy_batches["MTD_policy"]["MTD_metric"])=}'

        return train_results
