import logging
import numpy as np
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from ray.rllib.models.preprocessors import get_preprocessor


class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation,
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )
        self.preprocessor = get_preprocessor(original_space)(original_space)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # obs = input_dict["obs_flat"].float()
        # obs = input_dict["obs"]
        obs = input_dict if type(input_dict) == torch.Tensor else input_dict["obs"]
        # if type(input_dict) == torch.Tensor:  # when input_dict is torch.Tensor
        #     obs = input_dict
        # else:  # when input_dict is SampleBatch, input_dict["obs"] is obs tensor
        #     obs = input_dict["obs"]
        if type(obs) == tuple:
            obs = obs[0]
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            out = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            # assert not torch.isnan(priors).any() or not torch.isnan(self.value)
            return priors, value

    # def single_value_function(self):
    # return self._value_out
    # torch.reshape(nn.Tanh(nn.self._value_out, [-1])
    # assert not torch.isnan(nn.Sigmoid()(self._values_logit).view(-1)).any()
    # return nn.Sigmoid()(self._values_logit).view(-1)
    # return self.value_function()


from abc import ABC
import numpy as np

from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

# import torch.autograd.profiler as profiler

torch, nn = try_import_torch()


def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor


class ActorCriticModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space)(obs_space)

        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None
        self._values_logit = None

    def forward(self, input_dict, state, seq_lens):
        if type(input_dict) == torch.Tensor:  # when input_dict is torch.Tensor
            x = input_dict
        else:  # when input_dict is SampleBatch, input_dict["obs"] is obs tensor
            x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._values_logit = self.critic_layers(x)
        # return logits, None
        # assert not torch.isnan(logits).any() or not torch.isnan(self._values_logit)
        return logits, []

    # @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            out = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out

    def single_value_function(self):
        # return self._value_out
        # torch.reshape(nn.Tanh(nn.self._value_out, [-1])
        # assert not torch.isnan(nn.Sigmoid()(self._values_logit).view(-1)).any()
        # return nn.Sigmoid()(self._values_logit).view(-1)
        return self._values_logit

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.single_value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            # assert not torch.isnan(priors).any() or not torch.isnan(self.value)
            return priors, value


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvNetModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ActorCriticModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        in_channels = model_config["custom_model_config"]["in_channels"]
        feature_dim = model_config["custom_model_config"]["feature_dim"]

        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            Flatten(),
            nn.Linear(1024, feature_dim),
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=1)
        )

        self._value_out = None


class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ActorCriticModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        """
        最終層の活性化関数がsoftmaxなどであるため，最終層のあとにはReLUを入れない．
        """
        """
        なぜか上のconvではflattenを自作しているが理由はわからない．
        """
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=obs_space.shape[0], out_features=256),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            # nn.ReLU()
            nn.Tanh(),
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n),
        )
        self.critic_layers = nn.Sequential(nn.Linear(in_features=256, out_features=1))
        self._value_out = None


# class DenseModelWithPrediction(ActorCriticModelWithPrediction):
#         def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#             ActorCriticModelWithPrediction.__init__(
#                 self, obs_space, action_space, num_outputs, model_config, name
#             )
#             """
#             最終層の活性化関数がsoftmaxなどであるため，最終層のあとにはReLUを入れない．
#             """
#             """
#             なぜか上のconvではflattenを自作しているが理由はわからない．
#             """
#             self.shared_layers = nn.Sequential(
#                 nn.Linear(
#                     in_features=obs_space.shape[0], out_features=256
#                 ),
#                 # nn.BatchNorm1d(num_features=256),
#                 nn.ReLU(),
#                 nn.Linear(in_features=256, out_features=256),
#                 # nn.BatchNorm1d(num_features=256),
#                 nn.ReLU()
#             )
#             self.actor_layers = nn.Sequential(
#                 nn.Linear(in_features=256, out_features=action_space.n),
#             )
#             self.critic_layers = nn.Sequential(
#                 nn.Linear(in_features=256, out_features=1)
#             )
#             self._value_out = None
#             self.prediction_layers = nn.Sequential(
#                 nn.Linear(in_features=256, out_features=(3+1)*3),
#                 nn.Unflatten(dim=1, unflattened_size=(3,4))
#             )
# def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#     super().__init__(obs_space, action_space, num_outputs, model_config, name)


# class DenseModelWithOneBias(ActorCriticModel):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         ActorCriticModel.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name
#         )

#         self.shared_layers = nn.Sequential(
#             nn.Linear(
#                 in_features=obs_space.shape[0], out_features=256
#             ),
#             nn.ReLU(),
#             nn.Linear(in_features=256, out_features=256),
#             nn.ReLU()
#         )
#         self.actor_layers = nn.Sequential(
#             nn.Linear(in_features=256, out_features=action_space.n),
#             # nn.ReLU()
#         )

#         x = nn.Linear(in_features=256, out_features=1)
#         nn.init.ones_(x.bias)
#         self.critic_layers = nn.Sequential(
#             x
#             # nn.Tanh()
#         )
# self._value_logit = None
