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

        self.preprocessor = get_preprocessor(obs_space)(
            obs_space
        )

        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None
        self._values_logit = None

    def forward(self, input_dict, state, seq_lens):
        if type(input_dict) == torch.Tensor: # when input_dict is torch.Tensor
            x = input_dict
        else: # when input_dict is SampleBatch, input_dict["obs"] is obs tensor
            x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._values_logit = self.critic_layers(x)
        # return logits, None
        # assert not torch.isnan(logits).any() or not torch.isnan(self._values_logit)
        return logits, []

    def value_function(self):
        # return nn.Sigmoid()(self._values_logit).view(-1)
        # return nn.Sigmoid()(self._values_logit).view(-1)
        return self._values_logit

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
        
# class ActorCriticModelWithPrediction(ActorCriticModel):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
#         self.prediction_layers = None
    
#     def prediction_logits_function(self):
#         return self._prediction_logits
    
#     def forward(self, input_dict, state, seq_lens):
#         # with profiler.record_function("FORWARD"):
#         x = input_dict["obs"]
#         x = self.shared_layers(x)
#         # actor outputs
#         logits = self.actor_layers(x)

#         # compute value
#         self._value_out = self.critic_layers(x)
#         self._prediction_logits = self.prediction_layers(x)
#         return logits, None
    
#     def compute_prediction(self, obs, action):
        
#         obs = convert_to_tensor([self.preprocessor.transform(obs)])
#         input_dict = restore_original_dimensions(obs, self.obs_space, "torch")
        
#         with torch.no_grad():
#             _ = self.forward(input_dict, None, [1])
#             prediction_logits = self.prediction_logits_function()
#             prediction_logits = torch.squeeze(prediction_logits)
#             prediction = nn.Softmax(dim=1)(prediction_logits)
            
#             prediction = prediction.cpu().numpy()
            
#             return  prediction
    
#     def compute_priors_and_value(self, obs):
#         obs = convert_to_tensor([self.preprocessor.transform(obs)])
#         input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

#         with torch.no_grad():
#             model_out = self.forward(input_dict, None, [1])
#             logits, _ = model_out
#             value = self.value_function()
#             logits, value = torch.squeeze(logits), torch.squeeze(value)
#             priors = nn.Softmax(dim=-1)(logits)
#             value_sig = nn.Sigmoid()(value)

#             priors = priors.cpu().numpy()
#             value_sig = value_sig.cpu().numpy()

#             return priors, value_sig


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
            nn.Linear(
                in_features=obs_space.shape[0], out_features=256
            ),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            # nn.ReLU()
            nn.Tanh()
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n),
        )
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1)
        )
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
