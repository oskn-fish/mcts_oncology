# automatically register our environment when imported
# registration not needed for not using gym explicitly
# custom environment will be used in rllib implicitely

from RLE.env.RLEEnv import RLEEnv
from RLE.env.Bayes_RLEEnv import Bayes_RLEEnv
from RLE.env.RLEEnv_expectation import RLEEnv_expectation
# from gymnasium.envs.registration import register

# register(
#     id = "RLEEnv",
#     entry_point="env.RLEEnv:RLEEnv",
# )