import pytest
from context import Bayes_RLEEnv
import numpy as np
import random

# config = {"D": 6, "N_cohort": 3, "N_total": 36}
# env = gym.make("Bayes_RLE", config = config)

# obs, info = env.reset()

# for _ in range(5):
#     while not terminated:
#         action = random.randint(0, config["D"]*2-1)
#         observation, reward, terminated, truncated, info = env.step(action)
#     obs, info = env.reset()

class test_Bayes_RLEEnv:
    def __init__(self):
        self.config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
        self.env = Bayes_RLEEnv(self.config)
        self.env.reset()
        
    def test_reset(self):
        dictionary, info = self.env.reset()
        assert type(dictionary) == dict and type(info) == dict
        p_true = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.env.reset(options={"p_true":p_true})
        assert type(dictionary) == dict and type(info) == dict
        
    def test_get_state(self):
        last_obs, history = self.env.get_state()
        assert type(last_obs) == tuple and type(history) == tuple
    
    def test_set_state(self):
        env_state = ({"obs":(1.0/5.0, self.config["N_cohort"]/self.config["N_total"])}, ([0,0,1], [0,1,2]))
        self.env.set_state(env_state)
    
    def step(self):
        pass
    
if __name__ == "__main__":
    """ 
    self.D        = config["D"]
    self.N_cohort = config["N_cohort"]
    self.N_total  = config["N_total"]
    self.phi = config["phi"]
    """
    env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25}
    # action_space = [i for i in range(env_config["D"]*2+1)]
    env = Bayes_RLEEnv(env_config)
    
    states = []
    
    obs, info = env.reset()
    valid_actions = obs["action_mask"].astype(np.bool_)
    action_space = valid_actions.nonzero()[0]
    terminated = False
    while not terminated:
        action = np.random.choice(action_space)
        obs, reward, terminated, truncated, info = env.step(action)

        states.append(env.get_state())
    
    state = random.choice(states)
    obs = env.set_state(state)
    terminated = False
    while not terminated:
        action = random.choice(action_space)
        obs, reward, terminated, truncated, info = env.step(action)