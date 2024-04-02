import gymnasium as gym
from gymnasium.wrappers import transform_observation
import numpy as np
from copy import deepcopy

scenarios = [
                [0.25,0.35,0.465,0.584,0.694,0.786],
                [0.171,0.25,0.35,0.465,0.584,0.694],
                [0.113,0.171,0.25,0.35,0.465,0.584],
                [0.073,0.113,0.171,0.25,0.35,0.465],
                [0.05,0.073,0.113,0.171,0.25,0.35],
                [0.05,0.05,0.073,0.113,0.171,0.25],
                [0.35,0.465,0.584,0.694,0.786,0.8],
                [0.21,0.35,0.522,0.688,0.8,0.8],
                [0.116,0.21,0.35,0.522,0.688,0.8],
                [0.061,0.116,0.21,0.35,0.522,0.688],
                [0.05,0.061,0.116,0.21,0.35,0.522],
                [0.05,0.05,0.061,0.116,0.21,0.35],
                [0.05,0.05,0.05,0.061,0.116,0.21],
                [0.35,0.522,0.688,0.8,0.8,0.8],
                [0.29,0.486,0.686,0.8,0.8,0.8],
                [0.15,0.29,0.486,0.686,0.8,0.8],
                [0.071,0.15,0.29,0.486,0.686,0.8],
                [0.05,0.071,0.15,0.29,0.486,0.686],
                [0.05,0.05,0.071,0.15,0.29,0.486],
                [0.05,0.05,0.05,0.071,0.15,0.29]
            ]

target_DLT_prob = .25

# https://www.gymlibrary.dev/api/wrappers/
class RLEEnv4Alphazero(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        """
        observation:Tuple(Box(残り被験者割合，1cohort割合), Sequence)
        """
        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Tuple(gym.spaces.Box(low=np.array(0.0, 0.0), high=np.array([1.0, 1.0])), gym.spaces.Sequence(gym.spaces.Discrete(self.env.N_cohort))),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8)
            }
        )
        self.last_obs = None
    
    # def observation(self, obs):
    #     return {"obs": super().observation(obs), "action_mask": np.array([1]*self.action_space.n, dtype=np.int8)}
    
    # def _get_obs(self):
    #     return {"obs":self.env._get_obs(), "action_mask":np.array([1]*self.action_space.n, dtype=np.int8)}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        self.last_obs = obs.astype(np.float32)
        return {"obs": obs, "action_mask": np.array([1]*self.action_space.n, dtype=np.int8)}, reward, done, truncated, info
    
    def _get_obs(self):
        """
        action_maskを入れておくので，前回投与されたdose記憶はobservationに入れない．
        obs = tuple(Box(残った被験者の割合，1cohortの割合)，history)
        """
        return np.

    def reset(self, seed=None, options=None):
        """
        # REEnvでp_trueを設定するとき
        # TODO: 元のREEnvのobservationを変える
        # observation:Tuple(Box(残り被験者割合，1cohort割合), Sequence)
        """
        
        # scenarioの設定
        self.env.p_true = options["p_true"] # should be np.array
        self.env.MTD_true, _ = max([(dose, DLT) for dose, DLT in enumerate(self.env.p_true) if DLT<target_DLT_prob]) # NOTE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2684552/ 参照に，アメリカ式を採用．
            
        # 初期obervationを生成
        
            
        return {"obs": obs, "action_mask": np.array([1]*self.action_space.n, dtype=np.int8)}, {"scenario":scenarios.index(self.p_true.tolist()), "p_true":self.p_true, "MTD_true":self.MTD_true}
    
    def set_state(self, state):
        self.env = deepcopy(state[0])
        obs = deepcopy(state[1])
        return {"obs": obs, "action_mask": np.array([1]*self.action_space.n, dtype=np.int8)}
    
    def get_state(self):
        return deepcopy(self.env), deepcopy(self.last_obs)

# rleenv = gym.make("RLEEnv", config=config)