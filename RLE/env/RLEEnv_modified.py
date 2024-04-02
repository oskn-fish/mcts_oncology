import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import copy

class RLEEnv_modified(gym.Env):
    def __init__(self, config):
        self.D        = config['D']
        self.N_cohort = config['N_cohort']
        self.N_total  = config['N_total']
        self.phi = config['phi']
        self.early_stop = config['early_stop']
        # self.scenario = config['scenario']
        # self.action_space = spaces.Discrete(3 + self.D)  # down/stay/up + MTD from 1:D
        self.action_space = spaces.Discrete(3+self.D+1)
        self.observation_space = gym.spaces.Dict(
            {   # Tupleを指定するときはTuple((Box,,,))のように，Tupleの引数を本当にtupleにすることに注意．
                "obs": gym.spaces.Box(low=0.0, high=1.0, shape=(1+self.D+self.D+1+1,)), # current dose a
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8),  
            }
        )
        # self.observation_space = spaces.Box(
        #     low = np.hstack([
        #         np.repeat(0.0, 1),       # current dose
        #         np.repeat(0.0, self.D),  # ratio of Ns   to N_total
        #         np.repeat(0.0, self.D),  # ratio of DLTs to N_total
        #         np.repeat(0.0, 1),       # ratio of sum of DLTs to N_total
        #         np.repeat(0.0, 1)        # ratio of sum of Ns to N_total
        #     ]),
        #     high = np.hstack([
        #         np.repeat(1.0, 1),
        #         np.repeat(1.0, self.D),
        #         np.repeat(1.0, self.D),
        #         np.repeat(1.0, 1),
        #         np.repeat(1.0, 1)
        #     ]),
        #     dtype=np.float32
        # )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action in [0,1,2]:  # down/stay/up
            # if action == 0:    # down
                # if self.current_dose == 0:
                #     done = True
                #     MTD = self.D
                #     reward = 1 if self.MTD_true == MTD else 0
                #     return self._get_obs(), reward, done, {'MTD': MTD}
                # else:
                #     draw_dose = self.current_dose - 1
            # elif action == 1:  # stay
            #     draw_dose = self.current_dose
            # elif action == 2:  # up
            #     draw_dose = self.current_dose + 1 #  if self.current_dose <= self.D - 2 else self.current_dose

            """
            no need to care about current dose and up or down
            action_mask takes care of those
            """
            assert sum(self.Ns) < self.N_total
            draw_dose = self.current_dose + (action-1)
            done = False
            MTD = -1
            reward = 0
            draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = self.p_true[draw_dose])
            self.Ns[draw_dose] += self.N_cohort
            self.DLTs[draw_dose] += draw_DLT
            self.current_dose = draw_dose
            # if sum(self.Ns) < self.N_total:
            #     done = False
            #     MTD = -1
            #     reward = 0
            #     draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = self.p_true[draw_dose])
            #     self.Ns[draw_dose] += self.N_cohort
            #     self.DLTs[draw_dose] += draw_DLT
            #     self.current_dose = draw_dose
            # else:
            #     done = True
            #     MTD = -1
            #     reward = 0
        else:   # stop the study and determine the choice as MTD
            done = True
            MTD = action - 3
            reward = 1 if self.MTD_true == MTD else 0
            
        terminated = False
        self.last_obs = self.get_obs()
        return copy.deepcopy((self.last_obs, reward, terminated, done, {'MTD': MTD}))

    def get_obs(self):        
        obs = np.concatenate((
            np.array([self.current_dose / (self.D - 1)]),
            self.Ns / self.N_total,
            self.DLTs / self.N_total,
            np.array([sum(self.DLTs) / self.N_total]),
            np.array([sum(self.Ns) / self.N_total])
        ))   
        
        # if sum(self.Ns)==self.N_total-self.N_cohort:
        if sum(self.Ns)==self.N_total:
            action_dose = [0 for _ in range(3)]
            action_stop = [1 for _ in range(self.D+1)]
        else:
            if self.current_dose == 0:
                action_dose = [0, 1, 1]
            elif self.current_dose == self.D-1:
                action_dose = [1, 1, 0]
            else:
                action_dose = [1 for _ in range(3)]
                
            if self.early_stop:
                action_stop = [1 for _ in range(self.D+1)]
            else:
                action_stop = [0 for _ in range(self.D+1)]

        action_mask = np.array(action_dose+action_stop, dtype=np.int8)

        return copy.deepcopy({"obs":obs, "action_mask":action_mask})
    
    def reset(self, seed=None, options=None):
        
        scenarios = np.array([
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
        ])
        scenario_idx = self.np_random.integers(low = 0, high = 20)
        self.p_true = scenarios[scenario_idx]
        MTD_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]
        self.MTD_true = MTD_true[scenario_idx]
        # self.p_true = self.generate_p_true()
        # self.MTD_true = self.find_MTD(self.p_true)

        self.Ns   = np.zeros(self.D)
        self.DLTs = np.zeros(self.D)
        draw_dose = 0
        draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = self.p_true[draw_dose])
        self.Ns[draw_dose] += self.N_cohort
        self.DLTs[draw_dose] += draw_DLT
        self.current_dose = draw_dose
        # self.N_done_patients = self.N_cohort
        self.last_obs = self.get_obs()
        return copy.deepcopy((self.last_obs, {}))
    
    def generate_p_true(self):
        """
        returns MTD, p_true, M
        """
        selected_dose = np.random.randint(self.D)
        M = np.random.beta(max(self.D-selected_dose, 0.5), 1.0)
        
        B = self.phi + (1-self.phi)*M

        """TODO:
        zhouとすこし違うので，検討する．
        """
        p_true = B*np.random.rand(self.D)
        p_true.sort()

        return p_true #, M
    
    def find_MTD(self, p_true):
        """
        TODO: MTD_trueの定義検証
        """
        p_difference = np.abs(p_true-self.phi)
        MTD = p_difference.argmin()
        if MTD == 0 and p_difference[0]>0.1:
            MTD = self.D
        return MTD
    
    """
    might as well delete below
    """
    
    def get_state(self):
        env_state = {}
        # インスタンスのプロパティを全て取得
        # https://www.yoheim.net/blog.php?q=20161002
        for key, value in self.__dict__.items():
            if callable(value):
                continue
            env_state[key] = value
        env_state = copy.deepcopy(env_state)
        return env_state
    
    def set_state(self, env_state):
        env_state = copy.deepcopy(env_state)
        for key, value in self.__dict__.items():
            if callable(value):
                continue
            setattr(self, key, env_state[key])
        obs = copy.deepcopy(self.last_obs)
        return obs
    # def __init__(self, config):
    #     self.D        = config['D']
    #     self.N_cohort = config['N_cohort']
    #     self.N_total  = config['N_total']
    #     self.phi = config['phi']
    #     self.early_stop = config['early_stop']
    #     self.action_space = spaces.Discrete(3+(self.D+1), start=0)
    #     self.observation_space = gym.spaces.Dict(
    #         {   # Tupleを指定するときはTuple((Box,,,))のように，Tupleの引数を本当にtupleにすることに注意．
    #             "obs": gym.spaces.Box(low=0.0, high=1.0, shape=(1+self.D+self.D+1+1,)), # current dose a
    #             "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8),  
    #         }
    #     )
    #     self.seed()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    # def step(self, action):
    #     assert self.action_space.contains(action)
    #     if action in [0,1,2]:  # down/stay/up
    #         if action == 0:    # down
    #             if self.current_dose == 0:
    #                 done = True
    #                 MTD = self.D
    #                 reward = 1 if self.MTD_true == MTD else 0
    #                 return self._get_obs(), reward, done, {'MTD': MTD}
    #             else:
    #                 draw_dose = self.current_dose - 1
    #         elif action == 1:  # stay
    #             draw_dose = self.current_dose
    #         elif action == 2:  # up
    #             draw_dose = self.current_dose + 1 if self.current_dose <= self.D - 2 else self.current_dose

    #         if sum(self.Ns) < self.N_total:
    #             done = False
    #             MTD = -1
    #             reward = 0
    #             draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = self.p_true[draw_dose])
    #             self.Ns[draw_dose] += self.N_cohort
    #             self.DLTs[draw_dose] += draw_DLT
    #             self.current_dose = draw_dose
    #         else:
    #             done = True
    #             MTD = -1
    #             reward = 0
    #     else:   # stop the study and determine the choice as MTD
    #         done = True
    #         MTD = action - 3
    #         reward = 1 if self.MTD_true == MTD else 0
    #     return self._get_obs(), reward, done, {'MTD': MTD}

    # def get_obs(self):
    #     obs = np.concatenate((
    #         np.array([self.current_dose / (self.D - 1)]),
    #         self.Ns / self.N_total, # self.Nsはnp.arrayだからこのtermもnp.array
    #         self.DLTs / self.N_total,
    #         np.array([sum(self.DLTs) / self.N_total]),
    #         np.array([sum(self.Ns) / self.N_total])
    #     ))
        

    #     if self.N_done_patients == self.N_total: # patientsを使い切っているならactionはMTD推定のみ
    #         # action_dose = [0 for i in range(self.D)]
    #         action_dose = [0 for  _ in range(3)]
    #         action_stop = [1 for _ in range(self.D+1)] # no MTDも含めるので self.D+1
    #     else:
    #         # action_dose = [1 if self.current_dose-1<=i<=self.current_dose+1 else 0 for i in range(self.D)]
    #         if self.current_dose == 0:
    #             action_dose = [0, 1, 1]
    #         elif self.current_dose == self.D-1:
    #             action_dose = [1, 1, 0]
    #         else:
    #             action_dose = [1 for _ in range(3)]
                
    #         if self.early_stop:
    #             action_stop = [1 for _ in range(self.D+1)]
    #         else:
    #             action_stop = [0 for _ in range(self.D+1)]
                
    #     action_mask = np.array(action_dose+action_stop, dtype=np.int8)

    #     return {"obs":obs, "action_mask":action_mask}

    
    # def get_state(self):
    #     env_state = {}
    #     # インスタンスのプロパティを全て取得
    #     # https://www.yoheim.net/blog.php?q=20161002
    #     for key, value in self.__dict__.items():
    #         if callable(value):
    #             continue
    #         env_state[key] = value
    #     env_state = copy.deepcopy(env_state)
    #     return env_state
    
    # def set_state(self, env_state):
    #     env_state = copy.deepcopy(env_state)
    #     for key, value in self.__dict__.items():
    #         if callable(value):
    #             continue
    #         setattr(self, key, env_state[key])
    #     obs = copy.deepcopy(self.last_obs)
    #     return obs
        
    
    

    
    
            
    