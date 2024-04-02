import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class RLEEnv(gym.Env):
    def __init__(self, config):
        self.D        = config['D']
        self.N_cohort = config['N_cohort']
        self.N_total  = config['N_total']
        self.scenario = config['scenario']
        self.action_space = spaces.Discrete(3 + self.D)  # down/stay/up + MTD from 1:D
        self.observation_space = spaces.Box(
            low = np.hstack([ # NOTE: 水平方向にndarrayを結合
                np.repeat(0.0, 1),       # current dose
                np.repeat(0.0, self.D),  # ratio of Ns   to N_total
                np.repeat(0.0, self.D),  # ratio of DLTs to N_total
                np.repeat(0.0, 1),       # ratio of sum of DLTs to N_total
                np.repeat(0.0, 1)        # ratio of sum of Ns to N_total
            ]),
            high = np.hstack([
                np.repeat(1.0, 1),
                np.repeat(1.0, self.D),
                np.repeat(1.0, self.D),
                np.repeat(1.0, 1),
                np.repeat(1.0, 1)
            ]),
            dtype=np.float32
        )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action in [0,1,2]:  # down/stay/up
            if action == 0:    # down
                if self.current_dose == 0:
                    done = True
                    MTD = self.D
                    reward = 1 if self.MTD_true == MTD else 0
                    return self._get_obs(), reward, done, {'MTD': MTD}
                else:
                    draw_dose = self.current_dose - 1
            elif action == 1:  # stay
                draw_dose = self.current_dose
            elif action == 2:  # up
                draw_dose = self.current_dose + 1 if self.current_dose <= self.D - 2 else self.current_dose

            if sum(self.Ns) < self.N_total:
                done = False
                MTD = -1
                reward = 0
                draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = self.p_true[draw_dose])
                self.Ns[draw_dose] += self.N_cohort
                self.DLTs[draw_dose] += draw_DLT
                self.current_dose = draw_dose
            else:
                done = True
                MTD = -1
                reward = 0
        else:   # stop the study and determine the choice as MTD
            done = True
            MTD = action - 3
            reward = 1 if self.MTD_true == MTD else 0
        return self._get_obs(), reward, done, {'MTD': MTD}

    def _get_obs(self):
        return np.concatenate((
            np.array([self.current_dose / (self.D - 1)]),
            self.Ns / self.N_total, # self.Nsはnp.arrayだからこのtermもnp.array
            self.DLTs / self.N_total,
            np.array([sum(self.DLTs) / self.N_total]),
            np.array([sum(self.Ns) / self.N_total])
        ))

    def reset(self):
        if self.scenario == 'random':
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
        elif self.scenario == '1':
            self.p_true = np.array([0.26, 0.34, 0.47, 0.64, 0.66, 0.77])
            self.MTD_true = 0
        elif self.scenario == '2':
            self.p_true = np.array([0.18, 0.25, 0.32, 0.36, 0.60, 0.69])
            self.MTD_true = 1
        elif self.scenario == '3':
            self.p_true = np.array([0.09, 0.16, 0.23, 0.34, 0.51, 0.74])
            self.MTD_true = 2
        elif self.scenario == '4':
            self.p_true = np.array([0.07, 0.12, 0.17, 0.27, 0.34, 0.55])
            self.MTD_true = 3
        elif self.scenario == '5':
            self.p_true = np.array([0.03, 0.13, 0.17, 0.19, 0.26, 0.31])
            self.MTD_true = 4
        elif self.scenario == '6':
            self.p_true = np.array([0.04, 0.05, 0.09, 0.14, 0.15, 0.24])
            self.MTD_true = 5
        elif self.scenario == '7':
            self.p_true = np.array([0.34, 0.42, 0.46, 0.49, 0.58, 0.62])
            self.MTD_true = 6
        elif self.scenario == '8':
            self.p_true = np.array([0.13, 0.41, 0.45, 0.58, 0.75, 0.76])
            self.MTD_true = 0
        elif self.scenario == '9':
            self.p_true = np.array([0.05, 0.08, 0.11, 0.15, 0.60, 0.72])
            self.MTD_true = 3
        elif self.scenario == '10':
            self.p_true = np.array([0.15, 0.17, 0.19, 0.21, 0.23, 0.25])
            self.MTD_true = 5

        self.Ns   = np.zeros(self.D)
        self.DLTs = np.zeros(self.D)
        draw_dose = 0
        draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = self.p_true[draw_dose])
        self.Ns[draw_dose] += self.N_cohort
        self.DLTs[draw_dose] += draw_DLT
        self.current_dose = draw_dose
        return self._get_obs()
