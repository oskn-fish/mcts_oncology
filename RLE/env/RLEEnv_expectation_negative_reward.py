import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
import math


class RLEEnv_expectation_negative_reward(gym.Env):
    def __init__(self, config):
        self.D = config["D"]
        self.N_cohort = config["N_cohort"]
        self.N_total = config["N_total"]
        # self.phi = config['phi']
        # self.early_stop = config['early_stop']
        # self.scenario = config['scenario']
        # self.action_space = spaces.Discrete(3 + self.D)  # down/stay/up + MTD from 1:D
        self.action_space = spaces.Discrete(3 + self.D + 1)
        # self.observation_space = gym.spaces.Dict(
        #     {   # Tupleを指定するときはTuple((Box,,,))のように，Tupleの引数を本当にtupleにすることに注意．
        #         "obs": gym.spaces.Box(low=0.0, high=1.0, shape=(1+self.D+self.D+1+1,)), # current dose a
        #         "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8),
        #     }
        # )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1 + self.D + self.D + 1 + 1,)
        )
        self.scenarios = np.array(
            [
                [0.25, 0.35, 0.465, 0.584, 0.694, 0.786],
                [0.171, 0.25, 0.35, 0.465, 0.584, 0.694],
                [0.113, 0.171, 0.25, 0.35, 0.465, 0.584],
                [0.073, 0.113, 0.171, 0.25, 0.35, 0.465],
                [0.05, 0.073, 0.113, 0.171, 0.25, 0.35],
                [0.05, 0.05, 0.073, 0.113, 0.171, 0.25],
                [0.35, 0.465, 0.584, 0.694, 0.786, 0.8],
                [0.21, 0.35, 0.522, 0.688, 0.8, 0.8],
                [0.116, 0.21, 0.35, 0.522, 0.688, 0.8],
                [0.061, 0.116, 0.21, 0.35, 0.522, 0.688],
                [0.05, 0.061, 0.116, 0.21, 0.35, 0.522],
                [0.05, 0.05, 0.061, 0.116, 0.21, 0.35],
                [0.05, 0.05, 0.05, 0.061, 0.116, 0.21],
                [0.35, 0.522, 0.688, 0.8, 0.8, 0.8],
                [0.29, 0.486, 0.686, 0.8, 0.8, 0.8],
                [0.15, 0.29, 0.486, 0.686, 0.8, 0.8],
                [0.071, 0.15, 0.29, 0.486, 0.686, 0.8],
                [0.05, 0.071, 0.15, 0.29, 0.486, 0.686],
                [0.05, 0.05, 0.071, 0.15, 0.29, 0.486],
                [0.05, 0.05, 0.05, 0.071, 0.15, 0.29],
            ]
        )
        self.MTD_trues = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]
        self.posteriors = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        """
        cal_rewardはlast_obsを用いるので，obsは先に計算しておくこと，
        """
        if action in [0, 1, 2]:  # down/stay/up
            if action == 0:  # down
                if self.current_dose == 0:
                    done = True
                    MTD = self.D
                    # reward = 1 if self.MTD_true == MTD else 0
                    # self.last_obs = self.get_obs()
                    if type(self.posteriors) == np.ndarray:
                        assert self.MTD_true == None and self.p_true == None
                        reward = self.cal_reward(
                            self.last_obs, MTD
                        )  # 投与せず，last_obsの交信もない
                    else:
                        # reward = 1 if self.MTD_true == MTD else 0
                        reward = 1 if self.MTD_true == MTD else -1
                    truncated = False
                    return copy.deepcopy(
                        (self.last_obs, reward, done, truncated, {"MTD": MTD})
                    )
                else:
                    draw_dose = self.current_dose - 1
            elif action == 1:  # stay
                draw_dose = self.current_dose
            elif action == 2:  # up
                draw_dose = (
                    self.current_dose + 1
                    if self.current_dose <= self.D - 2
                    else self.current_dose
                )

            """
            ↑の設定（upでもcurrent doseがstay）は変なので，変更検討
            """

            """
            no need to care about current dose and up or down
            action_mask takes care of those
            """

            if type(self.posteriors) == np.ndarray:
                index = np.random.choice(
                    np.arange(len(self.scenarios)), 1, p=self.posteriors
                )[0]
                p_true = self.scenarios[index]
            else:
                p_true = self.p_true

            # assert sum(self.Ns) < self.N_total
            # done = False
            # MTD = -1
            # reward = 0
            # draw_DLT  = self.np_random.binomial(n = self.N_cohort, p = p_true[draw_dose])
            # self.Ns[draw_dose] += self.N_cohort
            # self.DLTs[draw_dose] += draw_DLT
            # self.current_dose = draw_dose
            if sum(self.Ns) < self.N_total:
                done = False
                MTD = -1
                reward = 0
                draw_DLT = self.np_random.binomial(n=self.N_cohort, p=p_true[draw_dose])
                self.Ns[draw_dose] += self.N_cohort
                self.DLTs[draw_dose] += draw_DLT
                self.current_dose = draw_dose
            else:
                # print("unreasonable action: action\in \{0,1,2\} when final timestep")
                done = True
                MTD = -1
                # reward = 0
                reward = -1
        else:  # stop the study and determine the choice as MTD
            done = True
            MTD = action - 3
            self.last_obs = self.get_obs()
            # reward = self.cal_reward(self.last_obs, MTD) # 投薬実験を行わないので，obsは交信なし
            if type(self.posteriors) == np.ndarray:
                assert self.MTD_true == None and self.p_true == None
                reward = self.cal_reward(self.last_obs, MTD)  # 投与せず，last_obsの交信もない
            else:
                # reward = 1 if self.MTD_true == MTD else 0
                reward = 1 if self.MTD_true == MTD else -1
            truncated = False
            return copy.deepcopy((self.last_obs, reward, done, truncated, {"MTD": MTD}))

        truncated = False
        self.last_obs = self.get_obs()
        return copy.deepcopy((self.last_obs, reward, done, truncated, {"MTD": MTD}))

    def cal_reward(self, obs, MTD):
        MTD_posterior = self.cal_MTD_posterior(obs)
        reward = 1 * MTD_posterior[MTD] + (-1) * (1 - MTD_posterior[MTD])
        return reward

    def cal_MTD_posterior(self, obs):
        MTD_posterior = [0 for _ in range(self.D + 1)]
        posterior = self.cal_posterior(obs)
        for index in range(len(self.scenarios)):
            MTD_posterior[self.MTD_trues[index]] += posterior[index]
        return MTD_posterior

    def get_obs(self):
        obs = np.concatenate(
            (
                np.array([self.current_dose / (self.D - 1)]),
                self.Ns / self.N_total,
                self.DLTs / self.N_total,
                np.array([sum(self.DLTs) / self.N_total]),
                np.array([sum(self.Ns) / self.N_total]),
            ),
            dtype=np.float32,
        )

        # if sum(self.Ns)==self.N_total:
        #     action_dose = [0 for _ in range(3)]
        #     action_stop = [1 for _ in range(self.D+1)]
        # else:
        #     if self.current_dose == 0:
        #         action_dose = [0, 1, 1]
        #     elif self.current_dose == self.D-1:
        #         action_dose = [1, 1, 0]
        #     else:
        #         action_dose = [1 for _ in range(3)]

        #     if self.early_stop:
        #         action_stop = [1 for _ in range(self.D+1)]
        #     else:
        #         action_stop = [0 for _ in range(self.D+1)]

        # action_mask = np.array(action_dose+action_stop, dtype=np.int8)

        # return copy.deepcopy({"obs":obs, "action_mask":action_mask})
        return copy.deepcopy(obs)

    def reset(self, seed=None, options=None):
        scenario_idx = self.np_random.integers(low=0, high=20)
        self.p_true = self.scenarios[scenario_idx]
        # MTD_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]
        self.MTD_true = self.MTD_trues[scenario_idx]
        # self.p_true = self.generate_p_true()
        # self.MTD_true = self.find_MTD(self.p_true)

        self.Ns = np.zeros(self.D)
        self.DLTs = np.zeros(self.D)
        draw_dose = 0
        draw_DLT = self.np_random.binomial(n=self.N_cohort, p=self.p_true[draw_dose])
        self.Ns[draw_dose] += self.N_cohort
        self.DLTs[draw_dose] += draw_DLT
        self.current_dose = draw_dose
        # self.N_done_patients = self.N_cohort
        self.last_obs = self.get_obs()
        return copy.deepcopy((self.last_obs)), {}

    def cal_posterior(self, obs):
        """
        returns posterior: scenario_index -> posterior_prob
        """
        """
        np.array([self.current_dose / (self.D - 1)]),
        self.Ns / self.N_total, # self.Nsはnp.arrayだからこのtermもnp.array
        self.DLTs / self.N_total,
        np.array([sum(self.DLTs) / self.N_total]),
        np.array([sum(self.Ns) / self.N_total])
        """
        # observation = obs["obs"]
        Ns = (obs[1 : 1 + self.D] * self.N_total).astype(np.int8)
        DLTs = (obs[1 + self.D : 1 + self.D * 2] * self.N_total).astype(np.int8)
        scenarios = self.scenarios

        num_scenarios = len(self.scenarios)
        priors = [1.0 / num_scenarios for _ in range(num_scenarios)]
        likelihoods = [
            self.cal_likelihood(Ns=Ns, DLTs=DLTs, scenario=scenario)
            for scenario in scenarios
        ]
        posteriors = np.array(
            [likelihood * prior for likelihood, prior in zip(likelihoods, priors)]
        )
        posteriors = posteriors / sum(posteriors)
        return posteriors

    def cal_likelihood(self, Ns, DLTs, scenario):
        likelihood = 1.0
        for N, DLT, p in zip(Ns, DLTs, scenario):
            likelihood *= p**DLT * (1 - p) ** (N - DLT) * math.comb(N, DLT)
        return likelihood

    def get_state(self):
        # return self.cal_MTD_posterior(self.last_obs)
        return copy.deepcopy(self.last_obs)

    def set_state(self, last_obs):
        self.posteriors = self.cal_posterior(last_obs)
        self.current_dose = round(last_obs[0] * (self.D - 1))
        self.Ns = (last_obs[1 : self.D + 1] * self.N_total).astype(np.int8)
        self.DLTs = (
            last_obs[self.D + 1 : (self.D + 1) + self.D] * self.N_total
        ).astype(np.int8)
        self.last_obs = last_obs
        self.p_true = None
        self.MTD_true = None
        return copy.deepcopy(last_obs)
