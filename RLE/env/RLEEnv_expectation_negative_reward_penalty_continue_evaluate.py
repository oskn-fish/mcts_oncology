import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
import math


class RLEEnv_expectation_negative_reward_penalty_continue_evaluate(gym.Env):
    def __init__(self, config):
        self.D = config["D"]
        self.N_cohort = config["N_cohort"]
        self.N_total = config["N_total"]
        self.DLT_max = config["DLT_max"]
        self.penalty_abs = config["penalty_abs"]
        # # self.phi = config['phi']
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
                [0.26, 0.34, 0.47, 0.64, 0.66, 0.77],
                [0.18, 0.25, 0.32, 0.36, 0.60, 0.69],
                [0.09, 0.16, 0.23, 0.34, 0.51, 0.74],
                [0.07, 0.12, 0.17, 0.27, 0.34, 0.55],
                [0.03, 0.13, 0.17, 0.19, 0.26, 0.31],
                [0.04, 0.05, 0.09, 0.14, 0.15, 0.24],
                [0.34, 0.42, 0.46, 0.49, 0.58, 0.62],
                [0.13, 0.41, 0.45, 0.58, 0.75, 0.76],
                [0.05, 0.08, 0.11, 0.15, 0.60, 0.72],
                [0.15, 0.17, 0.19, 0.21, 0.23, 0.25]
            ]
        )
        self.MTD_trues = [0, 1, 2, 3, 4, 5, 6, 0, 3, 5]
        self.scenario = config["scenario"]
        self.p_true = self.scenarios[config["scenario"]]
        self.MTD_true = self.MTD_trues[config["scenario"]]
        # self.posteriors = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # compute penalty
        # penalty = (
        #     -self.penalty_abs
        #     if round(final_obs[-2] * self.N_total) >= self.DLT_max
        #     else 0
        # )
        # self.DLT_max
        # self.penalty_prob
        # self.penalty_abs

        # penaltyは，まだわからんときは0, なくて終了したら-1， あって終了したら1にする

        """
        reward == if_MTD_correct + penalty
        """

        """
        cal_rewardはlast_obsを用いるので，obsは先に計算しておくこと，
        """
        if action in [0, 1, 2]:  # down/stay/up
            if action == 0:  # down
                if self.current_dose == 0:  # stop experiment
                    done = True
                    MTD = self.D
                    # reward = 1 if self.MTD_true == MTD else 0
                    self.last_obs = self.get_obs()
                    # if type(self.posteriors) == np.ndarray:
                    #     # posterior更新
                    #     self.posterior = self.cal_posterior(self.last_obs) 
                    #     assert self.MTD_true == None and self.p_true == None
                    #     MTD_metric = self.cal_reward(
                    #         self.last_obs, MTD
                    #     )  # 投与せず，last_obsの交信もない

                    # else:
                        # reward = 1 if self.MTD_true == MTD else 0
                    MTD_metric = 1 if self.MTD_true == MTD else -1
                    truncated = False

                    penalty_metric = (
                        1
                        if round(self.last_obs[-2] * self.N_total) >= self.DLT_max
                        else -1
                    )
                    penalty = -max(penalty_metric, 0) * self.penalty_abs
                    reward = penalty + MTD_metric

                    return copy.deepcopy(
                        (
                            self.last_obs,
                            reward,
                            done,
                            truncated,
                            {
                                "MTD": MTD,
                                "penalty_metric": penalty_metric,
                                "MTD_metric": MTD_metric,
                                "DLTs": round(self.last_obs[-2] * self.N_total)
                            },
                        )
                    )
                    # return copy.deepcopy(
                    #     (
                    #         self.last_obs,
                    #         reward,
                    #         done,
                    #         truncated,
                    #         {"MTD": MTD, "penalty_metric": "first"},
                    #     )
                    # )
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

            # if type(self.posteriors) == np.ndarray:
            #     self.posterior = self.cal_posterior(self.last_obs) 
            #     index = np.random.choice(
            #         np.arange(len(self.scenarios)), 1, p=self.posteriors
            #     )[0]
            #     p_true = self.scenarios[index]
            # else:
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

                penalty_metric = 0
                MTD_metric = 0
                self.last_obs = self.get_obs()
                # if type(self.posteriors) == np.ndarray:
                #     self.posterior = self.cal_posterior(self.last_obs) 
                
            else:
                # print("unreasonable action: action\in \{0,1,2\} when final timestep")
                done = True
                MTD = -1
                # reward = -1
                # penalty = (
                #     -self.penalty_abs
                #     if round(self.last_obs[-2] * self.N_total) >= self.DLT_max
                #     else 0
                # )
                # reward += penalty
                # penalty_metric = -1 if penalty == 0 else 1
                MTD_metric = -1
                self.last_obs = self.get_obs()
                penalty_metric = (
                    1 if round(self.last_obs[-2] * self.N_total) >= self.DLT_max else -1
                )
                penalty = -max(penalty_metric, 0) * self.penalty_abs
                reward = penalty + MTD_metric

        else:  # stop the study and determine the choice as MTD
            done = True
            MTD = action - 3
            self.last_obs = self.get_obs()
            # reward = self.cal_reward(self.last_obs, MTD) # 投薬実験を行わないので，obsは交信なし
            # if type(self.posteriors) == np.ndarray:
            #     self.posterior = self.cal_posterior(self.last_obs) 
            #     assert self.MTD_true == None and self.p_true == None
            #     MTD_metric = self.cal_reward(self.last_obs, MTD)  # 投与せず，last_obsの交信もない
            # else:

            MTD_metric = 1 if self.MTD_true == MTD else -1
            truncated = False

            # impose penalty
            # penalty = (
            #     -self.penalty_abs
            #     if round(self.last_obs[-2] * self.N_total) >= self.DLT_max
            #     else 0
            # )
            # reward += penalty
            # penalty_metric = -1 if penalty == 0 else 1
            penalty_metric = (
                1 if round(self.last_obs[-2] * self.N_total) >= self.DLT_max else -1
            )
            penalty = -max(penalty_metric, 0) * self.penalty_abs
            reward = penalty + MTD_metric

            return copy.deepcopy(
                (
                    self.last_obs,
                    reward,
                    done,
                    truncated,
                    {
                        "MTD": MTD,
                        "penalty_metric": penalty_metric,
                        "MTD_metric": MTD_metric,
                        "DLTs": round(self.last_obs[-2] * self.N_total)
                    },
                )
            )
            # return copy.deepcopy(
            #     (
            #         self.last_obs,
            #         reward,
            #         done,
            #         truncated,
            #         {"MTD": MTD, "penalty_metric": "second"},
            #     )
            # )

        # 終了していない or 人数超過で終了
        truncated = False
        self.last_obs = self.get_obs()
        return copy.deepcopy(
            (
                self.last_obs,
                reward,
                done,
                truncated,
                {
                    "MTD": MTD,
                    "penalty_metric": penalty_metric,
                    "MTD_metric": MTD_metric,
                    "DLTs": np.nan if not done else round(self.last_obs[-2] * self.N_total)
                },
            )
        )
        # return copy.deepcopy(
        #     (
        #         self.last_obs,
        #         reward,
        #         done,
        #         truncated,
        #         {"MTD": MTD, "penalty_metric": "third"},
        #     )
        # )

    # def cal_reward(self, obs, MTD):
    #     MTD_posterior = self.cal_MTD_posterior(obs)
    #     reward = 1 * MTD_posterior[MTD] + (-1) * (1 - MTD_posterior[MTD])
    #     return reward

    # def cal_MTD_posterior(self, obs):
    #     MTD_posterior = [0 for _ in range(self.D + 1)]
    #     posterior = self.cal_posterior(obs)
    #     for index in range(len(self.scenarios)):
    #         MTD_posterior[self.MTD_trues[index]] += posterior[index]
    #     return MTD_posterior

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
        return copy.deepcopy(obs)

    def reset(self, seed=None, options=None):
        # scenario_idx = self.np_random.integers(low=0, high=20)
        # self.p_true = self.scenarios[scenario_idx]
        # MTD_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]
        # self.MTD_true = self.MTD_trues[scenario_idx]
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

        # self.penalty_metric = 0

        return copy.deepcopy((self.last_obs)), {"penalty_metric": 0, "MTD_metric": 0}

    # def cal_posterior(self, obs):
    #     """
    #     returns posterior: scenario_index -> posterior_prob
    #     """
    #     """
    #     np.array([self.current_dose / (self.D - 1)]),
    #     self.Ns / self.N_total, # self.Nsはnp.arrayだからこのtermもnp.array
    #     self.DLTs / self.N_total,
    #     np.array([sum(self.DLTs) / self.N_total]),
    #     np.array([sum(self.Ns) / self.N_total])
    #     """
    #     # observation = obs["obs"]
    #     Ns = (obs[1 : 1 + self.D] * self.N_total).astype(np.int8)
    #     DLTs = (obs[1 + self.D : 1 + self.D * 2] * self.N_total).astype(np.int8)
    #     scenarios = self.scenarios

    #     num_scenarios = len(self.scenarios)
    #     priors = [1.0 / num_scenarios for _ in range(num_scenarios)]
    #     likelihoods = [
    #         self.cal_likelihood(Ns=Ns, DLTs=DLTs, scenario=scenario)
    #         for scenario in scenarios
    #     ]
    #     posteriors = np.array(
    #         [likelihood * prior for likelihood, prior in zip(likelihoods, priors)]
    #     )
    #     posteriors = posteriors / sum(posteriors)
    #     return posteriors

    # def cal_likelihood(self, Ns, DLTs, scenario):
    #     likelihood = 1.0
    #     for N, DLT, p in zip(Ns, DLTs, scenario):
    #         likelihood *= p**DLT * (1 - p) ** (N - DLT) * math.comb(N, DLT)
    #     return likelihood

    def get_state(self):
        return copy.deepcopy(self.last_obs)


    # def get_last_obs(self):
    #     return copy.deepcopy(self.last_obs)

    # def set_state(self, last_obs):
    #     self.posteriors = self.cal_posterior(last_obs)
    #     self.current_dose = round(last_obs[0] * (self.D - 1))
    #     self.Ns = (last_obs[1 : self.D + 1] * self.N_total).astype(np.int8)
    #     self.DLTs = (
    #         last_obs[self.D + 1 : (self.D + 1) + self.D] * self.N_total
    #     ).astype(np.int8)
    #     self.last_obs = last_obs
    #     self.p_true = None
    #     self.MTD_true = None
    #     return copy.deepcopy(last_obs)
