"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import numpy as np

# from alpha_zero_bayes.stan_model import stan_model
# import stan

import copy

import alpha_zero_bayes.constants as constants

# from functools import cache
import contextlib
import os

# from logging import getLogger, ERROR
# logger_stan = getLogger("pystan")
# logger_stan.propagate = False
# logger_stan.setLevel(ERROR)
# logger_cmdstan = getLogger("cmdstanpy")
# logger_cmdstan.disabled = True

import pickle
import os
import math
# from functools import cache

# import debugpy
# debugpy.debug_this_thread()

# TODO: action_maskを利用して，actionをup stay donwじゃなくて普通にdose指定にする

RANDOM_SEED = 1
NUM_CHAINS = 4
NUM_SAMPLES = 1000

from logging import getLogger, DEBUG, ERROR, StreamHandler

logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(ERROR)
logger.addHandler(handler)
logger.propagate = False


class StateNode:
    # def __init__(self, action, obs, done, reward, state, mcts, parent=None):
    def __init__(self, obs, done, mcts, reward, parent=None):
        self.env = parent.env
        # self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space.n
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # N
        # self.valid_actions = obs["action_mask"].astype(np.bool_)

        # self.reward = reward
        self.done = done
        # self.state = state
        self.obs = obs

        self.mcts = mcts
        self.reward = reward

    @property
    def number_visits(self):
        return self.parent.child_number_visits[tuple(self.obs)]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[tuple(self.obs)] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[tuple(self.obs)]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[tuple(self.obs)] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)  #

    def child_U(self):
        return (
            math.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )
        """
        ここ変更
        """
        # return np.sqrt((np.log(self.number_visits)/(1+self.child_number_visits)))

    def best_action(self):  # TODO: ActionNode, StateNode
        child_score = (
            self.child_Q() + self.mcts.c_puct * self.child_U()
        )  # self.mcts.c_puct == mcts_param["puct_coefficient"]
        # masked_child_score = child_score
        # masked_child_score[~self.valid_actions] = -np.inf
        # return np.argmax(masked_child_score)
        return np.argmax(child_score)

    def select(self):  # TODO: add bayes sampling
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_grand_child(
                best_action
            )  # TODO: get_childでActionStateを隠蔽しても良い．
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_grand_child(self, action):
        """
        returns self's child generated from action.
        when child is absent, generates it.

        Node.children == {1: None, 2:Node, ...}

        """

        if action not in self.children:
            self.children[action] = ActionNode(parent=self, action=action)

        self.env.set_state(self.obs)
        obs, reward, done, truncated, info = self.env.step(action)
        tupled_obs = tuple(obs)

        if not tupled_obs in self.children[action].children:
            self.children[action].children[tupled_obs] = StateNode(
                obs=obs,
                done=done,
                mcts=self.mcts,
                parent=self.children[action],
                reward=reward,
            )

        return self.children[action].children[tupled_obs]

    # def restore_int_obs(self, obs):
    #     action_mask = obs["action_mask"]
    #     observation = (obs["obs"]*self.env.N_total).astype(int)

    #     current_dose = round(observation[0])
    #     Ns = observation[1:self.env.D+1]
    #     DLTs = observation[self.env.D+1:(self.env.D+1)+self.env.D]
    #     sum_Ns = sum(Ns)
    #     sum_DLTs = sum(DLTs)

    #     return action_mask, current_dose, Ns, DLTs, sum_Ns, sum_DLTs

    # def revert_to_obs(self, action_mask, current_dose, Ns, DLTs, sum_Ns, sum_DLTs):
    #     obs = np.concatenate((
    #         np.array([current_dose / (self.env.D - 1)]),
    #         Ns / self.env.N_total,
    #         DLTs / self.env.N_total,
    #         np.array([sum_DLTs / self.env.N_total]),
    #         np.array([sum_Ns / self.env.N_total])
    #     ))

    #     return {"obs":obs, "action_mask": action_mask}

    # def obs_to_tuple(self, obs):
    #     action_mask_tuple = tuple(obs["action_mask"])
    #     observation_tuple = tuple(obs["obs"])
    #     return action_mask_tuple + observation_tuple

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent


# class ActionNode(Node):
class ActionNode:
    def __init__(self, parent, action):
        self.children = {}
        self.parent = parent
        self.action = action
        self.env = parent.env
        self.action_space_size = self.env.action_space.n
        self.child_total_value = collections.defaultdict(int)
        self.child_number_visits = collections.defaultdict(int)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value


class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env


class MCTS:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
        # self.prior_cache = {}
        self.database_posterior = {}
        self.temp_threshold = mcts_param["temp_threshold"]
        self.use_Q = mcts_param["use_Q"]
        self.mcts_action = mcts_param["mcts_action"]
        # self.en_onehot = mcts_param["en_onehot"]

    def compute_action(self, node: StateNode):
        """
        stateが最終で，次のactionがstopしかないとき，
        """
        # 最終ステップならmctsしない
        # done_patients_ratio = node.obs["obs"][-1]
        # done_patients_ratio = node.obs[-1]
        # if done_patients_ratio == 1.0:
        #     # tree_policy = node.child_number_visits / node.number_visits
        #     pure_policy, _ = self.model.compute_priors_and_value(node.obs)

        #     """
        #     def best_action(self):
        #         child_score = self.child_Q() + self.mcts.c_puct * self.child_U() # self.mcts.c_puct == mcts_param["puct_coefficient"]
        #         masked_child_score = child_score
        #         masked_child_score[~self.valid_actions] = -np.inf
        #         return np.argmax(masked_child_score)

        #     self.valid_actions = obs["action_mask"].astype(np.bool_)
        #     """
        #     # valid_actions = node.obs["action_mask"].astype(np.bool_)
        #     masked_pure_policy = pure_policy
        #     # masked_pure_policy[~valid_actions] = 0
        #     masked_pure_policy = masked_pure_policy / sum(masked_pure_policy)
        #     if self.exploit:
        #         action = np.argmax(masked_pure_policy)
        #     else:
        #         action = np.random.choice(
        #             np.arange(node.action_space_size), p=masked_pure_policy
        #         )

        #     # データ数1の交差エントロピー
        #     correct_action = node.env.MTD_true + 3
        #     correct_action_onehot = np.identity(node.env.action_space.n)[correct_action]
        #     return correct_action_onehot, action

        # else:
        for _ in range(self.num_sims):
            assert type(node) == StateNode
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.compute_priors_and_value(leaf.obs)
                # assert not any(np.isnan(np.hstack([child_priors, value])))
                if self.add_dirichlet_noise and leaf == node:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size
                    )
                """無くしてる"""
                # value = max(value, max(node.env.cal_MTD_posterior(leaf.obs)))
                # assert value > 0.0
                # print(value)
                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)

        """
        Q最大化に変更
        """
        if self.use_Q:
            tree_policy = node.child_Q() + 1
        else:
            tree_policy = node.child_number_visits / node.number_visits
        # print(tree_policy)
        # if not np.max(tree_policy) > 0.0:
        #     print(f"{tree_policy}=")
        #     print(f"{node.child_number_visits=}")
        #     print(f"{node.total_value=}")
        # assert np.max(tree_policy) > 0.0
        tree_policy = tree_policy / np.max(
            tree_policy
        )  # to avoid overflows when computing softmax
        """
        ↓変更1/temp
        
        """
        """
        obs[-1]では，timestepが少なくなるごとにtempを調整している．
        """
        # print(f"{tree_policy=}")
        if (
            self.exploit
            or self.temp_threshold
            <= round((node.env.N_total) * node.obs[-1] / node.env.N_cohort)
            # or math.isclose(node.obs[-1], 0)
        ):
            max_index = np.argmax(tree_policy)
            tree_policy = np.zeros_like(tree_policy)
            tree_policy[max_index] = 1.0
        else:
            # temperature = 1 - node.obs[-1]
            assert self.temperature != 0.0
            tree_policy = np.power(tree_policy, 1 / self.temperature)
            tree_policy = tree_policy / np.sum(tree_policy)
        """
        actionの選択
        """
        if self.mcts_action:
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        else:
            action = np.random.choice(
                np.arange(node.action_space_size),
                p=self.model.compute_priors_and_value(node.obs)[0],
            )
        assert not any(np.isnan(np.hstack([tree_policy, action])))
        # tree_policy = [1 / node.env.action_space.n] * node.env.action_space.n
        # action = np.random.randint(node.env.action_space.n)
        # priors, _ = self.model.compute_priors_and_value(leaf.obs)
        # assert priors[action] != 0.0  # Falseになるとき，cross entropy == infty
        
        # print(f"{node.child_number_visits=}")
        
        # if self.en_onehot:
        #     tree_policy = np.identity(len(tree_policy))[action]
        return tree_policy, action  # , node.children[action]

    def cal_MTD_posterior(self, obs):
        """
        np.array([self.current_dose / (self.D - 1)]),
        self.Ns / self.N_total, # self.Nsはnp.arrayだからこのtermもnp.array
        self.DLTs / self.N_total,
        np.array([sum(self.DLTs) / self.N_total]),
        np.array([sum(self.Ns) / self.N_total])
        """
        observation = obs["obs"]
        Ns = round(observation[1 : 1 + self.env.D] * self.N_total)
        DLTs = round(observation[1 + self.env.D : 1 + self.env.D * 2] * self.N_total)
        scenarios = self.env.scenarios

        num_scenarios = len(self.env.scenarios)
        priors = [1.0 / num_scenarios for _ in range(num_scenarios)]
        likelihoods = [
            self.cal_likelihood(Ns=Ns, DLTs=DLTs, scenario=scenario)
            for scenario in scenarios
        ]
        posteriors = np.array(
            [likelihood * prior for likelihood, prior in zip(likelihoods, priors)]
        )
        posteriors = posteriors / sum(posteriors)

        MTD_trues = self.env.MTD_trues
        posterior_MTD = [0.0 for _ in (self.env.D + 1)]
        for index in len(scenarios):
            posterior_MTD[MTD_trues[index]] += posteriors[index]
        return posterior_MTD

    def cal_likelihood(self, Ns, DLTs, scenario):
        likelihood = 1.0
        for p, N, DLT in zip(scenario, Ns, DLTs):
            likelihood *= p**N * (1 - p) ** (N - DLT)
        return likelihood


    # @cache
    # def compute_priors_and_value(self, obs):
    #     return self.model.compute_priors_and_value(obs)