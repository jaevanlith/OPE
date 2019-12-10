import logging
import itertools

from typing import List

import numpy as np

class TraditionalIS(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def evaluate(self, info):

        (actions,
        rewards,
        base_propensity,
        target_propensities,
        estimated_q_values) = info

        num_actions = len(target_propensities[0][0])
        actions = [np.eye(num_actions)[act] for act in actions]

        base_propensity_for_logged_action = [np.sum(np.multiply(bp, acts), axis=1) for bp, acts in zip(base_propensity, actions)]
        target_propensity_for_logged_action = [np.sum(np.multiply(tp, acts), axis=1) for tp, acts in zip(target_propensities, actions)]

        importance_weights = [ np.array(p_target)/np.array(p_base) for p_target, p_base in zip(target_propensity_for_logged_action, base_propensity_for_logged_action)]

        V_IS = self.IS(importance_weights, rewards)
        V_step_IS = self.step_IS(importance_weights, rewards)
        V_WIS = self.WIS(importance_weights, rewards)
        V_step_WIS = self.step_WIS(importance_weights, rewards)
        V_naive = self.naive(importance_weights, rewards)

        return V_naive, V_IS, V_step_IS, V_WIS, V_step_WIS

    def naive(self, episode_rhos, episode_rews):
        V_naive = [np.sum(self.gamma**np.arange(len(rews)) * np.array(rews))   for rhos, rews in zip(episode_rhos, episode_rews)]
        return np.mean(V_naive)

    def IS(self, episode_rhos, episode_rews):
        V_IS = [np.prod(rhos) * np.sum(self.gamma**np.arange(len(rews)) * np.array(rews))  for rhos, rews in zip(episode_rhos, episode_rews)]
        return np.mean(V_IS)

    def step_IS(self, episode_rhos, episode_rews):
        V_step_IS = [np.sum(self.gamma**np.arange(len(rews)) * np.cumprod(rhos) * np.array(rews))  for rhos, rews in zip(episode_rhos, episode_rews)]
        return np.mean(V_step_IS)

    def WIS(self, episode_rhos, episode_rews):
        V_WIS = [np.prod(rhos) * np.sum(self.gamma**np.arange(len(rews)) * np.array(rews))  for rhos, rews in zip(episode_rhos, episode_rews)]
        return np.sum(V_WIS) / np.sum([np.prod(rhos) for rhos in  episode_rhos])

    def step_WIS(self, episode_rhos, episode_rews):
        def to_equal_length(x, fill_value):
            x_equal_length = np.array(
                list(itertools.zip_longest(*x, fillvalue=fill_value))
            ).swapaxes(0, 1)
            return x_equal_length

        ws = np.sum(np.cumprod(to_equal_length(episode_rhos, 1), axis=1), axis=0)
        V_step_IS = [np.sum(self.gamma**np.arange(len(rews)) * np.cumprod(rhos) / ws[:len(rhos)] * np.array(rews))  for rhos, rews in zip(episode_rhos, episode_rews)]
        return np.sum(V_step_IS)