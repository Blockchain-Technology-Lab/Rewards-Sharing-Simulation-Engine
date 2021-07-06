# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:15:26 2021

@author: chris
"""

import numpy as np
from numpy.random import default_rng

from logic.pool import Pool

TOTAL_EPOCH_REWARDS_R = 1


def generate_stake_distr(num_agents, total_stake, pareto_param, truncated=False):
    """
    Generate a distribution for the players' initial stake (wealth),
    sampling from a Pareto distribution
    :param num_agents:
    :param total_stake:
    :return:
    """
    rng = default_rng(seed=156)
    distr = rng.pareto(pareto_param, num_agents)
    return normalize_distr(distr, normal_sum=total_stake)


def generate_cost_distr(num_agents, low, high):
    """
    Generate a distribution for the players' costs of operating pools,
    sampling from a unifrom distribution
    :param num_agents:
    :param low:
    :param high:
    :return:
    """
    rng = default_rng(seed=156)
    return rng.uniform(low=low, high=high, size=num_agents)


def normalize_distr(distr, normal_sum=1):
    """
    returns an equivalent distribution where the sum equals 1 (or another value defined by normal_sum)
    :param distr:
    :param normal_sum:
    :return:
    """
    s = sum(distr)
    return [normal_sum * float(i) / s for i in distr] if s != 0 else distr


def calculate_potential_profit(pledge, cost, alpha, beta):
    """

    :param pledge:
    :param cost:
    :param alpha:
    :param beta:
    :return: float, the maximum possible profit that this pool can yield
    """
    potential_reward = calculate_pool_reward(beta, pledge, alpha, beta)
    return potential_reward - cost


def calculate_pool_reward(stake, pledge, alpha, beta):
    # use current formula but keep in mind that it may change (e.g. also depend on aggregate values)
    l = min(pledge, beta)
    s = min(stake, beta)
    reward = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (s + (l * alpha * ((s - l * (1 - s / beta)) / beta)))
    return reward


def calculate_pool_stake_NM(pool, pools, pool_index, alpha, beta, k):
    """
    Calculate the non-myopic stake of a pool, given the pool and the state of the system
    :param pool:
    :param pools:
    :param pool_index:
    :param alpha:
    :param beta:
    :param k:
    :return:
    """
    desirabilities = [pool.desirability if pool is not None else 0 for pool in pools]
    # add desirability of current pool
    if pool is None:
        pool = pools[pool_index]
    potential_pool_profit = calculate_potential_profit(pool.pledge, pool.cost, alpha, beta)
    desirability = pool.calculate_desirability(potential_pool_profit)
    desirabilities[pool_index] = desirability
    # todo maybe cache?
    rank = calculate_rank(desirabilities,
                          pool_index)  # the rank can be defined as the index of the sorted desirabilities, in descending order
    return pool.calculate_stake_NM(k, beta, rank)


def calculate_ranks(desirabilities):
    ranks = [0 for i in range(len(desirabilities))]
    indices = np.argsort(
        -np.array(desirabilities))  # the rank is the index of the sorted desirabilities (in descending order)
    for rank, index in enumerate(indices):
        ranks[index] = rank
    return ranks


def calculate_rank(desirabilities, player_id):
    ranks = calculate_ranks(desirabilities)
    return ranks[player_id]


def is_list_flat(l):
    # assume that the list is homogeneous, so only check the first element
    return not isinstance(l[0], list)


def flatten_list(l):
    if not is_list_flat(l):
        l = sum(l, [])
    return l


'''
unused for now but could be useful in the future

# Examine whether a pool leading strategy has the potential to rank the player's pool in the top k
def check_pool_potential(strategy, model, player_id):
    alpha = model.alpha
    beta = model.beta
    k = model.k

    # to check if the pool's margin is competitive we look at all other players
    players = model.schedule.agents
    pools = model.pools.copy()

    pool = Pool(players[player_id].cost, strategy.pledge, player_id,
                strategy.margin)  # todo create pool constructor that takes strategy as argument
    potential_profit = calculate_potential_profit(pool.pledge, pool.cost, alpha, beta)
    pool.calculate_desirability(potential_profit)
    pools[player_id] = pool

    for i, player in enumerate(players):
        # we assume that the ones who already run pools will keep running them with the same margin
        # for players who don't run pools, we assume that they will start a pool with margin m' =(u-(r-c)q)/((r-c)(1-q))
        if pools[i] is None:
            u = player.utility
            s = player.stake
            c = player.cost
            sigma = max(s, beta)
            r = calculate_pool_reward(sigma, s, alpha, beta)
            q = s / sigma
            if r <= c:
                continue  # since the potential reward cannot even cover the cost, player i has no chance for a pool
            m_prime = max((u - (r - c) * q) / ((r - c) * (1 - q)), 0) if q < 1 else 0
            # monitor m_prime (is it often 0?)
            pool = Pool(c, s, i, m_prime)
            potential_profit = calculate_potential_profit(pool.pledge, pool.cost, alpha, beta)
            pool.calculate_desirability(potential_profit)
            pools[i] = pool

    desirabilities = [pool.desirability if pool is not None else 0 for pool in pools]
    rank = calculate_rank(desirabilities, player_id)
    result = rank < k
    return rank < k

def calculate_pool_saturation_prob(desirabilities, pool_index):
    # todo cache the softmax result to use in other calls?
    # todo add temperature param to adjust the influence of the desirability on the resulting probability
    probs = softmax(desirabilities)
    return probs[pool_index]


def calculate_pool_stake_NM_myWay(pool, pools, pool_index, alpha, beta):
    desirabilities = [pool.desirability if pool is not None else 0 for pool in pools]
    # add desirability of current pool
    if (pool is None):
        pool = pools[pool_index]
    potential_pool_profit = calculate_potential_profit(pool.pledge, pool.cost, alpha, beta)
    desirability = pool.calculate_desirability(potential_pool_profit)
    desirabilities[pool_index] = desirability
    sat_prob = calculate_pool_saturation_prob(desirabilities, pool_index)
    return pool.calculate_stake_NM(beta, sat_prob)
    
def softmax(vector):
    np_vector = np.array(vector)
    e = np.exp(np_vector)
    return e / np.sum(e)

'''
