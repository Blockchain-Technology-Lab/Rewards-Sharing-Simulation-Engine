# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:15:26 2021

@author: chris
"""
import random
from numpy.random import default_rng
import csv

TOTAL_EPOCH_REWARDS_R = 1


def generate_stake_distr(num_agents, total_stake=1, pareto_param=None):
    """
    Generate a distribution for the players' initial stake (wealth),
    sampling from a Pareto distribution
    :param pareto_param:
    :param num_agents:
    :param total_stake:
    :return:
    """
    if pareto_param is not None:
        # Sample from a Pareto distribution with the specified shape
        rng = default_rng(seed=156)
        stake_sample = rng.pareto(pareto_param, num_agents)
    else:
        # Sample from file that contains the (real) stake distribution
        distribution_file = 'stake_distribution_275.csv'
        all_stakes = get_stakes_from_file(distribution_file)
        stake_sample = random.sample(all_stakes, num_agents)
    normalized_stake_sample = normalize_distr(stake_sample, normal_sum=total_stake)
    random.shuffle(normalized_stake_sample)
    return normalized_stake_sample


def get_stakes_from_file(filename):  # todo if we keep this function replace with sth more efficient (e.g. pandas)
    stakes = []
    with open(filename) as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i > 0:  # skip header row
                stake = float(row[-1])  # the last column represents the wallet's stake
                if stake > 0:
                    stakes.append(stake)
    return stakes


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
    costs = rng.uniform(low=low, high=high, size=num_agents)
    random.shuffle(costs)
    return costs


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
    Calculate a pool's potential profit, which can be defined as the profit it would get at saturation level

    :param pledge:
    :param cost:
    :param alpha:
    :param beta:
    :return: float, the maximum possible profit that this pool can yield
    """
    potential_reward = calculate_pool_reward(beta, pledge, alpha, beta)
    return potential_reward - cost


def calculate_current_profit(stake, pledge, cost, alpha, beta):
    reward = calculate_pool_reward(stake, pledge, alpha, beta)
    return reward - cost


def calculate_pool_reward(stake, pledge, alpha, beta):
    # use current formula but keep in mind that it may change (e.g. also depend on aggregate values)
    l = min(pledge, beta)
    s = min(stake, beta)
    reward = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (s + (l * alpha * ((s - l * (1 - s / beta)) / beta)))
    return reward


def calculate_pool_stake_NM(pool_id, pools, beta, k):
    """
    Calculate the non-myopic stake of a pool, given the pool and the state of the system (current pools)
    :param pool_id:
    :param pools:
    :param beta:
    :param k:
    :return:
    """
    desirabilities = {id: pool.calculate_desirability() for id, pool in pools.items()}
    rank = calculate_ranks(desirabilities)[
        pool_id]  # todo use potential profit to break ties between desirability ranks
    pool = pools[pool_id]
    return pool.calculate_stake_NM(k, beta, rank)


def calculate_ranks(ranking_dict, secondary_ranking_dict=None):
    if secondary_ranking_dict is None:
        total_ranking_dict = ranking_dict
    else:
        total_ranking_dict = {key: (ranking_dict[key], secondary_ranking_dict[key]) for key in ranking_dict}
    ranks = {sorted_item[0]: i + 1 for i, sorted_item in
             enumerate(sorted(total_ranking_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranks


'''
unused for now but could be useful in the future

# Examine whether a pool leading strategy has the potential to rank the player's pool in the top k
def check_pool_potential(strategy, model, player_id):
    alpha = model.alpha
    beta = model.beta
    k = model.k

    # to check if the pool's margin is competitive we look at all other players
    players = model.schedule.agents
    pools = deepcopy(model.pools)

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


def calculate_pool_stake_NM_myWay(pool, pools, pool_index, beta):
    desirabilities = [p.calculate_desirability() if p is not None else 0 for p in pools]
    if pool is None:
        pool = pools[pool_index]
    else:
        desirability = pool.calculate_desirability()
        desirabilities[pool_index] = desirability
    sat_prob = calculate_pool_saturation_prob(desirabilities, pool_index)
    return pool.calculate_stake_NM_myWay(beta, sat_prob)
    
def softmax(vector):
    np_vector = np.array(vector)
    e = np.exp(np_vector)
    return e / np.sum(e)


def list_is_flat(l):
    # assume that the list is homogeneous, so only check the first element
    return not isinstance(l[0], list)


def flatten_list(l):
    if len(l) == 0:
        return l
    if list_is_flat(l):
        return l
    return sum(l, [])'''
