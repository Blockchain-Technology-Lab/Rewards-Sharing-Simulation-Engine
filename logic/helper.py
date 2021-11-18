# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:15:26 2021

@author: chris
"""
from numpy.random import default_rng
import csv
import pandas as pd
import pathlib

TOTAL_EPOCH_REWARDS_R = 1
MAX_NUM_POOLS = 1000


def generate_stake_distr(num_agents, total_stake=1, pareto_param=None, seed=156):
    """
    Generate a distribution for the players' initial stake (wealth),
    sampling from a Pareto distribution
    :param pareto_param:
    :param num_agents:
    :param total_stake:
    :return:
    """
    rng = default_rng(seed=seed)
    if pareto_param > 0:
        # Sample from a Pareto distribution with the specified shape
        stake_sample = rng.pareto(pareto_param, num_agents)
    else:
        # Sample from file that contains the (real) stake distribution
        distribution_file = 'stake_distribution_275.csv'
        all_stakes = get_stakes_from_file(distribution_file)
        stake_sample = rng.choice(all_stakes, num_agents)
    normalized_stake_sample = normalize_distr(stake_sample, normal_sum=total_stake)
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


def generate_cost_distr(num_agents, low, high, seed=156):
    """
    Generate a distribution for the players' costs of operating pools,
    sampling from a unifrom distribution
    :param num_agents:
    :param low:
    :param high:
    :return:
    """
    rng = default_rng(seed=seed)
    costs = rng.uniform(low=low, high=high, size=num_agents)
    return costs


def normalize_distr(dstr, normal_sum=1):
    """
    returns an equivalent distribution where the sum equals 1 (or another value defined by normal_sum)
    :param dstr:
    :param normal_sum:
    :return:
    """
    s = sum(dstr)
    if s == 0:
        return dstr
    nrm_dstr = [normal_sum * i / s for i in dstr]
    flt_error = normal_sum - sum(nrm_dstr)
    nrm_dstr[-1] += flt_error
    return nrm_dstr


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
    l = min(pledge, beta)
    s = min(stake, beta)
    reward = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (s + (l * alpha * ((s - l * (1 - s / beta)) / beta)))
    return reward


def calculate_pool_stake_NM(pool_id, pools, beta, k):
    """
    Calculate the non-myopic stake of a pool, given the pool and the state of the system (current pools)
    :param pool_id:
    :param pools: dictionary of pools with the pool id as the key
    :param beta:
    :param k:
    :return:
    """
    desirabilities = {pool_id: pool.calculate_desirability() for pool_id, pool in pools.items()}
    potential_profits = {pool_id: pool.potential_profit for pool_id, pool in pools.items()}
    rank = calculate_ranks(desirabilities, potential_profits)[pool_id]
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


def to_latex(row_list, sim_id):
    row_list_latex = [row[2:4] + row[5:8] + row[9:10] + row[12:14] for row in row_list]
    df = pd.DataFrame(row_list_latex[1:], columns=row_list_latex[0])
    # shift desirability rank column to first position to act as index
    first_column = df.pop('Pool desirability rank')
    df.insert(0, 'Pool desirability rank', first_column)
    sorted_df = df.sort_values(by=['Pool desirability rank'], ascending=True)

    latex_dir = "output/latex/"
    path = pathlib.Path.cwd() / latex_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with open(latex_dir + sim_id + "-output.tex", 'w', newline='') as file:
        sorted_df.to_latex(file, index=False)
