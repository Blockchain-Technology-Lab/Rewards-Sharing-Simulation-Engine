# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:15:26 2021

@author: chris
"""
from numpy.random import default_rng
from scipy import stats
import csv
import pandas as pd
import pathlib
from math import sqrt
from functools import lru_cache
import heapq
import time

TOTAL_EPOCH_REWARDS_R = 1
MAX_NUM_POOLS = 1000
MIN_STAKE_UNIT = 2.2e-17 #todo change to reflect how much 1 lovelace is depending on total stake?
MIN_COST_PER_POOL = 1e-6


def read_stake_distr_from_file(filename='synthetic-stake-distribution-10K-active-agents', num_agents=10000, seed=42):
    stk_dstr = []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            stk_dstr.append(float(row[0]))
    if num_agents == len(stk_dstr):
        return stk_dstr
    rng = default_rng(seed=int(seed))
    if num_agents < len(stk_dstr):
        return rng.choice(stk_dstr, num_agents, replace=False)
    return rng.choice(stk_dstr, num_agents, replace=True)


def generate_stake_distr_pareto(num_agents, pareto_param=2, seed=156, truncation_factor=-1, total_stake=-1):
    """
    Generate a distribution for the agents' initial stake (wealth),
    sampling from a Pareto distribution
    sampling from a Pareto distribution
    :param pareto_param: the shape parameter to be used for the Pareto distribution
    :param num_agents: the number of samples to draw
    :param total_stake: the sum to normalize all stakes to, if positive
    :return:
    """
    rng = default_rng(seed=int(seed))
    # Sample from a Pareto distribution with the specified shape
    a, m = pareto_param, 1.  # shape and mode
    stake_sample = list((rng.pareto(a, num_agents) + 1) * m)
    if truncation_factor > 0:
        stake_sample = truncate_pareto(rng, (a, m), stake_sample, truncation_factor)
    if total_stake > 0:
        stake_sample = normalize_distr(stake_sample, normal_sum=total_stake)
    return stake_sample


def truncate_pareto(rng, pareto_params, sample, truncation_factor):
    a, m = pareto_params
    while 1:
        # rejection sampling to ensure that the distribution is truncated
        max_value = max(sample)
        if max_value > sum(sample) / truncation_factor:
            sample.remove(max_value)
            sample.append((rng.pareto(a) + 1) * m)
        else:
            return sample


def generate_stake_distr_flat(num_agents, total_stake=1):
    stake_per_agent = total_stake / num_agents if num_agents > 0 else 0
    return [stake_per_agent for _ in range(num_agents)]


def generate_cost_distr_unfrm(num_agents, low, high, seed=156):
    """
    Generate a distribution for the agents' costs of operating pools,
    sampling from a uniform distribution
    :param num_agents:
    :param low:
    :param high:
    :return:
    """
    rng = default_rng(seed=int(seed))
    costs = rng.uniform(low=low, high=high, size=num_agents)
    return costs


def generate_cost_distr_bands(num_agents, low, high, num_bands, seed=156):
    rng = default_rng(seed=seed)
    bands = rng.uniform(low=low, high=high, size=num_bands)
    costs = rng.choice(bands, num_agents)
    return costs


def generate_cost_distr_nrm(num_agents, low, high, mean, stddev):
    """
    Generate a distribution for the agents' costs of operating pools,
    sampling from a truncated normal distribution
    """
    costs = stats.truncnorm.rvs(low, high,
                                loc=mean, scale=stddev,
                                size=num_agents)
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

@lru_cache(maxsize=1024)
def calculate_potential_profit(pledge, cost, alpha, beta, reward_function_option, total_stake):
    """
    Calculate a pool's potential profit, which can be defined as the profit it would get at saturation level
    :param pledge:
    :param cost:
    :param alpha:
    :param beta:
    :return: float, the maximum possible profit that this pool can yield
    """
    relative_stake = beta / total_stake
    relative_pledge = pledge / total_stake
    potential_reward = calculate_pool_reward(relative_stake, relative_pledge, alpha, beta, reward_function_option, total_stake)
    return potential_reward - cost

@lru_cache(maxsize=1024)
def calculate_current_profit(stake, pledge, cost, alpha, beta, reward_function_option, total_stake):
    relative_pledge = pledge / total_stake
    relative_stake = stake / total_stake
    reward = calculate_pool_reward(relative_stake, relative_pledge, alpha, beta, reward_function_option, total_stake)
    return reward - cost

@lru_cache(maxsize=1024)
def calculate_pool_reward(relative_stake, relative_pledge, alpha, beta, reward_function_option, total_stake, curve_root=3, crossover_factor=8):
    beta = beta / total_stake
    if reward_function_option == 0:
        return calculate_pool_reward_old(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 1:
        return calculate_pool_reward_new(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 2:
        return calculate_pool_reward_flat_pledge_benefit(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 3:
        return calculate_pool_reward_new_sqrt(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 4:
        return calculate_pool_reward_curve_pledge_benefit(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor)
    elif reward_function_option == 5:
        return calculate_pool_reward_curve_pledge_benefit_min_first(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor)
    elif reward_function_option == 6:
        return calculate_pool_reward_curve_pledge_benefit_no_min(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor)
    else:
        raise ValueError("Invalid option for reward function.")

def calculate_pool_reward_old(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (stake_ + (pledge_ * alpha * ((stake_ - pledge_ * (1 - stake_ / beta)) / beta)))
    return r

def calculate_pool_reward_new(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * stake_ * (1 + (alpha * pledge_ / beta))
    return r

def calculate_pool_reward_flat_pledge_benefit(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (stake_ + alpha * pledge_)
    return r

def calculate_pool_reward_new_sqrt(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * stake_ * (1 + (alpha * sqrt(pledge_) / beta))
    return r

def calculate_pool_reward_curve_pledge_benefit(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor):
    crossover = beta / crossover_factor
    pledge_ = (relative_pledge ** (1 / curve_root)) * (crossover ** ((curve_root - 1) / curve_root))
    return calculate_pool_reward_old(relative_stake, pledge_, alpha, beta)

def calculate_pool_reward_curve_pledge_benefit_min_first(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor):
    crossover = beta / crossover_factor
    pledge = min(relative_pledge, beta)
    pledge_ = (pledge ** (1 / curve_root)) * (crossover ** ((curve_root - 1) / curve_root))
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (
                stake_ + (pledge_ * alpha * ((stake_ - pledge_ * (1 - stake_ / beta)) / beta)))
    return r

def calculate_pool_reward_curve_pledge_benefit_no_min(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor):
    crossover = beta / crossover_factor
    pledge_ = (relative_pledge ** (1 / curve_root)) * (crossover ** ((curve_root - 1) / curve_root))
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (
                stake_ + (pledge_ * alpha * ((stake_ - pledge_ * (1 - stake_ / beta)) / beta)))
    return r

@lru_cache(maxsize=1024)
def calculate_delegator_reward_from_pool(pool_margin, pool_cost, pool_reward, delegator_stake_fraction):
    margin_factor = (1 - pool_margin) * delegator_stake_fraction
    pool_profit = pool_reward - pool_cost
    r_d = max(margin_factor * pool_profit, 0)
    return r_d

@lru_cache(maxsize=1024)
def calculate_operator_reward_from_pool(pool_margin, pool_cost, pool_reward, operator_stake_fraction):
    margin_factor = pool_margin + ((1 - pool_margin) * operator_stake_fraction)
    pool_profit = pool_reward - pool_cost
    return pool_profit if pool_profit <= 0 else pool_profit * margin_factor

def calculate_pool_stake_NM(pool_id, pools, beta, k):
    """
    Calculate the non-myopic stake of a pool, given the pool and the state of the system (other active pools)
    :param pool_id: the id of the pool that is examined
    :param pools: dictionary of pools with the pool id as key and the pool object as value
    :param beta: the saturation point of the system
    :param k: the desired number of pools of the system
    :return: the value of the non-myopic stake of the pool with id pool_id
    """
    pool = pools[pool_id]
    if len(pools) <= k:
        rank_in_top_k = True
    else:
        d = [(pool.desirability, pool.potential_profit, pool.stake, -pool_id) for pool_id, pool in pools.items()]
        top_k_pools = heapq.nlargest(k, d)
        top_k_pool_ids = [-p[3] for p in top_k_pools]
        rank_in_top_k = pool_id in top_k_pool_ids

    return calculate_pool_stake_NM_from_rank(pool_pledge=pool.pledge, pool_stake=pool.stake, beta=beta, rank_in_top_k=rank_in_top_k)

def calculate_ranks(ranking_dict, *tie_breaking_dicts, rank_ids=True):
    """
    Rank the values of a dictionary from highest to lowest (highest value gets rank 1, second highest rank 2 and so on)
    @param ranking_dict:
    @param tie_breaking_dicts:
    @param rank_ids: if True, then the lowest id (e.g. the one corresponding to a pool created earlier) takes precedence
                    during ties that persist even after the other tie breaking rules have been applied.
                    If False and ties still exist, then the tie breaking is arbitrary.
    @return: dictionary with the item id as the key and the calculated rank as the value
    """
    if rank_ids:
        tie_breaking_dicts = list(tie_breaking_dicts)
        tie_breaking_dicts.append({key: -key for key in ranking_dict.keys()})
    final_ranking_dict = {
        key:
            (ranking_dict[key],) + tuple(tie_breaker_dict[key] for tie_breaker_dict in tie_breaking_dicts)
        for key in ranking_dict
    }
    ranks = {
        sorted_item[0]: i + 1 for i, sorted_item in
        enumerate(sorted(final_ranking_dict.items(), key=lambda item: item[1], reverse=True))
    }
    return ranks

def save_as_latex_table(df, sim_id, output_dir):
    path = pathlib.Path.cwd() / output_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with open(output_dir + sim_id + "-output.tex", 'w', newline='') as file:
        df.save_as_latex_table(file, index=False)

def generate_execution_id(args_dict):
    num_args_to_use = 5
    max_characters = 100
    primitive = (int, str, bool, float)
    return "".join([str(key) + '-' + str(value) + '-' for key, value in list(args_dict.items())[:num_args_to_use]
                    if type(value) in primitive])[:max_characters]

@lru_cache(maxsize=1024)
def calculate_cost_per_pool(num_pools, initial_cost, cost_factor):
    """
    Calculate the average cost of an agent's pools, assuming that any additional pool costs less than the previous one
    Specifically if the first pool costs c1 and we use a factor of 0.6 then a second pool would cost c2 = 0.6 * c1,
    a third pool would cost c3 = 0.6 * c2 = 0.6^2 * c1, and so on. Can be calculated using the sum of a geometrical sequence.
    @param num_pools:
    @param initial_cost:
    @param cost_factor:
    @return:
    """
    if cost_factor < 1:
        return max((initial_cost * (1 - cost_factor ** num_pools) / (1 - cost_factor)) / num_pools, MIN_COST_PER_POOL)
    else:
        return initial_cost

@lru_cache(maxsize=1024)
def calculate_cost_per_pool_fixed_fraction(num_pools, initial_cost, cost_factor):
    return (initial_cost + (num_pools - 1) * cost_factor * initial_cost) / num_pools

@lru_cache(maxsize=1024)
def calculate_pool_desirability(margin, potential_profit):
    return max((1 - margin) * potential_profit, 0)

@lru_cache(maxsize=1024)
def calculate_myopic_pool_desirability(stake, pledge, cost, margin, alpha, beta, total_stake):
    current_profit = calculate_current_profit(stake, pledge, cost, alpha, beta, total_stake)
    return max((1 - margin) * current_profit, 0)

@lru_cache(maxsize=1024)
def calculate_pool_stake_NM_from_rank(pool_pledge, pool_stake, beta, rank_in_top_k):
    return max(beta, pool_stake) if rank_in_top_k else pool_pledge

@lru_cache(maxsize=1024)
def determine_pledge_per_pool(agent_stake, beta, num_pools):
    """
    The agents choose to allocate their entire stake as the pledge of their pools,
    so they divide it equally among them
    However, if they saturate all their pools with pledge and still have remaining stake,
    then they don't allocate all of it to their pools, as a pool with such a pledge above saturation
     would yield suboptimal rewards
    :return: list of pledge values
    """
    if num_pools <= 0:
        raise ValueError("Agent tried to calculate pledge for zero or less pools.")
    return [min(agent_stake / num_pools, beta)] * num_pools

def export_csv_file(rows, filename ):
    today = time.strftime("%d-%m-%Y")
    output_dir = "output/" + today
    path = pathlib.Path.cwd() / output_dir
    filename = path / filename
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

