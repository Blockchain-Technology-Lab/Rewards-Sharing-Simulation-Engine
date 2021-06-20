# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:15:26 2021

@author: chris
"""

from numpy.random import default_rng

PARETO_ALPHA_PARAM = 1.5
C_MIN = 0.001
C_MAX = 0.002
TOTAL_EPOCH_REWARDS_R = 1 #todo find suitable value


def generate_stake_distr(num_agents, total_stake):
    rng = default_rng(seed=156) 
    distr = rng.pareto(PARETO_ALPHA_PARAM, num_agents)
    return normalize_distr(distr, normal_sum=total_stake)

def generate_cost_distr(num_agents, low=C_MIN, high=C_MAX):
    rng = default_rng(seed=156) 
    return rng.uniform(low=low, high=high, size=num_agents)
    
#  returns an equivalent distribution where the sum equals 1 (or another value defined by normal_sum)
def normalize_distr(distr, normal_sum=1):
    s = sum(distr)
    return [normal_sum*float(i)/s for i in distr] if s != 0 else distr

def calculate_pool_reward(stake, pledge, alpha, beta):
    # use current formula but keep in mind that it may change (e.g. also depend on aggregate values)
    l = min (pledge, beta)
    s = min (stake, beta)
    reward = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (s + (l*alpha*((s-l*(1 - s/beta))/beta)))
    return reward

def flatten_list(l):
    return sum(l, [])