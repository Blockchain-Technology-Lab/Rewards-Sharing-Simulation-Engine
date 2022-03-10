# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:16 2021

@author: chris
"""
import logic.helper as hlp
from logic.helper import MIN_STAKE_UNIT


class Pool:

    def __init__(self, pool_id, cost, pledge, owner, alpha, beta, reward_function_option, total_stake, margin=-1, is_private=False):
        self.id = pool_id
        self.margin = margin
        self.margin_change = 0
        self.cost = cost
        self.pledge = pledge
        self.stake = pledge
        self.owner = owner
        self.is_private = is_private
        self.delegators = dict()
        self.set_potential_profit(alpha, beta, reward_function_option, total_stake)

    def set_potential_profit(self, alpha, beta, reward_function_option, total_stake):
        self.potential_profit = hlp.calculate_potential_profit(self.pledge, self.cost, alpha, beta, reward_function_option, total_stake)

    def update_delegation(self, stake, delegator_id):
        self.stake += stake
        if delegator_id in self.delegators:
            self.delegators[delegator_id] += stake
        else:
            self.delegators[delegator_id] = stake
        if self.delegators[delegator_id] <= MIN_STAKE_UNIT:
            self.delegators.pop(delegator_id)

    def calculate_desirability(self):
        """
        Note: this follows the paper's approach, where the desirability is always non-negative
        """
        return max((1 - self.margin) * self.potential_profit, 0)

    def calculate_myopic_desirability(self, alpha, beta, total_stake):
        current_profit = hlp.calculate_current_profit(self.stake, self.pledge, self.cost, alpha, beta, total_stake)
        return max((1 - self.margin) * current_profit, 0)

    def calculate_desirability_myWay(self, potential_profit):
        """
        Note: the desirability can be negative (if the pool's potential reward does not suffice to cover its cost)
        :param potential_profit:
        :return:
        """
        return (1 - self.margin) * potential_profit

    def calculate_stake_NM_myWay(self, beta, sat_prob):
        return sat_prob * beta + (1 - sat_prob) * self.pledge

    def calculate_stake_NM(self, k, beta, rank):
        return self.pledge if rank > k else max(beta, self.stake)
