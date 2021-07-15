# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:16 2021

@author: chris
"""
import helper as hlp


class Pool:

    def __init__(self, cost, pledge, owner, margin, alpha, beta):
        self.margin = margin
        self.cost = cost
        self.pledge = pledge
        self.stake = pledge
        self.owner = owner
        self.potential_profit = hlp.calculate_potential_profit(pledge, cost, alpha, beta)

    def update_stake(self, stake):
        self.stake += stake

    def calculate_desirability(self):
        """
        Note: this follows the paper's approach, where the desirability is always non-negative
        :param potential_profit:
        :return:
        """
        return (1 - self.margin) * self.potential_profit if self.potential_profit > 0 else 0

    def calculate_desirability_myWay(self, potential_profit):
        """
        Note: the desirability can be negative (if the pool's potential reward does not suffice to cover its cost)
        :param potential_profit:
        :return:
        """
        self.desirability = (1 - self.margin) * potential_profit
        return self.desirability

    def calculate_stake_NM_myWay(self, beta, sat_prob):
        return sat_prob * beta + (1 - sat_prob) * self.pledge

    def calculate_stake_NM(self, k, beta, rank):
        return self.pledge if rank >= k else max(beta, self.stake)
