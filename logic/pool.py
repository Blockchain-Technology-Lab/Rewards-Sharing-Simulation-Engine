# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:16 2021

@author: chris
"""

from dataclasses import dataclass, field #can use field to determine which fields are printed in the repr method etc

@dataclass
class Pool:
    owner: int
    cost: float
    pledge: float
    stake: float
    margin: float = 0
    desirability: float = -1
    stake_NM: float = -1
    
    def __init__(self, cost, pledge, owner, margin=0):
        self.margin = margin
        self.cost = cost
        self.pledge = pledge
        self.stake = pledge
        self.owner = owner
        self.desirability = 0
        self.stake_NM = 0

        
    def update_stake(self, stake):
        self.stake += stake

    def calculate_desirability(self, potential_profit):
        """
        Note: this follows the paper's approach, where the desirability is always non-negative
        :param potential_profit:
        :return:
        """
        self.desirability = (1 - self.margin) * potential_profit if potential_profit > 0 else 0
        return self.desirability

    def calculate_desirability_myWay(self, potential_profit):
        """
        Note: the desirability can be negative (if the pool's potential reward does not suffice to cover its cost)
        :param potential_profit:
        :return:
        """
        self.desirability = (1 - self.margin) * potential_profit
        return self.desirability


    def calculate_stake_NM_myWay(self, beta, sat_prob):
        self.stake_NM = sat_prob * beta + (1 - sat_prob) * self.pledge
        return self.stake_NM

    def calculate_stake_NM(self, k, beta, rank):
        self.stake_NM = self.pledge if rank > k else max(beta, self.stake)
        return self.stake_NM

