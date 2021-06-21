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
        self.desirability = 0 #todo redundant?
        self.stake_NM = 0

        
    def update_stake(self, stake):
        self.stake += stake

    def calculate_desirability(self, potential_profit):
        """
        Note: the desirability can be negative (if the pool's potential reward does not suffice to cover its cost)
        :param potential_profit:
        :return:
        """
        self.desirability = (1 - self.margin) * potential_profit
        return self.desirability


    def calculate_stake_NM(self, beta, sat_prob):
        self.stake_NM = sat_prob * beta + (1 - sat_prob) * self.pledge
        return self.stake_NM

