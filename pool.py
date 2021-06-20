# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:16 2021

@author: chris
"""

from dataclasses import dataclass

@dataclass
class Pool:
    margin: float
    cost: float
    pledge: float
    stake: float
    owner: int
    
    def __init__(self, margin, cost, pledge, owner):
        self.margin = margin
        self.cost = cost
        self.pledge = pledge
        self.stake = pledge
        self.owner = owner
        
    def update_stake(self, stake):
        self.stake += stake
