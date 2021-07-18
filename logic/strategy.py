# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:13:20 2021

@author: chris
"""
from abc import ABC, abstractmethod
import random
import logic.helper as hlp

MAX_MARGIN = 0.5
MARGIN_INCREMENT = 0.01


class Strategy(ABC):

    @abstractmethod
    def __init__(self, stake_allocations, is_pool_operator):
        self.stake_allocations = stake_allocations
        self.is_pool_operator = is_pool_operator

    @abstractmethod
    def create_random_operator_strategy(self, pools, player_id, player_stake):
        pass

    @abstractmethod
    def create_random_delegator_strategy(self, pools, player_id, player_stake):
        pass


class SinglePoolStrategy(Strategy):
    def __init__(self, pledge=-1, margin=-1, stake_allocations=[], is_pool_operator=False):
        super().__init__(stake_allocations, is_pool_operator)
        self.pledge = pledge
        self.margin = margin
        # maybe also have a "last_update" field? to perform updates more easily

    def create_random_operator_strategy(self, pools, player_id, player_stake):
        margin = random.random()
        stake_allocations = [0 for i in range(len(pools))]
        pledge = stake_allocations[player_id] = player_stake  # assume that pledge == α_i
        return SinglePoolStrategy(pledge, margin, stake_allocations, True)

    def create_random_delegator_strategy(self, pools, player_id, player_stake):
        # stake allocations must sum to (at most) the player's stake
        stake_allocations = [random.random() if (i != player_id and pool is not None) else 0
                             for i, pool in enumerate(pools)]
        stake_allocations = hlp.normalize_distr(stake_allocations, normal_sum=player_stake)
        return SinglePoolStrategy(stake_allocations=stake_allocations, is_pool_operator=False)

    def print_(self):
        print("Pledge: {} \n Margin: {} \n Allocations: {}"
              .format(self.pledge, self.margin, self.stake_allocations))
