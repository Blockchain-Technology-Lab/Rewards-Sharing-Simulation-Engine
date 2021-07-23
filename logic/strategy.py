# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:13:20 2021

@author: chris
"""
from abc import ABC, abstractmethod
import random
from collections import defaultdict

import logic.helper as hlp

MAX_MARGIN = 0.2
MARGIN_INCREMENT = 0.01


class Strategy(ABC):

    @abstractmethod
    def __init__(self, stake_allocations, is_pool_operator, num_pools):
        self.stake_allocations = stake_allocations
        self.is_pool_operator = is_pool_operator
        self.num_pools = num_pools

    @abstractmethod
    def create_random_operator_strategy(self, pools, player_id, player_stake):
        pass

    @abstractmethod
    def create_random_delegator_strategy(self, pools, player_id, player_stake):
        pass


class SinglePoolStrategy(Strategy):
    def __init__(self, pledge=-1, margin=-1, stake_allocations=[], is_pool_operator=False, num_pools=0):
        super().__init__(stake_allocations, is_pool_operator, num_pools)
        self.pledge = pledge
        self.margin = margin

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


class MultiPoolStrategy(Strategy):
    def __init__(self, pledges=None, margins=None, stake_allocations=None, is_pool_operator=False, owned_pools=None,
                 num_pools=0):
        if pledges is None:
            pledges = []
        if margins is None:
            margins = []
        if owned_pools is None:
            owned_pools = defaultdict(lambda: None)
        if stake_allocations is None:
            stake_allocations = defaultdict(lambda: 0)
        super().__init__(stake_allocations, is_pool_operator, num_pools)
        self.pledges = pledges
        self.margins = margins
        self.owned_pools = owned_pools

    def create_random_operator_strategy(self, pools, player_id, player_stake):
        pass

    def create_random_delegator_strategy(self, pools, player_id, player_stake):
        pass
