# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:13:20 2021

@author: chris
"""
from collections import defaultdict


STARTING_MARGIN = 0.2
MARGIN_INCREMENT = 0.01


class Strategy:
    __slots__ = ['is_pool_operator', 'num_pools', 'pledges', 'margins', 'owned_pools', 'stake_allocations']

    def __init__(self, pledges=None, margins=None, stake_allocations=None,
                 is_pool_operator=False, owned_pools=None, num_pools=0):
        if pledges is None:
            pledges = []
        if margins is None:
            margins = []
        if owned_pools is None:
            owned_pools = defaultdict(lambda: None)
        if stake_allocations is None:
            stake_allocations = defaultdict(lambda: 0)
        self.stake_allocations = stake_allocations
        self.is_pool_operator = is_pool_operator
        self.num_pools = num_pools
        self.pledges = pledges
        self.margins = margins
        self.owned_pools = owned_pools
