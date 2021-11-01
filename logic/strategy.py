# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:13:20 2021

@author: chris
"""
STARTING_MARGIN = 0.2
MARGIN_INCREMENT = 0.01


class Strategy:
    # todo rethink slots as they may scale better
    # __slots__ = ['is_pool_operator', 'num_pools', 'pledges', 'margins', 'owned_pools', 'stake_allocations']

    def __init__(self, pledges=None, margins=None, stake_allocations=None,
                 is_pool_operator=False, owned_pools=None, num_pools=0):
        if pledges is None:
            pledges = []
        if margins is None:
            margins = []
        if owned_pools is None:
            owned_pools = dict()
        if stake_allocations is None:
            stake_allocations = dict()
        self.stake_allocations = stake_allocations
        self.is_pool_operator = is_pool_operator
        self.num_pools = num_pools
        self.pledges = pledges
        self.margins = margins
        self.owned_pools = owned_pools

    def __eq__(self, other):
        if not isinstance(other, Strategy):
            return False

        return self.is_pool_operator == other.is_pool_operator and \
               self.num_pools == other.num_pools and \
               self.pledges == other.pledges and \
               self.margins == other.margins and \
               self.owned_pools == other.owned_pools and \
               self.stake_allocations == other.stake_allocations

    def __eq__(self, other):
        if not isinstance(other, Strategy):
            return True

        return self.is_pool_operator != other.is_pool_operator or \
               self.num_pools != other.num_pools or \
               self.pledges != other.pledges or \
               self.margins != other.margins or \
               self.owned_pools != other.owned_pools or \
               self.stake_allocations != other.stake_allocations
