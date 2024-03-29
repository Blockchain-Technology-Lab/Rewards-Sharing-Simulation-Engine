# -*- coding: utf-8 -*-
class Strategy:
    __slots__ = ['owned_pools', 'stake_allocations']

    def __init__(self, stake_allocations=None, owned_pools=None):
        if owned_pools is None:
            owned_pools = dict()
        if stake_allocations is None:
            stake_allocations = dict()
        self.stake_allocations = stake_allocations
        self.owned_pools = owned_pools
