# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:13:20 2021

@author: chris
"""

import random

MAX_POOLS = 10 # max number of pools that a stakeholder can open 

class Strategy: #todo rename to multi-pool strategy and include single pool strategies?
    def __init__(self, number_of_pools=0, pool_pledges=[], pool_margins=[], stake_allocations=[]):
        self.number_of_pools = number_of_pools
        self.pool_pledges = pool_pledges
        self.pool_margins = pool_margins
        self.stake_allocations = stake_allocations
        # maybe also have a "last_update" field? to perform updates more easily 
        
    ''' 
    Creates a random **not necessarily valid** strategy
    '''
    def set_random_strategy(self, n_pools):
        self.number_of_pools = random.randint(0, MAX_POOLS)
        self.pool_pledges = [random.random() for pool in range(self.number_of_pools)]
        self.pool_margins = [random.random() for pool in range(self.number_of_pools)]
        self.stake_allocations = [random.random() for pool in range(n_pools)]
        
        return self
        
    def print_(self):
        print("Number of pools owned: {} \n Pledges: {} \n Margins: {} \n Allocations: {}"
              .format(self.number_of_pools, self.pool_pledges, self.pool_margins, self.stake_allocations))
        
class SinglePoolStrategy:
    def __init__(self, pledge=0, margin=0, stake_allocations=[]):
        self.pledge = pledge
        self.margin = margin
        self.stake_allocations = stake_allocations
        # maybe also have a "last_update" field? to perform updates more easily 
        
    ''' 
    Creates a random **not necessarily valid** strategy
    '''
    def set_random_strategy(self, n_pools):
        self.pledge = random.random() 
        self.margin = random.random()
        self.stake_allocations = [random.random() for pool in range(n_pools)]
        #maybe stake allocations better as hashmap/dict? otherwise very sparse
        
        return self
        
    def print_(self):
        print("Pledge: {} \n Margin: {} \n Allocations: {}"
              .format(self.pledge, self.margin, self.stake_allocations))