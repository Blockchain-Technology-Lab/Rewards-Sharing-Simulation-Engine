# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""

from mesa import Agent
import random

import helper as hlp
from pool import Pool
from strategy import SinglePoolStrategy
from strategy import MAX_POOLS

UTILITY_THRESHOLD = 0 #relates to inertial equilibrium 

#todo when using the sim/model object, stick to calling methods instead of accessing fields directly

class Stakeholder(Agent):
    def __init__(self, unique_id, model, agent_type, stake=0, cost=0, utility=0):
        super().__init__(unique_id, model)
        self.cost = cost #the cost of running one pool (assume that running n pools will cost n times as much)
        self.stake = stake
        self.utility = utility
        self.type = agent_type
        # pools field that points to the player's pools?
        
        self.initialize_strategy()
    
    # In every step the agent needs to decide what to do
    def step(self):   
        #self.get_status()
        #assume that all players will open a pool (sequentially)
        strategy_changed, allocation_changes = self.update_strategy()
        #print("Allocation changes: {}".format(allocation_changes))
        if (strategy_changed):
            # the player has changed their strategy, so now they have to execute it 
            self.execute_strategy(allocation_changes)
            self.model.current_step_idle = False
        #self.get_status()
        #print('----------------------')
        
        
    def initialize_strategy(self):
        strategy = SinglePoolStrategy(pledge=self.stake)
        #todo find player's competitive margin?
        strategy.stake_allocations = [0 for i in range(self.model.num_agents)]
        
        self.strategy = strategy

    def execute_strategy(self, allocation_changes):
        """
        Execute the player's current strategy
        :param allocation_changes:
        :return: void

        """
        for i,change in enumerate(allocation_changes):
            if (change != 0):
                if (i == self.unique_id):
                    if (allocation_changes[i] == self.strategy.stake_allocations[i]):
                        #means that the pool needs to be created now
                        #todo change margin
                        self.open_pool(pledge=allocation_changes[self.unique_id])
                        continue
                    elif (self.strategy.stake_allocations[i] == 0):
                        # means that the pool nees to be destroyed, since it has been abandoned by its owner
                        self.close_pool()
                        continue
                self.model.pools[i].update_stake(change)          
                    
            #todo do I have to recalculate all players' utilities now??
        

    '''def set_random_strategy(self, n_pools):
        
        strategy = SinglePoolStrategy()
        #strategy.number_of_pools = random.randint(0, MAX_POOLS)
        #todo sum to self.stake
        #strategy.pool_pledges = [random.random() for pool in range(strategy.number_of_pools)]
        #strategy.pool_margins = [random.random() for pool in range(strategy.number_of_pools)]
        #todo sum <= self.stake
        strategy.stake_allocations = [random.random() for pool in range(n_pools)]
        
        self.strategy = strategy'''
        
    def get_random_valid_strategy(self):
        """
        Creates a random **valid** strategy for the player
        Valid means:
            sum(pool_pledges) <= player's stake
            0 <= margin <= 1 for margin in pool_margins
            sum(stake_allocations) <= player's stake

        :return:
        """
        # todo make compatible with pool splitting
        #strategy.number_of_pools = random.randint(0, MAX_POOLS)
        #todo sum to self.stake
        #strategy.pool_pledges = [random.random() for pool in range(strategy.number_of_pools)]
        #strategy.pool_margins = [random.random() for pool in range(strategy.number_of_pools)]
        #todo sum <= self.stake
        
        margin = random.random()        
        sim = self.model        
        # stake alloations must sum to (at most) the player's stake
        stake_allocations = [random.random() if (i == self.unique_id or pool is not None) else 0 for i,pool in enumerate(sim.pools)]
        stake_allocations = hlp.normalize_distr(stake_allocations, normal_sum=self.stake)
        pledge = stake_allocations[self.unique_id] #assume that pledge == α_i
        
        return SinglePoolStrategy(pledge, margin, stake_allocations)
    
    def calculate_myopic_utility(self, strategy):
        utility = 0
        for i, a in enumerate(strategy.stake_allocations):
            if (a > 0):
                # player has allocated stake to this pool
                pool = self.model.pools[i]
                if (i==self.unique_id):
                    #calculate pool owner utility
                    if (pool is None):
                        # player hasn't created their pool yet, so we calculate the utility of a hypothetical pool
                        pool = Pool(strategy.margin, self.cost, strategy.pledge, self.unique_id)
                    utility += self.calculate_po_utility(pool, a)
                else:
                    #calculate delegator utility
                    if (pool is None):
                        print("POOL IS NOOOOOONE")
                        continue #todo should never be none when choosing strategy non-randomly, so raise exception or sth
                    utility += self.calculate_delegator_utility(pool, a)
        
        return utility
    
    def calculate_non_myopic_utility(self, strategy):
        pass
    
    def calculate_utility(self, strategy): 
        # todo have different utilities to choose from
        #for now just myopic utility
        if (self.type == 'M'):
            return self.calculate_myopic_utility(strategy)
        else:
            print("what are you doing here??")
            return self.calculate_non_myopic_utility(strategy)
        
    
    def calculate_po_utility(self, pool, stake_allocation): 
        pledge = pool.pledge
        m = pool.margin
        pool_stake = pool.stake
        alpha = self.model.alpha
        beta = 1/self.model.k
        r = hlp.calculate_pool_reward(pool_stake, pledge, alpha, beta)
        u_0 = r - self.cost
        utility = u_0 if u_0 <= 0 else u_0*(m + ((1-m)*stake_allocation/pool_stake))
       
        return utility
    
    '''def calculate_po_utility_multi(self, strategy): 
        utility = 0
        pledges = strategy.pledges
        alpha = strategy.allocations[self.unique_id]
        for i,pledge in enumerate(pledges):
            m = strategy.margins[i]
            pool_stake = 1 #TODO calculate pool stake 
            r = hlp.calculate_reward(pool_stake, pledge)
            u_0 = r - self.cost
            u = u_0 if u_0 <= 0 else u_0*(m + ((1-m)*alpha[i]/pool_stake))
            utility += u
        return utility'''
    
    def calculate_delegator_utility(self, pool, stake_allocation): 
        #calculate the pool's reward
        alpha = self.model.alpha
        beta = 1/self.model.k
        r = hlp.calculate_pool_reward(pool.stake, pool.pledge, alpha, beta)
        u = (1 - pool.margin) * (r - pool.cost) * stake_allocation/pool.stake
        utility = max(0, u)
        
        return utility
    
    ''' Randomly pick a new strategy and if it yields higher utility than the current one use it, else repeat '''
    def random_walk(self, max_steps = 10):
        if (max_steps == 0): #termination condition so that we don't get stuck for ever in case of (local) max
            return (False, None)
        new_strategy = self.get_random_valid_strategy()
        new_utility = self.calculate_utility(new_strategy)
        if (new_utility - self.utility > UTILITY_THRESHOLD):
            old_strategy = self.strategy
            self.strategy = new_strategy
            self.utility = new_utility
            return (True, [new_strategy.stake_allocations[i] - old_strategy.stake_allocations[i] for i in range(len(self.strategy.stake_allocations))])
        return self.random_walk(max_steps-1)

    def update_strategy(self):
        """

        :return: bool (true if strategy was changed and false if it wasn't), allocation changes (or None in case of no changes)
        """
        #strategy = self.strategy
        strategy_changed, allocation_changes  = self.random_walk()
        return strategy_changed, allocation_changes
        
    
    def get_status(self):
        print("Agent id: {}, type: {}, utility: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.type, self.utility, self.stake, self.cost))
        self.strategy.print_()
        print("\n")
        
    def delegate(self):
        pass
    
    def calculate_margin(self):
        m = 1
        return m
    
    def open_pool(self, pledge):
        m = self.calculate_margin()
        pool = Pool(owner=self.unique_id, margin=m, cost=self.cost, pledge=pledge)
        #self.model.pools[self.unique_id].append(pool) #for multipool strategies
        self.model.pools[self.unique_id] = pool
        #todo assert that len(self.model.pools[self.unique_id] == self.strategy.number_of_pools)
        
    def close_pool(self):
          self.model.pools[self.unique_id] = None
          #undelegate delegators' stake
          for agent in self.model.schedule.agents:
              agent.strategy.stake_allocations[self.unique_id] = 0
          #todo also update aggregate values