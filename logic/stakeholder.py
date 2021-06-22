# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""

from mesa import Agent
import random

from logic import helper as hlp
from logic.pool import Pool
from logic.strategy import SinglePoolStrategy, MultiPoolStrategy

UTILITY_THRESHOLD = 0.0001 # for defining an inertial equilibrium

#todo when using the sim/model object, stick to calling methods instead of accessing fields directly

class Stakeholder(Agent):
    def __init__(self, unique_id, model, agent_type, stake=0, cost=0, utility=0, canSplitPools=False):
        super().__init__(unique_id, model)
        self.cost = cost #the cost of running one pool (assume that running n pools will cost n times as much)
        self.stake = stake
        self.utility = utility
        self.isMyopic = agent_type == 'M'
        self.canSplitPools = canSplitPools
        # pools field that points to the player's pools?
        
        self.initialize_strategy()
    
    # In every step the agent needs to decide what to do
    def step(self):
        strategy_changed, allocation_changes = self.update_strategy()
        #print("Allocation changes: {}".format(allocation_changes))
        if (strategy_changed):
            # the player has changed their strategy, so now they have to execute it 
            self.execute_strategy(allocation_changes)
            self.model.current_step_idle = False
        #self.get_status()
        #print('----------------------')
        
        
    def initialize_strategy(self):
        if self.canSplitPools:
            strategy = MultiPoolStrategy()
            strategy.stake_allocations = [[0] for i in range(self.model.num_agents)]
        else:
            strategy = SinglePoolStrategy()
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
        sim = self.model
        return self.strategy.create_random_valid_strategy(sim.pools, self.unique_id, self.stake)
    
    def calculate_utility(self, strategy):
        utility = 0
        for i, allocation in enumerate(strategy.stake_allocations):
            if not isinstance(allocation, list): # in case of multi-pool strategy
                allocation = [allocation]
            for j,a in enumerate(allocation):
                if (a > 0):
                    # player has allocated stake to this pool
                    pool = self.model.pools[i] #todo fix for pool splitting
                    if (i==self.unique_id):
                        #calculate pool owner utility
                        if (pool is None):
                            # player hasn't created their pool yet, so we calculate the utility of a hypothetical pool
                            pool = Pool(margin=strategy.margin, cost=self.cost, pledge=strategy.pledge, owner=self.unique_id)
                            # calculate non-myopic stake for hypothetical pool
                            hlp.calculate_pool_stake_NM(pool,
                                                        self.model.pools,
                                                        self.unique_id,
                                                        self.model.alpha,
                                                        1/self.model.k
                                                        )
                        utility += self.calculate_po_utility(pool, a)
                    else:
                        # calculate delegator utility
                        try:
                            utility += self.calculate_delegator_utility(pool, a)
                        except ZeroDivisionError: #todo change exception handling
                            print("POOL IS NOOOOOONE") #should never be none when choosing strategy non-randomly

        
        return utility
        
    
    def calculate_po_utility(self, pool, stake_allocation):
        pledge = pool.pledge
        m = pool.margin
        pool_stake = pool.stake if self.isMyopic else pool.stake_NM
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
        pool_stake = pool.stake if self.isMyopic else pool.stake_NM
        r = hlp.calculate_pool_reward(pool_stake, pool.pledge, alpha, beta)
        u = (1 - pool.margin) * (r - pool.cost) * stake_allocation/pool_stake
        utility = max(0, u)
        
        return utility

    def random_walk(self, max_steps = 10): #todo experiment with max_steps values
        """
        Randomly pick a new strategy and if it yields higher utility than the current one, use it else repeat
        :param max_steps: if no better strategy is found after trying max_steps times, then keep the old strategy
        :return: bool (true if strategy was changed and false if it wasn't), allocation changes (or None in case of no changes)
        """
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
        strategy_changed, allocation_changes = self.random_walk()
        return strategy_changed, allocation_changes
        
    def delegate(self):
        pass

    #todo only use to calculate other players' margins in a non-myopic way (to confirm current player's margin)
    def calculate_margin(self, potential_pool_reward, potential_pool_stake):
        """
        Calculate a player's optimal margin, based on the formula suggested in the paper
        :return: float that describes the margin the player will use when opening a pool
        #todo maybe return negative margin (e.g. -1) to signify that the pool can't be profitable (e.g. if r < c)?
        """
        q = self.stake / potential_pool_stake if potential_pool_stake else 0
        denom = (potential_pool_reward - self.cost) * (1 - q)
        m = (self.utility - (potential_pool_reward - self.cost) * q) / denom if denom else 0
        return m
    
    def open_pool(self, pledge):
        pool = Pool(owner=self.unique_id, cost=self.cost, pledge=pledge)

        alpha = self.model.alpha
        beta = 1 / self.model.k

        # calculate optimal margin
        potential_pool_stake = max(pool.stake, beta)
        potential_pool_reward = hlp.calculate_pool_reward(potential_pool_stake, pledge, alpha, beta)
        m = self.calculate_margin(potential_pool_reward, potential_pool_stake)
        pool.margin = m #todo either add this margin to the player's strategy or use the margin that is already there

        # calculate non-myopic stake
        hlp.calculate_pool_stake_NM(pool, self.model.pools, self.unique_id, alpha, beta)
        #todo update for pool splitting
        #self.model.pools[self.unique_id].append(pool) #for multipool strategies
        self.model.pools[self.unique_id] = pool
        #todo assert that len(self.model.pools[self.unique_id] == self.strategy.number_of_pools)
        
    def close_pool(self):
          self.model.pools[self.unique_id] = None
          #undelegate delegators' stake
          for agent in self.model.schedule.agents:
              agent.strategy.stake_allocations[self.unique_id] = 0
          #todo also update aggregate values


    def get_status(self):
        print("Agent id: {}, is myopic: {}, utility: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.isMyopic, self.utility, self.stake, self.cost))
        self.strategy.print_()
        print("\n")