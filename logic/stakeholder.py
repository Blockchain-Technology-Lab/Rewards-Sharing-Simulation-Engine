# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""
import random

from mesa import Agent

from logic import helper as hlp
from logic.pool import Pool
from logic.strategy import SinglePoolStrategy, MultiPoolStrategy

UTILITY_THRESHOLD = 0.00000001  # note: if the threshold is high then many delegation moves are ignored


class Stakeholder(Agent):
    def __init__(self, unique_id, model, agent_type, stake=0, cost=0, utility=0, strategy=None, can_split_pools=False):
        super().__init__(unique_id, model)
        # the player's cost of running one pool (for now assume that running n pools will cost n times as much)
        self.cost = cost
        self.stake = stake
        self.utility = utility
        self.isMyopic = agent_type == 'M'
        self.canSplitPools = can_split_pools
        # pools field that points to the player's pools?

        if strategy is None:
            # initialise strategy to being a delegator with no allocated stake
            self.initialize_strategy()
        else:
            self.strategy = strategy

    def initialize_strategy(self):
        if self.canSplitPools:
            strategy = MultiPoolStrategy(stake_allocations=[[0] for i in range(self.model.num_agents)])
        else:
            strategy = SinglePoolStrategy(stake_allocations=[0 for i in range(self.model.num_agents)])

        self.strategy = strategy

    # In every step the agent needs to decide what to do
    def step(self):
        strategy_changed, allocation_changes, own_pool_changes = self.update_strategy()
        if strategy_changed:
            # The player has changed their strategy, so now they have to execute it
            self.execute_strategy(allocation_changes, own_pool_changes)
            self.model.current_step_idle = False
        # self.get_status()
        # print('----------------------')

    def update_strategy(self):
        """

        :return: bool (true if strategy was changed and false if it wasn't), allocation changes
                                                            (or None in case of no changes)
        """
        strategy_changed, allocation_changes, own_pool_changes = self.random_walk()
        return strategy_changed, allocation_changes, own_pool_changes

    def random_walk(self, max_steps=100):  # todo experiment with max_steps values
        """
        Randomly pick a new strategy and if it yields higher utility than the current one, use it else repeat
        :param max_steps: if no better strategy is found after trying max_steps times, then keep the old strategy
        :return: bool (true if strategy was changed and false if it wasn't), allocation changes
                                                                            (or None in case of no changes)
        """
        if max_steps == 0:  # termination condition so that we don't get stuck for ever in case of (local) max
            return False, None, None
        new_strategy = self.get_random_valid_strategy()
        self.utility = self.calculate_utility(
            self.strategy)  # recalculate utility because pool formation may have changed since last calculation
        new_utility = self.calculate_utility(new_strategy)
        # The player will only consider switching strategies if the new strategy
        # yields sufficiently higher utility than the old one
        if new_utility - self.utility > UTILITY_THRESHOLD:
            # If the strategy includes operating a pool, the player has to consider if the pool can get into the top k
            # and get saturated (if the player is non-myopic)
            # todo maybe do that before calculating utility? check which one is more expensive
            has_potential = True
            if new_strategy.is_pool_operator:
                has_potential = hlp.check_pool_potential(new_strategy, self.model, self.unique_id)
            if has_potential:
                old_strategy = self.strategy
                self.strategy = new_strategy
                self.utility = new_utility
                allocation_changes = [new_strategy.stake_allocations[i] - old_strategy.stake_allocations[i] for i in
                                      range(len(self.strategy.stake_allocations))]
                own_pool_changes = {'margin': new_strategy.margin - old_strategy.margin,
                                    'pledge': new_strategy.pledge - old_strategy.pledge}
                return True, allocation_changes, own_pool_changes
        return self.random_walk(max_steps - 1)  # todo maybe recursion is not efficient here?

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

        # Flip a coin to decide whether the random strategy will be about operating a pool
        # or about delegating to one or more pools
        strategy_type = random.randint(0, 1)  # 0 = operate pool, 1 = delegate todo could be boolean
        # todo add weights to the different strategy types to account for inertia (it is more likely that a pool owner
        #  will stick to being a pool owner and a delegator will remain a delegator) and time cost of being an operator
        if strategy_type == 0:
            return self.strategy.create_random_operator_strategy(sim.pools, self.unique_id, self.stake)
        else:
            return self.strategy.create_random_delegator_strategy(sim.pools, self.unique_id, self.stake)

    def calculate_utility(self, strategy):
        utility = 0
        # todo looping through all possible allocations is not efficient (maybe use hashing)
        for i, allocation in enumerate(strategy.stake_allocations):
            if not isinstance(allocation, list):  # in case of multi-pool strategy
                allocation = [allocation]
            for j, a in enumerate(allocation):
                if a > 0:
                    # player has allocated stake to this pool
                    pool = self.model.pools[i]  # todo fix for pool splitting
                    if i == self.unique_id:
                        # calculate pool owner utility
                        if pool is None:
                            # player hasn't created their pool yet, so we calculate the utility of a hypothetical pool
                            pool = Pool(margin=strategy.margin, cost=self.cost, pledge=strategy.pledge,
                                        owner=self.unique_id)
                            # calculate non-myopic stake for hypothetical pool
                            hlp.calculate_pool_stake_NM(pool,
                                                        self.model.pools,
                                                        self.unique_id,
                                                        self.model.alpha,
                                                        self.model.beta,
                                                        self.model.k
                                                        )
                        utility += self.calculate_operator_utility(pool, a)
                    else:
                        # calculate delegator utility
                        try:
                            utility += self.calculate_delegator_utility(pool, a)
                        except AttributeError:  # todo refine exception handling
                            print("POOL IS NOOOOOONE")  # should never be none when choosing strategy non-randomly

        return utility

    def calculate_operator_utility(self, pool, stake_allocation):
        pledge = pool.pledge
        m = pool.margin
        pool_stake = pool.stake if self.isMyopic else pool.stake_NM
        alpha = self.model.alpha
        beta = self.model.beta
        r = hlp.calculate_pool_reward(pool_stake, pledge, alpha, beta)
        u_0 = r - self.cost
        utility = u_0 if u_0 <= 0 else u_0 * (m + ((1 - m) * stake_allocation / pool_stake))

        return utility

    def calculate_delegator_utility(self, pool, stake_allocation):
        # calculate the pool's reward
        alpha = self.model.alpha
        beta = self.model.beta
        pool_stake = pool.stake if self.isMyopic else pool.stake_NM
        # todo maybe add reward as a pool field?
        r = hlp.calculate_pool_reward(pool_stake, pool.pledge, alpha, beta)
        u = (1 - pool.margin) * (r - pool.cost) * stake_allocation / pool_stake
        utility = max(0, u)

        return utility

    def execute_strategy(self, allocation_changes, own_pool_changes):
        """
        Execute the player's current strategy
        :param allocation_changes:
        :param pool_changes:
        :return: void

        """
        # first deal with possible margin or pledge changes
        if self.model.pools[self.unique_id] is not None:
            self.model.pools[self.unique_id].margin += own_pool_changes['margin']
            self.model.pools[self.unique_id].margin += own_pool_changes['pledge']

        # todo looping through all possible allocations is not efficient, do sth else
        for i, change in enumerate(allocation_changes):
            if change != 0:
                # there has been one of two possible changes with regards to this pool: the player added to or removed stake from it
                if i == self.unique_id:
                    # special case of own pool, where we need to consider moves that open / close pools
                    if allocation_changes[i] == self.strategy.stake_allocations[i]:
                        # means that the pool needs to be created now
                        self.open_pool(pledge=allocation_changes[self.unique_id],
                                       margin=self.strategy.margin)  # todo update for pool splitting
                        continue
                    elif self.strategy.stake_allocations[i] == 0:
                        # means that the pool needs to be destroyed, since it has been abandoned by its owner
                        self.close_pool()
                        continue
                self.model.pools[i].update_stake(change)  # add or subtract the relevant stake from the pool
                # Recalculate the pool's non-myopic stake after every stake update todo maybe it's not efficient
                hlp.calculate_pool_stake_NM(None,
                                            self.model.pools,
                                            i,
                                            self.model.alpha,
                                            self.model.beta,
                                            self.model.k
                                            )

    def open_pool(self, pledge, margin):
        pool = Pool(owner=self.unique_id, cost=self.cost, pledge=pledge, margin=margin)

        alpha = self.model.alpha
        beta = self.model.beta
        k = self.model.k

        # calculate non-myopic stake
        hlp.calculate_pool_stake_NM(pool, self.model.pools, self.unique_id, alpha, beta, k)
        # todo update for pool splitting and assert that len(self.model.pools[self.unique_id] == self.strategy.number_of_pools)
        # self.model.pools[self.unique_id].append(pool) #for multipool strategies
        self.model.pools[self.unique_id] = pool

    def close_pool(self):
        self.model.pools[self.unique_id] = None
        # Undelegate delegators' stake
        for agent in self.model.schedule.agents:
            agent.strategy.stake_allocations[self.unique_id] = 0  # todo update to accommodate pool splitting
        # todo also update aggregate values

    def get_status(self):
        print("Agent id: {}, is myopic: {}, utility: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.isMyopic, self.utility, self.stake, self.cost))
        self.strategy.print_()
        print("\n")

    ''' Currently not used methods below (but could be useful in the future)

    # todo only use to calculate other players' margins in a non-myopic way (to confirm current player's margin)
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

    def delegate(self):
        pass

    def calculate_po_utility_multi(self, strategy): 
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
