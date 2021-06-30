# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""

from mesa import Agent

import helper as hlp
from pool import Pool
from strategy import SinglePoolStrategy, MultiPoolStrategy

UTILITY_THRESHOLD = 0.0001  # note: if the threshold is high then many delegation moves are ignored
IDLE_STEPS_AFTER_OPENING_POOL = 10


class Stakeholder(Agent):
    def __init__(self, unique_id, model, agent_type, stake=0, cost=0, utility=0, strategy=None, can_split_pools=False):
        super().__init__(unique_id, model)
        # the player's cost of running one pool (for now assume that running n pools will cost n times as much)
        self.cost = cost
        self.stake = stake
        self.utility = utility
        self.isMyopic = agent_type == 'M'
        self.canSplitPools = can_split_pools
        self.idle_steps_remaining = 0
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
        if self.idle_steps_remaining > 0:
            # for players that are excluded from this round (e.g. operators who recently opened their pool)
            self.idle_steps_remaining -= 1
            return
        strategy_changed, new_utility = self.update_strategy()
        if strategy_changed:
            # The player has changed their strategy, so now they have to execute it
            self.execute_strategy(new_utility)
            self.model.current_step_idle = False
        # self.get_status()
        # print('----------------------')

    # todo if we want players to make moves simultaneously, then we need an additional advance method
    # specifically, "step() activates the agent and stages any necessary changes, but does not apply them yet
    #                advance() then applies the changes"
    def advance(self):
        pass

    def update_strategy(self):
        """

        :return: bool (true if strategy was changed and false if it wasn't), allocation changes
                                                            (or None in case of no changes)
        """
        strategy_changed, new_utility = self.not_so_random_walk()
        return strategy_changed, new_utility

    # todo maybe move to helper (with extra arguments)
    def should_open_pool(self):
        saturation_point = self.model.beta

        current_pool_owners = [pool.owner for pool in self.model.pools if pool is not None]  # assumes no pool splitting

        # Calculate the potential profits of all relevant players (current player + pool owners)
        potential_profits = {agent.unique_id: hlp.calculate_potential_profit(agent.stake, agent.cost, self.model.alpha, self.model.beta)
                             for agent in self.model.schedule.agents if agent.unique_id in current_pool_owners or agent.unique_id == self.unique_id}

        # if there are no pools at the current stage, it suffices to have a positive potential profit
        # what if the "best" player opens up a pool first? then no one could have a better potential profit
        # and therefore would not create a pool? therefore if there are less than k pools, create one anyway (as long as the potential profit is positive)

        if len(current_pool_owners) * saturation_point < self.model.total_stake:
            # This means that there is stake that is forced to remain undelegated or delegated to an oversaturated pool
            # Could also simply check if the current pools are less than k in number but this approach is more general
            # (in case we want to decouple beta from k)
            return potential_profits[self.unique_id] > 0

        # if there are enough pools to cover all players' stake without causing oversaturation,
        # then the player only opens a pool if his potential profit is higher than anyone who already owns a pool
        # (with the perspective of "stealing" the delegators from that pool)

        return any(profit < potential_profits[self.unique_id] for profit in potential_profits.values())

    def should_keep_pool(self):
        pass

    # consider an operator AND delegator strategy for EVERYONE, then decide between the 2 and then as usual
    # basically give the chance to everyone to create a pool
    def not_so_random_walk(self):

        if not self.strategy.is_pool_operator:
            # for players who don't already have pools check if they should open one

            if self.should_open_pool():
                # find the most suitable pool params and calculate the potential utility of operating a pool with these params

                # first calculate the potential profits of all players
                potential_profits = [
                    hlp.calculate_potential_profit(agent.stake, agent.cost, self.model.alpha, self.model.beta)
                    for agent in self.model.schedule.agents]

                potential_profit_ranks = hlp.calculate_ranks(potential_profits)

                k = self.model.k
                k_rank_index = potential_profit_ranks.index(k)  # find the player who is ranked at the kth position
                margin = 1 - (potential_profits[k_rank_index] / potential_profits[self.unique_id]) \
                    if potential_profit_ranks[self.unique_id] < k else 0
                pledge = self.stake
                allocations = [0 for i in range(len(self.model.schedule.agents))]
                allocations[self.unique_id] = pledge

                new_strategy = SinglePoolStrategy(pledge, margin, allocations, is_pool_operator=True) # todo update for pool splitting
                new_utility = self.calculate_utility(new_strategy)
                if new_utility - self.utility > UTILITY_THRESHOLD:
                    self.new_strategy = new_strategy
                    self.idle_steps_remaining = IDLE_STEPS_AFTER_OPENING_POOL
                    return True, new_utility #todo calculate utility earlier and only open pool if new utility is higher?

        # if the player doesn't want to open a pool (or already has one), then find a good delegation strategy and calculate its potential utility
        # for now random choice of delegation todo change
        max_steps = 100
        for i in range(max_steps):
            new_strategy = self.strategy.create_random_delegator_strategy(self.model.pools, self.unique_id, self.stake)
            self.utility = self.calculate_utility(
                self.strategy)  # recalculate utility because pool formation may have changed since last calculation
            new_utility = self.calculate_utility(new_strategy)

            if new_utility - self.utility > UTILITY_THRESHOLD:
                self.new_strategy = new_strategy
                return True, new_utility

        return False, None  # no strategy found that is better than the current one

        # compare the above with the utility of the current strategy and pick one of the 3 #todo also try that approach

    def get_perfect_strategy(self):
        pass

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
        q = stake_allocation / pool_stake
        u_0 = r - self.cost
        m_factor = m + ((1 - m) * q)
        utility = u_0 if u_0 <= 0 else u_0 * m_factor

        return utility

    # todo why is delegator utility (almost) as good as operator utility? make sure that calculations are correct
    # answer: because we don't count the added stake so his proportion appears to be bigger
    def calculate_delegator_utility(self, pool, stake_allocation):
        # calculate the pool's reward
        alpha = self.model.alpha
        beta = self.model.beta
        pool_stake = pool.stake if self.isMyopic else pool.stake_NM
        pool_stake += stake_allocation  # todo decide
        # todo maybe add reward as a pool field?
        r = hlp.calculate_pool_reward(pool_stake, pool.pledge, alpha, beta)
        q = stake_allocation / pool_stake
        m_factor = (1 - pool.margin) * q
        u_0 = (r - pool.cost)
        u = m_factor * u_0
        utility = max(0, u)

        return utility

    # todo instead of passing allocation changes, stash new strategy as a field of stakeholder and calculate the changes during execution
    # (especially useful approach for simultaneous moves, where the changes need to be staged and then committed at a later step)
    def execute_strategy(self, new_utility):
        """
        Execute the player's current strategy
        :return: void

        """

        allocation_changes = [self.new_strategy.stake_allocations[i] - self.strategy.stake_allocations[i] for i in
                              range(len(self.strategy.stake_allocations))]
        own_pool_changes = {'margin': self.new_strategy.margin - self.strategy.margin,
                            'pledge': self.new_strategy.pledge - self.strategy.pledge}

        self.utility = new_utility
        self.strategy = self.new_strategy
        self.new_strategy = None

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
            return utility
            
    def random_walk(self, max_steps=10):  # todo experiment with max_steps values
        """
        Randomly pick a new strategy and if it yields higher utility than the current one, use it else repeat
        :param max_steps: if no better strategy is found after trying max_steps times, then keep the old strategy
        :return: bool (true if strategy was changed and false if it wasn't), allocation changes
                                                                            (or None in case of no changes)
        """
        if max_steps == 0:  # termination condition so that we don't get stuck for ever in case of (local) max
            return False, None
        new_strategy = self.get_random_valid_strategy(self.strategy.is_pool_operator)
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
                # todo if margin m doesn't have potential then obviously margin m' > m won't have potential so don't even check
            if has_potential:
                self.new_strategy = new_strategy
                return True, new_utility
        return self.random_walk(max_steps - 1)  # todo maybe recursion is not efficient here?
        
    def get_random_valid_strategy(self, is_pool_operator):
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
        #strategy_type = random.randint(0, 1)  # 0 = operate pool, 1 = delegate todo could be boolean
        if is_pool_operator:
            weights = [0.95, 0.05]
        else:
            weights = [0.5, 0.5]
        strategy_type = random.choices(population=[0, 1], weights=weights)[0]
        if strategy_type == 0:
            return self.strategy.create_random_operator_strategy(sim.pools, self.unique_id, self.stake)
        else:
            return self.strategy.create_random_delegator_strategy(sim.pools, self.unique_id, self.stake)
 
            
    '''
