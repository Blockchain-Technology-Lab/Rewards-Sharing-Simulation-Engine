# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""
from contextlib import suppress

from mesa import Agent

import logic.helper as hlp
from logic.pool import Pool
from logic.strategy import SinglePoolStrategy

UTILITY_THRESHOLD = 0.00000001  # note: if the threshold is high then many delegation moves are ignored
IDLE_STEPS_AFTER_OPENING_POOL = 10


class Stakeholder(Agent):
    def __init__(self, unique_id, model, agent_type, stake=0, cost=0, strategy=None):
        super().__init__(unique_id, model)
        self.cost = cost  # the player's cost of running one pool
        self.stake = stake
        self.isMyopic = agent_type == 'M'
        self.idle_steps_remaining = 0
        self.new_strategy = None
        # pools field that points to the player's pools?

        if strategy is None:
            # initialise strategy to being a delegator with no allocated stake
            strategy = SinglePoolStrategy(stake_allocations=[0 for i in range(self.model.num_agents)])
        self.strategy = strategy

    # In every step the agent needs to decide what to do
    def step(self):
        if self.idle_steps_remaining > 0:
            # for players that are excluded from this round (e.g. operators who recently opened their pool)
            self.idle_steps_remaining -= 1
            return
        self.make_move()
        if self.model.player_activation_order != "Simultaneous":
            self.advance()

    # if we want players to make moves simultaneously, then we need an additional advance method
    # specifically, "step() activates the agent and stages any necessary changes, but does not apply them yet
    #                advance() then applies the changes"
    def advance(self):
        if self.new_strategy is not None:
            # The player has changed their strategy, so now they have to execute it
            self.execute_strategy()
            self.model.current_step_idle = False

    def make_move(self):
        self.update_strategy()

    def has_potential_for_pool(self):
        """
        Determine whether the player is at a good position to open a pool

        :return: bool
        """
        saturation_point = self.model.beta
        alpha = self.model.alpha

        current_pool_owners = [pool.owner for pool in self.model.pools if pool is not None]

        # Calculate the potential profits of all relevant players (current player + pool owners)
        potential_profits = {agent.unique_id:
                                 hlp.calculate_potential_profit(agent.stake, agent.cost, alpha, saturation_point)
                             for agent in self.model.schedule.agents
                             if agent.unique_id in current_pool_owners or agent.unique_id == self.unique_id}

        # If the current pools are not enough to cover the total stake of the system without getting oversaturated
        # then having a positive potential profit is a necessary and sufficient condition for a player to open a pool
        # (as it means that there is stake that is forced to remain undelegated
        # or delegated to an oversaturated pool that yields suboptimal rewards)
        if len(current_pool_owners) * saturation_point < self.model.total_stake:
            return potential_profits[self.unique_id] > 0

        # if there are enough pools to cover all players' stake without causing oversaturation,
        # then the player only opens a pool if his potential profit is higher than anyone who already owns a pool
        # (with the perspective of "stealing" the delegators from that pool)
        return any(profit < potential_profits[self.unique_id] for profit in potential_profits.values())

    def calculate_pledge(self):
        """
        Based on "perfect strategies", the players choose to allocate their entire stake as the pledge of their pool
        :return:
        """
        return self.stake

    def calculate_margin(self):
        """
        Based on "perfect strategies", the player ranks all pools (existing and hypothetical) based on their potential
        profit and chooses a margin that can keep his pool competitive
        :return: float, the margin that the player will use to open a new pool
        """
        # first calculate the potential profits of all players
        potential_profits = [
            hlp.calculate_potential_profit(agent.stake, agent.cost, self.model.alpha, self.model.beta)
            for agent in self.model.schedule.agents]

        potential_profit_ranks = hlp.calculate_ranks(potential_profits)
        k = self.model.k
        # fails if n < k -> todo make sure that n > k always
        k_rank_index = potential_profit_ranks.index(k)  # find the player who is ranked at the kth position

        margin = 1 - (potential_profits[k_rank_index] / potential_profits[self.unique_id]) \
            if potential_profit_ranks[self.unique_id] < k else 0
        return margin

    def find_operator_move(self):
        pledge = self.calculate_pledge()
        margin = self.calculate_margin()

        allocations = [0 for i in range(len(self.model.schedule.agents))]
        allocations[self.unique_id] = pledge

        new_strategy = SinglePoolStrategy(pledge, margin, allocations, is_pool_operator=True)
        new_utility = self.calculate_utility(new_strategy)

        return new_utility, new_strategy

    def find_delegation_move_random(self, current_utility, max_steps=100):
        """
        Choose a delegation move using a random walk
        :param current_utility:
        :param max_steps:
        :return: a delegation strategy that yields higher utility than the current one
        OR the last strategy that was examined (if none of the examined strategies had higher utility than the current)
        """
        for i in range(max_steps):
            strategy = self.strategy.create_random_delegator_strategy(self.model.pools, self.unique_id, self.stake)
            utility = self.calculate_utility(strategy)

            if utility - current_utility > UTILITY_THRESHOLD:
                break
        return utility, strategy

    def find_delegation_move_desirability(self):
        """
        Choose a delegation move based on the desirability of the existing pools
        :return:
        """
        saturation_point = self.model.beta
        stake_to_delegate = self.stake

        pools = self.model.pools.copy()
        desirabilities = {pool.owner: pool.desirability for pool in pools
                          if pool is not None and pool.owner != self.unique_id}
        allocations = [0 for _ in range(len(pools))]

        # Delegate the stake to the most pools with the highest desirability, as long as they're not oversaturated
        for pool_index in sorted(desirabilities, reverse=True):
            if stake_to_delegate == 0:
                break
            if pools[pool_index].stake < saturation_point:
                stake_to_saturation = saturation_point - pools[pool_index].stake
                allocations[pool_index] = min(stake_to_delegate, stake_to_saturation)
                stake_to_delegate -= allocations[pool_index]

        strategy = SinglePoolStrategy(stake_allocations=allocations, is_pool_operator=False)
        utility = self.calculate_utility(strategy)

        return utility, strategy

    def find_delegation_move_potential_profit(self):
        """
        Choose a delegation move based on the potential profit of the existing pools
        :return:
        """
        saturation_point = self.model.beta
        alpha = self.model.alpha

        pools = self.model.pools.copy()
        pool_owners = [pool.owner for pool in pools if pool is not None]
        with suppress(ValueError):
            # remove the current player from the list in case he's an SPO
            pool_owners.remove(self.unique_id)

        # Calculate the potential profits of all current pools
        potential_profits = {agent.unique_id:
                                 hlp.calculate_potential_profit(agent.stake, agent.cost, alpha, saturation_point)
                             for agent in self.model.schedule.agents
                             if agent.unique_id in pool_owners}

        stake_to_delegate = self.stake
        allocations = [0 for pool in range(len(pools))]

        # Delegate the stake to the most (potentially) profitable pools as long as they're not oversaturated
        for pool_index in sorted(potential_profits, reverse=True):
            if stake_to_delegate == 0:
                break
            if pools[pool_index].stake < saturation_point:
                allocations[pool_index] = min(stake_to_delegate, saturation_point - pools[pool_index].stake)
                stake_to_delegate -= allocations[pool_index]

        strategy = SinglePoolStrategy(stake_allocations=allocations, is_pool_operator=False)
        utility = self.calculate_utility(strategy)

        return utility, strategy

    def update_strategy(self):
        """

        :return: bool, float: whether the player's strategy changed
        and the potential utility from the new strategy
        """

        # Recalculate utility because pool formation may have changed since last calculation
        current_utility = self.calculate_utility(self.strategy)

        possible_moves = {"current": (
        current_utility + UTILITY_THRESHOLD, self.strategy)}  # every dict value is a tuple of utility, strategy

        if not self.strategy.is_pool_operator:
            # For players who don't already have pools check if they should open one

            if self.has_potential_for_pool():
                # Player is considering opening a pool, so he has to find the most suitable pool params
                # and calculate the potential utility of operating a pool with these params
                possible_moves["operator"] = self.find_operator_move()

        # todo also consider option of changing margin and pledge or only consider "perfect strategies"?

        # For all players (current pool owners, prospective pool owners
        # and players who don't even consider running a pool)
        # find a possible delegation strategy and calculate its potential utility
        possible_moves["delegator"] = self.find_delegation_move_desirability()

        # compare the above with the utility of the current strategy and pick one of the 3
        max_utility_option = max(possible_moves,
                                 key=lambda key: possible_moves[key][0])  # todo in case of tie choose easiest strategy

        self.new_strategy = None if max_utility_option == "current" else possible_moves[max_utility_option][1]

    def calculate_utility(self, strategy):
        utility = 0
        # todo looping through all possible allocations is not efficient (maybe use hashing)
        for i, allocation in enumerate(strategy.stake_allocations):
            if not isinstance(allocation, list):  # in case of multi-pool strategy
                allocation = [allocation]
            for j, a in enumerate(allocation):
                if a > 0:
                    # player has allocated stake to this pool
                    pool = self.model.pools[i]
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

    def calculate_delegator_utility(self, pool, stake_allocation):
        # calculate the pool's reward
        alpha = self.model.alpha
        beta = self.model.beta
        pool_stake = pool.stake if self.isMyopic else pool.stake_NM
        if pool_stake + stake_allocation > beta:
            pool_stake += stake_allocation
        # maybe add reward as a pool field?
        r = hlp.calculate_pool_reward(pool_stake, pool.pledge, alpha, beta)
        q = stake_allocation / pool_stake
        m_factor = (1 - pool.margin) * q
        u_0 = (r - pool.cost)
        u = m_factor * u_0
        utility = max(0, u)

        return utility

    def execute_strategy(self):
        """
        Execute the player's current strategy
        :return: void

        """
        current_pools = self.model.pools
        # since players are allowed to make moves simultaneously, we need to check for validity
        # when we apply their chosen strategies (e.g. delegating to a pool that just closed)
        move_still_valid = all([(current_pools[i] is not None or allocation == 0) for i, allocation
                                in enumerate(self.new_strategy.stake_allocations) if i != self.unique_id])
        # for now if the move is not 100% valid, we discard it completely (alternatively we could only discard the invalid part)
        if not move_still_valid:
            self.new_strategy = None
            return

        allocation_changes = [self.new_strategy.stake_allocations[i] - self.strategy.stake_allocations[i] for i in
                              range(len(self.strategy.stake_allocations))]
        own_pool_changes = {'margin': self.new_strategy.margin - self.strategy.margin,
                            'pledge': self.new_strategy.pledge - self.strategy.pledge}

        self.strategy = self.new_strategy
        self.new_strategy = None  # maybe redundant?

        # first deal with possible margin or pledge changes
        if current_pools[self.unique_id] is not None:  # replace with current_pools but make sure that it works
            self.model.pools[self.unique_id].margin += own_pool_changes['margin']
            self.model.pools[self.unique_id].margin += own_pool_changes['pledge']

        # todo looping through all possible allocations is not efficient, do sth else
        for i, change in enumerate(allocation_changes):
            if change != 0:
                # there has been one of two possible changes with regards to this pool:
                # the player added to or removed stake from it
                if i == self.unique_id:
                    # special case of own pool, where we need to consider moves that open / close pools
                    if allocation_changes[i] == self.strategy.stake_allocations[i]:
                        # means that the pool needs to be created now
                        self.open_pool(pledge=allocation_changes[self.unique_id], margin=self.strategy.margin)
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
        self.model.pools[self.unique_id] = pool
        self.idle_steps_remaining = IDLE_STEPS_AFTER_OPENING_POOL

    def close_pool(self):
        self.model.pools[self.unique_id] = None
        # Undelegate delegators' stake
        for agent in self.model.schedule.agents:
            agent.strategy.stake_allocations[self.unique_id] = 0

    def get_status(self):
        print("Agent id: {}, is myopic: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.isMyopic, self.stake, self.cost))
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
