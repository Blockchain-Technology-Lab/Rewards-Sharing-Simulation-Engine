# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""
from collections import defaultdict

from mesa import Agent
from copy import deepcopy
import operator

import logic.helper as hlp
from logic.pool import Pool
from logic.strategy import Strategy
from logic.strategy import STARTING_MARGIN
from logic.strategy import MARGIN_INCREMENT
from logic.custom_exceptions import PoolNotFoundError, NonPositiveAllocationError


class Stakeholder(Agent):
    __slots__ = ['unique_id', 'model', 'stake', 'cost', 'isMyopic', 'idle_steps_remaining', 'strategy',
                 'new_strategy']  # note that because the parent class doesn't use slots, the effect is not what it could otherwise have been

    def __init__(self, unique_id, model, stake=0, is_myopic=False,
                 cost=0, strategy=None):
        super().__init__(unique_id, model)
        self.cost = cost  # the player's individual cost of running one or more pools
        self.stake = stake
        self.isMyopic = is_myopic
        self.idle_steps_remaining = 0
        self.new_strategy = None

        if strategy is None:
            # Initialise strategy to being a delegator with no allocated stake
            strategy = Strategy()
        self.strategy = strategy

    # In every step the agent needs to decide what to do
    def step(self):
        if self.idle_steps_remaining > 0:
            # For players that are excluded from this round (e.g. operators who recently opened their pool)
            self.idle_steps_remaining -= 1
            return
        self.make_move()
        if self.model.player_activation_order != "Simultaneous":
            self.advance()

    # When players make moves simultaneously, "step() activates the agent and stages any necessary changes,
    # but does not apply them yet, and advance() then applies the changes"
    def advance(self):
        if self.new_strategy is not None:
            # The player has changed their strategy, so now they have to execute it
            self.execute_strategy()
            self.model.current_step_idle = False

    def make_move(self):
        self.update_strategy()

    def update_strategy(self):
        current_utility = self.calculate_utility(self.strategy)
        # hold the player's potential moves in a dict, where the values are tuples of (utility, strategy)
        possible_moves = {"current": (
            current_utility + self.model.utility_threshold, self.strategy)}

        # For all players, find a possible delegation strategy and calculate its potential utility
        delegator_strategy = self.find_delegation_move_desirability()
        delegator_utility = self.calculate_utility(delegator_strategy)
        possible_moves["delegator"] = delegator_utility, delegator_strategy

        if self.strategy.is_pool_operator or self.has_potential_for_pool():
            # Player is considering opening a pool, so he has to find the most suitable pool params
            # and calculate the potential utility of operating a pool with these params
            operator_strategies = self.find_operator_moves()
            max_operator_utility = 0
            max_operator_strategy = None
            for operator_strategy in operator_strategies.values():
                operator_utility = self.calculate_utility(operator_strategy)
                if operator_utility > max_operator_utility:
                    max_operator_utility = operator_utility
                    max_operator_strategy = operator_strategy
            possible_moves["operator"] = max_operator_utility, max_operator_strategy

        # Compare the above with the utility of the current strategy and pick one of the 3
        # in case of a tie, the max function picks the element with the lowest index, so we have strategically ordered
        # them earlier so that the "easiest" move is preferred ( current -> delegator -> operator)
        max_utility_option = max(possible_moves,
                                 key=lambda key: possible_moves[key][0])

        if "operator" in possible_moves.keys() and max_utility_option != "operator":
            # Discard the pool ids that were used for the hypothetical operator move
            old_owned_pools = set(self.strategy.owned_pools.keys())
            hypothetical_owned_pools = set(operator_strategy.owned_pools.keys())
            self.model.rewind_pool_id_seq(step=len(hypothetical_owned_pools - old_owned_pools))

        self.new_strategy = None if max_utility_option == "current" else possible_moves[max_utility_option][1]

    def calculate_utility(self, strategy):
        utility = 0
        # Calculate utility of operating pools
        if strategy.is_pool_operator:
            utility += self.calculate_operator_utility_by_strategy(strategy)

        pools = self.model.pools
        # Calculate utility of delegating to other pools
        for pool_id, a in strategy.stake_allocations.items():
            if a <= 0:
                continue
                # raise NonPositiveAllocationError(
                #   "Player tried to allocate zero or less stake to a pool.")  # todo replace with ValueError?
            pool = pools[pool_id]
            if pool is None:
                raise PoolNotFoundError("Player considered delegating to a non-existing pool!")
            else:
                utility += self.calculate_delegator_utility(pool, a)
        return utility

    def calculate_operator_utility_by_strategy(self, strategy):
        utility = 0
        potential_pools = strategy.owned_pools
        all_pools = self.model.pools | potential_pools
        for pool in potential_pools.values():
            utility += self.calculate_operator_utility_by_pool(pool, all_pools)
        return utility

    def calculate_operator_utility_by_pool(self, pool, all_pools):
        alpha = self.model.alpha
        beta = self.model.beta
        pledge = pool.pledge
        stake_allocation = pledge  # todo change if we allow pool owners to allocate stake to their pool separate to the pledge
        m = pool.margin
        '''pool_stake = pool.stake if self.isMyopic else hlp.calculate_pool_stake_NM(pool.id,
                                                                                  all_pools,
                                                                                  beta,
                                                                                  self.model.k
                                                                                  )'''  # assuming there is no myopic play for pool owners
        pool_stake = hlp.calculate_pool_stake_NM(pool.id,
                                                 all_pools,
                                                 beta,
                                                 self.model.k
                                                 )
        r = hlp.calculate_pool_reward(pool_stake, pledge, alpha, beta)
        q = stake_allocation / pool_stake
        u_0 = r - self.cost
        m_factor = m + ((1 - m) * q)
        return u_0 if u_0 <= 0 else u_0 * m_factor

    def calculate_delegator_utility(self, pool, stake_allocation):
        alpha = self.model.alpha
        beta = self.model.beta
        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id]
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        non_myopic_stake = max(hlp.calculate_pool_stake_NM(pool.id,
                                                           self.model.pools,
                                                           self.model.beta,
                                                           self.model.k
                                                           ),
                               current_stake)
        pool_stake = current_stake if self.isMyopic else non_myopic_stake
        r = hlp.calculate_pool_reward(pool_stake, pool.pledge, alpha, beta)
        q = stake_allocation / pool_stake
        m_factor = (1 - pool.margin) * q
        u_0 = (r - pool.cost)
        u = m_factor * u_0
        utility = max(0, u)
        return utility

    # todo how does a myopic player decide whether to open a pool or not?
    # todo if we use potential profit for that, then when k pools remain (and they are the k with the highest potential profit),
    # then they can do whatever they want (e.g. increase margins arbitrarily) and still no one else would open a pool, which makes no sense
    def has_potential_for_pool(self):
        """
        Determine whether the player is at a good position to open a pool, using the following rules:
        If the current pools are not enough to cover the total stake of the system without getting oversaturated
        then having a positive potential profit is a necessary and sufficient condition for a player to open a pool
        (as it means that there is stake that is forced to remain undelegated
        or delegated to an oversaturated pool that yields suboptimal rewards)

        If there are enough pools to cover all players' stake without causing oversaturation,
        then the player only opens a pool if his potential profit is higher than anyone who already owns a pool
        (with the perspective of "stealing" the delegators from that pool)

        :return: bool
        """
        saturation_point = self.model.beta
        alpha = self.model.alpha
        current_pools = self.model.get_pools_list()

        potential_profit = hlp.calculate_potential_profit(self.stake, self.cost, alpha, saturation_point)
        if len(current_pools) * saturation_point < self.model.total_stake:
            return potential_profit > 0
        potential_pool = Pool(pool_id=-1, cost=self.cost, pledge=self.stake, margin=0, owner=self.unique_id,
                              alpha=self.model.alpha, beta=self.model.beta, is_private=self.stake >= self.model.beta)
        potential_desirability = potential_pool.calculate_desirability()

        existing_desirabilities = [pool.calculate_desirability() for pool in current_pools]
        return potential_profit > 0 and any(
            desirability < potential_desirability for desirability in existing_desirabilities)
        # existing_potential_profits = [pool.potential_profit for pool in current_pools]
        # return any(profit < potential_profit for profit in existing_potential_profits)

    def calculate_pledges(self, num_pools):
        """
        The players choose to allocate their entire stake as the pledge of their pools,
        so they divide it equally among them
        However, if they saturate all their pools with pledge and still have remaining stake,
        then they don't allocate all of it to their pools, as a pool with such a pledge above saturation
         would yield suboptimal rewards
        :return:
        """
        if num_pools <= 0:
            raise ValueError("Player tried to calculate pledge for zero or less pools.")
        return [min(self.stake / num_pools, self.model.beta)] * num_pools

    # todo how do we avoid myopic margin increases?
    def calculate_margin_binary_search(self, pool, initial_margin=0.25):
        new_pool = deepcopy(pool)
        all_pools = deepcopy(self.model.pools)
        all_pools[new_pool.id] = new_pool

        max_depth = 5
        depth = 0
        lower_bound = 0
        mid = initial_margin
        upper_bound = min(2 * mid - lower_bound, 1)
        new_pool.margin = mid
        mid_utility = self.calculate_operator_utility_by_pool(new_pool,
                                                              all_pools)  # todo problem with player's other pools when calculating non-myopic stake

        while depth < max_depth:  # todo if max_depth is big, add an alternative condition to indicate convergence to a good margin
            new_margin = (lower_bound + mid) / 2
            new_pool.margin = new_margin
            new_utility = self.calculate_operator_utility_by_pool(new_pool, all_pools)
            if new_utility >= mid_utility:
                upper_bound = mid
            else:
                lower_bound = mid
            mid = (lower_bound + upper_bound) / 2
            new_pool.margin = mid
            mid_utility = self.calculate_operator_utility_by_pool(new_pool, all_pools)
            depth += 1
        return mid

    def calculate_margin_simple(self, current_margin):
        '''if self.strategy.stake_allocations[self.unique_id] >= self.model.beta:
            return 0  # single-man pool, so margin is irrelevant'''
        if current_margin < 0:
            # player doesn't have a pool yet so he sets the max margin
            return STARTING_MARGIN
        # player already has a pool
        return max(current_margin - MARGIN_INCREMENT, 0)

    def calculate_margin_increment(self, pool):
        current_margin = pool.margin
        new_pool = deepcopy(pool)
        all_pools = deepcopy(self.model.pools)
        if current_margin < 0:
            # player doesn't have a pool yet so he sets the max margin
            return STARTING_MARGIN
        # player already has a pool
        # compare current margin with one increment up and one increment down and choose the one with the highest utility
        max_utility = 0
        margin = -1
        margin_candidates = {max(current_margin - MARGIN_INCREMENT, 0), current_margin,
                             min(current_margin + MARGIN_INCREMENT, 1)}
        for margin_candidate in margin_candidates:
            new_pool.margin = margin_candidate
            utility = self.calculate_operator_utility_by_pool(new_pool, all_pools)
            if utility > max_utility:
                max_utility = utility
                margin = margin_candidate
        return margin

    def calculate_margin_increment_down(self, pool):
        current_margin = pool.margin
        new_pool = deepcopy(pool)
        all_pools = deepcopy(self.model.pools)
        if current_margin < 0:
            # player doesn't have a pool yet so he sets the max margin
            return STARTING_MARGIN
        # player already has a pool
        # compare current margin with one increment down and choose the one with the highest utility
        max_utility = 0
        margin = -1
        margin_candidates = {max(current_margin - MARGIN_INCREMENT, 0),
                             current_margin}  # todo keep in mind that order influences results
        for margin_candidate in margin_candidates:
            new_pool.margin = margin_candidate
            utility = self.calculate_operator_utility_by_pool(new_pool, all_pools)
            if utility > max_utility:
                max_utility = utility
                margin = margin_candidate
        return margin

    def calculate_margin_perfect_strategy(self):
        """
        Based on "perfect strategies", the player ranks all pools (existing and hypothetical) based on their potential
        profit and chooses a margin that can keep his pool competitive
        :return: float, the margin that the player will use to open a new pool
        """
        # first calculate the potential profits of all players
        potential_profits = {agent.unique_id:
            hlp.calculate_potential_profit(agent.stake, agent.cost, self.model.alpha, self.model.beta)
            for agent in self.model.schedule.agents}

        potential_profit_ranks = hlp.calculate_ranks(potential_profits)
        k = self.model.k
        n = self.model.num_agents
        keys = list(potential_profit_ranks.keys())
        values = list(potential_profit_ranks.values())
        # find the player who is ranked at position k+1, if such player exists
        reference_potential_profit = potential_profits[keys[values.index(k + 1)]] if k < n else 0

        margin = 1 - (reference_potential_profit / potential_profits[self.unique_id]) \
            if potential_profit_ranks[self.unique_id] <= k else 0
        return margin

    def determine_current_pools(self, num_pools):
        owned_pools = deepcopy(self.strategy.owned_pools)
        if num_pools < self.strategy.num_pools:
            # Ditch the pool(s) with the lowest stake todo or desirability?? basically lowest/highest rank (makes a difference when player has 2 pools of same stake but obviously one of them ranks higher than the other even if they are exactly the same due to necessary tie breaking)
            retiring_pools_num = self.strategy.num_pools - num_pools
            for i in range(retiring_pools_num):
                # owned_pools.pop(min(owned_pools, key=lambda key: owned_pools[key].stake))
                desirabilities = {id: pool.calculate_desirability() for id, pool in owned_pools.items()}
                ranks = hlp.calculate_ranks(desirabilities)
                # important to use rank and not desirabilities to make sure that the same tie breaking rule is followed
                owned_pools.pop(max(ranks, key=lambda key: ranks[key]))  # todo easier way to find max by value in dict
        return owned_pools

    def find_operator_moves(self):
        # Try out 3 options:
        # keep current number of pools (if > 0), increase by 1 or decrease by 1 (if > 1)
        moves = {}
        num_pools_options = []
        if self.model.pool_splitting:
            current_num_pools = self.strategy.num_pools
            if current_num_pools > 0:
                num_pools_options.append(current_num_pools)
                if current_num_pools > 1:
                    num_pools_options.append(current_num_pools - 1)
            num_pools_options.append(current_num_pools + 1)
        else:
            # If pool splitting is not allowed by the model, then the player only considers opening one pool
            num_pools_options = [1]
        for num_pools in num_pools_options:
            owned_pools = self.determine_current_pools(num_pools)
            moves[num_pools] = self.find_operator_move(num_pools, owned_pools)
        return moves

    def find_operator_move(self, num_pools, owned_pools):
        pledges = self.calculate_pledges(num_pools)
        margins = []

        cost_per_pool = self.model.common_cost + self.cost / num_pools if num_pools > 1 else self.cost  # we only apply the additional (common) cost in case of > 1 pools
        for i, (pool_id, pool) in enumerate(owned_pools.items()):
            # For pools that already exist, modify them to match the new strategy
            pool.stake -= pool.pledge - pledges[i]
            pool.pledge = pledges[i]
            pool.is_private = pool.pledge >= self.model.beta
            pool.cost = cost_per_pool
            pool.set_potential_profit(self.model.alpha, self.model.beta)
            # pool.margin = 0 if pool.is_private else self.calculate_margin_binary_search(pool, pool.margin)
            # pool.margin = 0 if pool.is_private else self.calculate_margin_increment(pool)
            # pool.margin = 0 if pool.is_private else self.calculate_margin_simple(pool.margin)
            pool.margin = 0 if pool.is_private else self.calculate_margin_increment_down(pool)
            margins.append(pool.margin)
            owned_pools[pool.id] = pool
        existing_pools_num = len(owned_pools)
        for i in range(existing_pools_num, num_pools):
            # For pools under consideration of opening, create according to the strategy
            pool_id = self.model.get_next_pool_id()
            self.model.pool_owner_id_mapping[pool_id] = self.unique_id
            pool = Pool(pool_id=pool_id, cost=cost_per_pool,
                        pledge=pledges[i], owner=self.unique_id, alpha=self.model.alpha,
                        beta=self.model.beta, is_private=pledges[i] >= self.model.beta)
            # private pools have margin 0 but don't allow delegations
            # pool.margin = 0 if pool.is_private else self.calculate_margin_binary_search(pool)
            # pool.margin = 0 if pool.is_private else self.calculate_margin_increment(pool)
            # pool.margin = 0 if pool.is_private else self.calculate_margin_simple(pool.margin)
            pool.margin = 0 if pool.is_private else self.calculate_margin_increment_down(pool)
            margins.append(pool.margin)
            owned_pools[pool.id] = pool

        allocations = self.find_delegation_move_for_operator(pledges)

        return Strategy(pledges=pledges, margins=margins, stake_allocations=allocations,
                        is_pool_operator=True, num_pools=num_pools, owned_pools=owned_pools)

    def find_delegation_move_for_operator(self, pledges):
        allocations = defaultdict(lambda: 0)
        remaining_stake = self.stake - sum(pledges)
        if remaining_stake > 0:
            # in some cases players may not want to allocate their entire stake to their pool (e.g. when stake > β)
            delegation_strategy = self.find_delegation_move_desirability(stake_to_delegate=remaining_stake)
            allocations = delegation_strategy.stake_allocations
        return allocations

    def find_delegation_move_desirability(self, stake_to_delegate=None):
        """
        Choose a delegation move based on the desirability of the existing pools. If two or more pools are tied,
        choose the one with the highest (current) stake, as it offers higher short-term rewards.
        :return:
        """
        saturation_point = self.model.beta
        if stake_to_delegate is None:
            stake_to_delegate = self.stake

        pools = deepcopy(self.model.pools)
        # remove the player's stake from the pools in case it's being delegated
        for pool_id, allocation in self.strategy.stake_allocations.items():
            if allocation > 0:
                pools[pool_id].update_delegation(stake=-allocation, delegator_id=self.unique_id)
        pools_list = list(pools.values())
        allocations = defaultdict(lambda: 0)

        if self.isMyopic:
            desirabilities_n_stakes = {pool.id: (pool.calculate_myopic_desirability(self.model.alpha, saturation_point),
                                                 pool.stake)
                                       for pool in pools_list
                                       if pool.owner != self.unique_id and not pool.is_private}
        else:
            desirabilities_n_stakes = {pool.id: (pool.calculate_desirability(), pool.stake)
                                       for pool in pools_list
                                       if
                                       pool.owner != self.unique_id and not pool.is_private}  # todo should we allow players to delegate to their own pools? makes sesne only if we consider that pledge is locked and delegation is not
        # Order the pools based on desirability and stake (to break ties in desirability) and delegate the stake to
        # the first non-saturated pool(s).
        for pool_id, (desirability, stake) in sorted(desirabilities_n_stakes.items(),
                                                     key=operator.itemgetter(1), reverse=True):
            if stake_to_delegate == 0:
                break
            if stake < saturation_point:
                stake_to_saturation = saturation_point - stake
                allocation = min(stake_to_delegate, stake_to_saturation)
                if allocation > 0:  # redundant?
                    stake_to_delegate -= allocation
                    allocations[pool_id] = allocation

        return Strategy(stake_allocations=allocations, is_pool_operator=False)

    def execute_strategy(self):
        """
        Execute the player's current strategy
        :return: void

        """
        current_pools = self.model.pools

        old_allocations = self.strategy.stake_allocations
        new_allocations = self.new_strategy.stake_allocations
        relevant_pools = old_allocations.keys() | new_allocations.keys()
        allocation_changes = {pool_id: new_allocations[pool_id] - old_allocations[pool_id] for pool_id in
                              relevant_pools}

        old_owned_pools = set(self.strategy.owned_pools.keys())
        new_owned_pools = set(self.new_strategy.owned_pools.keys())

        for pool_id in old_owned_pools - new_owned_pools:
            # pools have closed
            self.close_pool(pool_id)
        for pool_id in new_owned_pools & old_owned_pools:
            # updates in old pools
            updated_pool = self.new_strategy.owned_pools[pool_id]
            if updated_pool is None:
                current_pools.pop(pool_id)
                continue
            # todo alternatively keep delegators and stake(?) and set current_pools[pool_id] = updated_pool
            current_pools[pool_id].margin = updated_pool.margin
            pledge_diff = current_pools[pool_id].pledge - updated_pool.pledge
            current_pools[pool_id].stake -= pledge_diff
            current_pools[pool_id].pledge = updated_pool.pledge
            current_pools[pool_id].cost = updated_pool.cost
            current_pools[pool_id].is_private = updated_pool.is_private
            current_pools[pool_id].set_potential_profit(self.model.alpha, self.model.beta)
            # todo should we make the player idle after a pool update??
            # self.idle_steps_remaining = self.model.idle_steps_after_pool

        self.strategy = self.new_strategy
        self.new_strategy = None
        for pool_id in new_owned_pools - old_owned_pools:
            self.open_pool(pool_id)

        for pool_id in allocation_changes:
            if current_pools[pool_id] is not None:
                # add or subtract the relevant stake from the pool if it hasn't closed
                current_pools[pool_id].update_delegation(stake=allocation_changes[pool_id], delegator_id=self.unique_id)

    def open_pool(self, pool_id):
        self.model.pools[pool_id] = self.strategy.owned_pools[pool_id]
        self.idle_steps_remaining = self.model.idle_steps_after_pool

    def close_pool(self, pool_id):
        pools = self.model.pools
        try:
            if pools[pool_id].owner != self.unique_id:
                raise ValueError("Player tried to close pool that belongs to another player.")
        except AttributeError:
            raise ValueError("Given pool id is not valid.")
        pools.pop(pool_id, None)
        # Undelegate delegators' stake
        for agent in self.model.schedule.agents:
            agent.strategy.stake_allocations.pop(pool_id, None)
            if self.model.player_activation_order == "Simultaneous":
                # Also remove pool from players' upcoming moves in case of simultenous activation
                if agent.new_strategy is not None:
                    agent.new_strategy.stake_allocations.pop(pool_id, None)

    def get_status(self):
        print("Agent id: {}, is myopic: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.isMyopic, self.stake, self.cost))
        print("\n")
