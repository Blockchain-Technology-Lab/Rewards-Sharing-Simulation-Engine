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
from logic.strategy import MAX_MARGIN
from logic.strategy import MARGIN_INCREMENT
from logic.custom_exceptions import PoolNotFoundError, NonPositiveAllocationError

UTILITY_THRESHOLD = 1e-9


class Stakeholder(Agent):
    def __init__(self, unique_id, model, stake=0, is_myopic=False,
                 cost=0, strategy=None):
        super().__init__(unique_id, model)
        self.cost = cost  # the player's cost of running one pool
        self.stake = stake
        self.isMyopic = is_myopic
        self.idle_steps_remaining = 0
        self.new_strategy = None

        if strategy is None:
            # initialise strategy to being a delegator with no allocated stake
            strategy = Strategy()
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

    # Ιf we want players to make moves simultaneously, then we need an additional advance method
    # specifically, "step() activates the agent and stages any necessary changes, but does not apply them yet
    #                advance() then applies the changes"
    def advance(self):
        if self.new_strategy is not None:
            # The player has changed their strategy, so now they have to execute it
            self.execute_strategy()
            self.model.current_step_idle = False

    def make_move(self):
        self.update_strategy()

    def update_strategy(self):

        current_utility = self.calculate_utility(self.strategy)
        possible_moves = {"current": (
            current_utility + UTILITY_THRESHOLD, self.strategy)}  # every dict value is a tuple of utility, strategy

        # For all players, find a possible delegation strategy and calculate its potential utility
        delegator_strategy = self.find_delegation_move_desirability()
        delegator_utility = self.calculate_utility(delegator_strategy)
        possible_moves["delegator"] = delegator_utility, delegator_strategy

        if self.strategy.is_pool_operator:
            current_plus_strategy = self.find_current_plus_move()  # todo rename please
            current_plus_utility = self.calculate_utility(current_plus_strategy)
            possible_moves["current+"] = current_plus_utility, current_plus_strategy

        if self.strategy.is_pool_operator or self.has_potential_for_pool():
            # Player is considering opening a pool, so he has to find the most suitable pool params
            # and calculate the potential utility of operating a pool with these params
            operator_strategy = self.find_operator_move()
            operator_utility = self.calculate_utility(operator_strategy)
            possible_moves["operator"] = operator_utility, operator_strategy

        # compare the above with the utility of the current strategy and pick one of the 3
        # in case of a tie, the max function picks the element with the lowest index, so we have strategically ordered
        # them earlier so that the "easiest" move is preferred ( current -> delegator -> operator)
        max_utility_option = max(possible_moves,
                                 key=lambda key: possible_moves[key][0])

        if "operator" in possible_moves.keys() and max_utility_option != "operator":
            # discard the pool ids that were used for the hypothetical operator move
            old_owned_pools = set(self.strategy.owned_pools.keys())
            hypothetical_owned_pools = set(operator_strategy.owned_pools.keys())
            self.model.rewind_pool_id_seq(step=len(hypothetical_owned_pools - old_owned_pools))

        self.new_strategy = None if max_utility_option == "current" else possible_moves[max_utility_option][1]

    def calculate_utility(self, strategy):
        utility = 0
        pools = self.model.pools

        if strategy.is_pool_operator:
            cost_per_pool = self.model.common_cost + self.cost / strategy.num_pools
            for index, pool_id in enumerate(strategy.owned_pools):
                # for pools that already exist
                pool = deepcopy(pools[pool_id])
                pool.margin = strategy.margins[index]
                pool.stake -= pool.pledge - strategy.pledges[index]
                pool.pledge = strategy.pledges[index]
                pool.cost = cost_per_pool
                pool.set_potential_profit(self.model.alpha, self.model.beta)
                strategy.owned_pools[pool.id] = pool
                utility += self.calculate_operator_utility(pool)

            for i in range(len(strategy.owned_pools),
                           strategy.num_pools):  # todo make sure that index carries on from before
                # for pools under consideration of opening
                # we calculate the utility of operating a hypothetical pool
                pool = Pool(pool_id=self.model.get_next_pool_id(), margin=strategy.margins[i], cost=cost_per_pool,
                            pledge=strategy.pledges[i], owner=self.unique_id, alpha=self.model.alpha,
                            beta=self.model.beta)
                strategy.owned_pools[pool.id] = pool
                utility += self.calculate_operator_utility(pool)

        # calculate delegator utility
        for pool_id in strategy.stake_allocations:
            a = strategy.stake_allocations[pool_id]
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

    def calculate_operator_utility(self, pool, stake_allocation=-1):
        pledge = pool.pledge
        if stake_allocation == -1:
            stake_allocation = pledge
        m = pool.margin
        alpha = self.model.alpha
        beta = self.model.beta
        pool_stake = pool.stake if self.isMyopic else hlp.calculate_pool_stake_NM(pool,
                                                                                  self.model.get_pools_list(),
                                                                                  beta,
                                                                                  self.model.k
                                                                                  )
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
        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id]
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        non_myopic_stake = max(hlp.calculate_pool_stake_NM(pool,
                                                           self.model.get_pools_list(),
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

        potential_profit = (hlp.calculate_potential_profit(self.stake, self.cost, alpha, saturation_point))
        if len(current_pools) * saturation_point < self.model.total_stake:
            return potential_profit > 0

        existing_potential_profits = [pool.potential_profit for pool in current_pools]
        return any(profit < potential_profit for profit in existing_potential_profits)

    def calculate_pledges(self, num_pools):
        """
        Based on "perfect strategies", the players choose to allocate their entire stake as the pledge of their pool
        However, if their stake is larger than the saturation point, they don't allocate all of it,
        as a pool with such a pledge would yield suboptimal rewards
        :return:
        """
        if num_pools <= 0:
            raise ValueError("Player tried to calculate pledge for zero or less pools.")
        return [min(self.stake / num_pools, self.model.beta)] * num_pools

    def calculate_margin_binary_search(self):

        pass

    def calculate_margin_simple(self, current_margin):
        '''if self.strategy.stake_allocations[self.unique_id] >= self.model.beta:
            return 0  # single-man pool, so margin is irrelevant'''
        # todo keep in mind that when you access a dict key that doesn't exist a new entry is created with the default value (here 0)
        if current_margin < 0:
            # player doesn't have a pool yet so he sets the max margin
            return MAX_MARGIN
        # player already has a pool
        return max(current_margin - MARGIN_INCREMENT, 0)

        # alternative way to try out margins until a suitable one is found
        '''current_pool = self.model.pools[self.unique_id]
        current_operator_utility = self.calculate_operator_utility(current_pool,
                                                                   self.strategy.stake_allocations[self.unique_id])
        new_margin = max(current_margin - MARGIN_INCREMENT, 0)
        new_pool = deepcopy(current_pool)
        while True:
            new_pool.margin = new_margin
            new_operator_utility = self.calculate_operator_utility(new_pool,
                                                                   self.strategy.stake_allocations[self.unique_id])
            if new_operator_utility > current_operator_utility:
                return new_margin
            if new_margin == 0:
                break
            new_margin = max(new_margin - MARGIN_INCREMENT, 0)
        return current_margin'''

    def calculate_margin_perfect_strategy(self):
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
        # find the player who is ranked at the kth position (k+1 if we start from 1)
        k_rank_index = potential_profit_ranks.index(k)

        margin = 1 - (potential_profits[k_rank_index] / potential_profits[self.unique_id]) \
            if potential_profit_ranks[self.unique_id] < k else 0
        return margin

    # todo change logic to determine the number of pools for the operator move
    def determine_num_pools(self):
        return self.strategy.num_pools + 1  # increase possible number of pools by 1 each time

    def find_operator_move(self):
        if self.model.pool_splitting:
            num_pools = self.determine_num_pools()
        else:
            num_pools = 1

        previous_pools = self.strategy.owned_pools
        owned_pools = defaultdict(lambda: None)
        for i, pool_id in enumerate(
                previous_pools):  # todo this assumes that if a pool closes it will always be the last one in the dictionary
            if i < num_pools:
                owned_pools[pool_id] = previous_pools[pool_id]

        pledges = self.calculate_pledges(num_pools)

        current_margins = self.strategy.margins
        new_margins = []
        '''if pledges[i] >= self.model.beta: # special case for one-man pools
                        margins.append(0)'''
        for i, current_margin in enumerate(current_margins):
            new_margin = self.calculate_margin_simple(current_margin)
            new_margins.append(new_margin)
        for i in range(len(current_margins), num_pools):
            try:
                current_margin = current_margins[i - 1]
            except IndexError:
                current_margin = -1
            new_margin = self.calculate_margin_simple(current_margin)
            new_margins.append(new_margin)

        allocations = self.find_delegation_move_for_operator(pledges)

        return Strategy(pledges=pledges, margins=new_margins, stake_allocations=allocations,
                        is_pool_operator=True, num_pools=num_pools, owned_pools=owned_pools)

    def find_current_plus_move(self):
        num_pools = self.strategy.num_pools
        owned_pools = self.strategy.owned_pools
        pledges = self.strategy.pledges
        margins = self.strategy.margins

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
            # if allocation > 0:
            pools[pool_id].update_delegation(stake=-allocation, delegator_id=self.unique_id)
        pools_list = list(pools.values())
        allocations = defaultdict(lambda: 0)

        if self.isMyopic:
            desirabilities_n_stakes = {pool.id: (pool.calculate_myopic_desirability(self.model.alpha, saturation_point),
                                                 pool.stake)
                                       for pool in pools_list if pool.owner != self.unique_id}
        else:
            desirabilities_n_stakes = {pool.id: (pool.calculate_desirability(), pool.stake)
                                       for pool in pools_list if pool.owner != self.unique_id}  # todo would make s
        # Order the pools based on desirability and stake (to break ties in desirability) and delegate the stake to
        # the first non-saturated pool(s).
        for pool_id, (desirability, stake) in sorted(desirabilities_n_stakes.items(),
                                                     key=operator.itemgetter(1), reverse=True):
            if stake_to_delegate == 0:
                break
            if stake < saturation_point:
                stake_to_saturation = saturation_point - stake
                allocation = min(stake_to_delegate, stake_to_saturation)
                if allocation > 0:  # reduntant?
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
            current_pools[pool_id].set_potential_profit(self.model.alpha, self.model.beta)

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
            
    def find_delegation_move_potential_profit(self):
        """
        Choose a delegation move based on the potential profit of the existing pools
        :return:
        """
        saturation_point = self.model.beta
        alpha = self.model.alpha

        pools = deepcopy(self.model.pools)
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

        return SinglePoolStrategy(stake_allocations=allocations, is_pool_operator=False)
        
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
        return strategy
 
    '''
