# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:14:45 2021

@author: chris
"""
from mesa import Agent
from copy import deepcopy
import operator

import logic.helper as hlp
from logic.pool import Pool
from logic.strategy import Strategy
from logic.strategy import STARTING_MARGIN
from logic.strategy import MARGIN_INCREMENT
from logic.helper import MIN_STAKE_UNIT
from logic.custom_exceptions import PoolNotFoundError, NonPositiveAllocationError


class Stakeholder(Agent):

    def __init__(self, unique_id, model, stake=0, is_myopic=False, is_abstainer=False,
                 cost=0, strategy=None):
        super().__init__(unique_id, model)
        self.cost = cost  # the player's individual cost of running one or more pools
        self.stake = stake
        self.isMyopic = is_myopic
        self.abstains = is_abstainer
        self.remaining_min_steps_to_keep_pool = 0
        self.new_strategy = None

        if strategy is None:
            # Initialise strategy to being a delegator with no allocated stake
            strategy = Strategy()
        self.strategy = strategy

    # In every step the agent needs to decide what to do
    def step(self):
        self.update_strategy()
        if "simultaneous" not in self.model.player_activation_order.lower():
            self.advance()
        if self.remaining_min_steps_to_keep_pool > 0:
            # For players that are recently opened a pool
            self.remaining_min_steps_to_keep_pool -= 1

    # When players make moves simultaneously, "step() activates the agent and stages any necessary changes,
    # but does not apply them yet, and advance() then applies the changes"
    def advance(self):
        if self.new_strategy is not None:
            # The player has changed their strategy, so now they have to execute it
            self.execute_strategy()
            self.model.current_step_idle = False

    def update_strategy(self):
        if not self.abstains:
            self.make_move()
        else:
            # player abstains from this round
            if self.strategy.is_pool_operator or len(self.strategy.stake_allocations) > 0:
                # player did not abstain in the previous round
                self.new_strategy = Strategy()

    def make_move(self):
        current_utility = self.calculate_utility(self.strategy)
        current_utility_with_inertia = max(
            (1 + self.model.relative_utility_threshold) * current_utility,
            current_utility + self.model.absolute_utility_threshold
        )
        # hold the player's potential moves in a dict, where the values are tuples of (utility, strategy)
        possible_moves = {"current": (current_utility_with_inertia, self.strategy)}

        # For all players, find a possible delegation strategy and calculate its potential utility
        # unless they are pool operators with recently opened pools (we assume that they will keep them at least for a bit)
        if self.remaining_min_steps_to_keep_pool == 0:
            delegator_strategy = self.find_delegation_move_desirability()
            delegator_utility = self.calculate_utility(delegator_strategy)
            possible_moves["delegator"] = delegator_utility, delegator_strategy

        if self.strategy.is_pool_operator or self.has_potential_for_pool():
            # Player is considering opening a pool, so he has to find the most suitable pool params
            # and calculate the potential utility of operating a pool with these params
            operator_strategies = self.find_operator_moves()
            max_operator_utility = 0
            max_operator_strategy = None
            for num_pools in sorted(operator_strategies.keys()):
                operator_strategy = operator_strategies[num_pools]
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
            utility += self.calculate_operator_utility_from_strategy(strategy)

        pools = self.model.pools
        # Calculate utility of delegating to other pools
        for pool_id, a in strategy.stake_allocations.items():
            if a <= 0:
                continue
            if pool_id in pools:
                pool = pools[pool_id]
                utility += self.calculate_delegator_utility_from_pool(pool, a)
            else:
                raise PoolNotFoundError("Player {} considered delegating to a non-existing pool ({})!"
                                        .format(self.unique_id, pool_id))
        return utility

    def calculate_operator_utility_from_strategy(self, strategy):
        utility = 0
        potential_pools = strategy.owned_pools
        fixed_pools = {pool_id: pool for pool_id, pool in self.model.pools.items() if pool.owner != self.unique_id}
        all_considered_pools = fixed_pools | potential_pools
        for pool in potential_pools.values():
            utility += self.calculate_operator_utility_from_pool(pool, all_considered_pools)
        return utility

    def calculate_operator_utility_from_pool(self, pool, all_pools):
        alpha = self.model.alpha
        beta = self.model.beta
        pledge = pool.pledge
        '''pool_stake = pool.stake if self.isMyopic else hlp.calculate_pool_stake_NM(pool.id,
                                                                                  all_pools,
                                                                                  beta,
                                                                                  self.model.k
                                                                                  )'''  # assuming there is no myopic play for pool owners
        pool_stake = hlp.calculate_pool_stake_NM(
            pool.id,
            all_pools,
            beta,
            self.model.k
        )
        r = hlp.calculate_pool_reward(pool_stake, pledge, alpha, beta, self.model.reward_function_option)
        stake_allocation = pool.pledge  # todo change if we allow pool owners to allocate stake to their pool separate to the pledge
        q = stake_allocation / pool_stake
        return hlp.calculate_operator_reward_from_pool(pool, r, q)

    def calculate_delegator_utility_from_pool(self, pool, stake_allocation):
        alpha = self.model.alpha
        beta = self.model.beta

        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id] \
            if pool.id in self.strategy.stake_allocations else 0
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        non_myopic_stake = max(
            hlp.calculate_pool_stake_NM(
                pool.id,
                self.model.pools,
                self.model.beta,
                self.model.k
            ),
            current_stake
        )
        pool_stake = current_stake if self.isMyopic else non_myopic_stake
        r = hlp.calculate_pool_reward(pool_stake, pool.pledge, alpha, beta, self.model.reward_function_option)

        q = stake_allocation / pool_stake
        return hlp.calculate_delegator_reward_from_pool(pool, r, q)

    # how does a myopic player decide whether to open a pool or not? -> for now we assume that all players play non-myopically when it comes to pool moves
    def has_potential_for_pool(self):
        """
        Determine whether the player is at a good position to open a pool, using the following rules:
        If the current pools are not enough to cover the total active stake of the system without getting oversaturated
        then having a positive potential profit is a necessary and sufficient condition for a player to open a pool
        (as it means that there is stake that is forced to remain undelegated
        or delegated to an oversaturated pool that yields suboptimal rewards)
        Note that the total active stake is estimated by the stakeholders at predetermined intervals and may not be
        100% accurate

        If there are enough pools to cover all active players' stake without causing oversaturation,
        then the player only opens a pool if the maximum possible desirability of their pool
        (aka the potential profit) is higher than the desirability of at least one currently active pool
        (with the perspective of "stealing" the delegators from that pool)

        :return: bool true if the player has potential to open pool based on the above rules, false otherwise
        """
        saturation_point = self.model.beta
        alpha = self.model.alpha
        current_pools = self.model.get_pools_list()

        potential_profit = hlp.calculate_potential_profit(self.stake, self.cost, alpha, saturation_point, self.model.reward_function_option)
        if len(current_pools) * saturation_point < self.model.perceived_active_stake:  # note that we use active stake instead of total stake
            return potential_profit > 0

        existing_desirabilities = [pool.calculate_desirability() for pool in current_pools]
        # Note that the potential profit is equal to the desirability of a pool with 0 margin,
        # so, effectively, the player is comparing his best-case desirability with the desirabilities of the current pools
        return potential_profit > 0 and any(
            desirability < potential_profit for desirability in existing_desirabilities)

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

    def calculate_margin_binary_search(self, pool, current_margin=0.25):
        """
        Calculate a suitable margin for a pool by trying out different values in a range
        and comparing the expected utilities they yield. Since the range is continuous, the search terminates after a
        max number of tries has been reached.
        :param pool: the pool in question
        :param current_margin: the current margin of the pool if the pool already exists
                the current margin is used to determine the search range, i.e. the range is [0, 2 * current_margin].
                If it's for a new pool, then a default value of 0.25 is given as the hypothetical current margin.
                If the current margin is 0, then the range is set to [0, 0.2]
        :return: the margin value of the ones tested that yields the highest expected utility
        """
        new_pool = deepcopy(pool)
        all_pools = self.model.pools
        temp_pool = None
        if new_pool.id in all_pools:
            temp_pool = deepcopy(all_pools[new_pool.id])
        all_pools[new_pool.id] = new_pool

        if current_margin == 0:
            current_margin = 0.1
        lower_bound = 0
        upper_bound = min(2 * current_margin - lower_bound, 1)

        new_pool.margin = current_margin
        current_utility = self.calculate_operator_utility_from_pool(new_pool, all_pools)
        margin_utilities = {current_margin: current_utility}
        new_margin = (lower_bound + current_margin) / 2

        max_tries = 5
        current_try = 0

        while current_try < max_tries and new_margin != current_margin:
            new_pool.margin = new_margin
            new_utility = self.calculate_operator_utility_from_pool(new_pool, all_pools)
            margin_utilities[new_margin] = new_utility

            if new_utility >= current_utility:
                upper_bound = current_margin
            else:
                lower_bound = new_margin

            current_margin = (lower_bound + upper_bound) / 2
            new_pool.margin = current_margin
            current_utility = self.calculate_operator_utility_from_pool(new_pool, all_pools)
            margin_utilities[current_margin] = current_utility

            current_try += 1
            new_margin = (lower_bound + current_margin) / 2
        # make sure that model fields are left intact
        if temp_pool is None:
            all_pools.pop(new_pool.id)
        else:
            all_pools[new_pool.id] = temp_pool
        return max(margin_utilities, key=lambda key: margin_utilities[key])

    def calculate_margin_simple(self, current_margin=-1):
        if current_margin < 0:
            # player doesn't have a pool yet so he sets the max margin
            return STARTING_MARGIN
        # player already has a pool
        return max(current_margin - MARGIN_INCREMENT, 0)

    '''def calculate_margin_increment(self, pool):
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
        return margin'''

    def calculate_margin_semi_perfect_strategy(self, pool):
        """
        Inspired by "perfect strategies", the player ranks all existing pools based on their potential
        profit and chooses a margin that can guarantee the pool's desirability (non-zero only if the pool
        ranks in the top k)
        :return: float, the margin that the player will set for this pool
        """
        current_pool = deepcopy(pool)
        all_pools = self.model.pools
        temp_pool = None
        if current_pool.id in all_pools:
            temp_pool = deepcopy(all_pools[current_pool.id])
        all_pools[current_pool.id] = current_pool

        potential_profits = {
            pool_id: pool.potential_profit
            for pool_id, pool in all_pools.items()
        }

        potential_profit_ranks = hlp.calculate_ranks(potential_profits, rank_ids=True)
        k = self.model.k
        npools = len(all_pools)
        keys = list(potential_profit_ranks.keys())
        values = list(potential_profit_ranks.values())
        # find the pool that is ranked at position k+1, if such pool exists
        reference_potential_profit = potential_profits[keys[values.index(k + 1)]] if npools > k else min(potential_profits.values())
        margin = 1 - (reference_potential_profit / potential_profits[current_pool.id]) \
            if potential_profit_ranks[current_pool.id] <= k else 0
        # make sure that model fields are left intact
        if temp_pool is None:
            all_pools.pop(current_pool.id)
        else:
            all_pools[current_pool.id] = temp_pool

        return margin

    def calculate_margin_perfect_strategy(self):
        """
        Based on "perfect strategies", the player ranks all pools (existing and hypothetical) based on their potential
        profit and chooses a margin that can keep his pool competitive
        :return: float, the margin that the player will use to open a new pool
        """
        # first calculate the potential profits of all players
        players = self.model.get_players_dict()
        potential_profits = {player_id:
                                 hlp.calculate_potential_profit(player.stake, player.cost, self.model.alpha,
                                                                self.model.beta, self.model.reward_function_option)
                             for player_id, player in players.items()}

        potential_profit_ranks = hlp.calculate_ranks(potential_profits, rank_ids=True)
        k = self.model.k
        n = self.model.n
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
            # Ditch the pool(s) with the lowest desirability / rank
            retiring_pools_num = self.strategy.num_pools - num_pools
            for i in range(retiring_pools_num):
                # owned_pools.pop(min(owned_pools, key=lambda key: owned_pools[key].stake))
                desirabilities = {id: pool.calculate_desirability() for id, pool in owned_pools.items()}
                ranks = hlp.calculate_ranks(desirabilities, rank_ids=True)
                # important to use rank and not desirabilities to make sure that the same tie breaking rule is followed
                owned_pools.pop(max(ranks, key=lambda key: ranks[key]))
        return owned_pools

    def find_operator_moves(self):
        """
        When players are allowed to operate multiple pools, they try out different options to decide
        exactly how many pools to operate. Specifically, they always consider running any number of pools up to their
        current number (i.e. closing some pools or keeping the same number) and they also consider opening one
        extra pool. In case an operator has recently opened a new pool, they are not allowed to close any currently open
        pools, so they only consider keeping the same number of pools or adding one to it.
        :return:
        """
        moves = dict()
        current_num_pools = self.strategy.num_pools
        if self.model.pool_splitting:
            if self.remaining_min_steps_to_keep_pool > 0:
                num_pools_options = {current_num_pools, current_num_pools + 1}
            else:
                num_pools_options = {i for i in range(1, current_num_pools + 2)}
        else:
            # If pool splitting is not allowed by the model, there are no options
            num_pools_options = {1}

        for num_pools in num_pools_options:
            owned_pools = self.determine_current_pools(num_pools)
            moves[num_pools] = self.find_operator_move(num_pools, owned_pools)
        return moves

    def find_operator_move(self, num_pools, owned_pools):
        pledges = self.calculate_pledges(num_pools)
        margins = []

        cost_per_pool = hlp.calculate_cost_per_pool_fixed_fraction(num_pools, self.cost, self.model.cost_factor) if \
            self.model.extra_cost_type == 'fixed_fraction' else hlp.calculate_cost_per_pool(num_pools, self.cost, self.model.cost_factor)
        for i, (pool_id, pool) in enumerate(owned_pools.items()):
            # For pools that already exist, modify them to match the new strategy
            pool.stake -= pool.pledge - pledges[i]
            pool.pledge = pledges[i]
            pool.is_private = pool.pledge >= self.model.beta
            pool.cost = cost_per_pool
            pool.set_potential_profit(self.model.alpha, self.model.beta, self.model.reward_function_option)
            pool.margin = self.calculate_margin_semi_perfect_strategy(pool)
            margins.append(pool.margin)
            owned_pools[pool.id] = pool
        existing_pools_num = len(owned_pools)
        for i in range(existing_pools_num, num_pools):
            # For pools under consideration of opening, create according to the strategy
            pool_id = self.model.get_next_pool_id()
            self.model.pool_owner_id_mapping[pool_id] = self.unique_id
            pool = Pool(pool_id=pool_id, cost=cost_per_pool,
                        pledge=pledges[i], owner=self.unique_id, alpha=self.model.alpha,
                        beta=self.model.beta, is_private=pledges[i] >= self.model.beta,
                        reward_function_option=self.model.reward_function_option)
            # private pools have margin 0 but don't allow delegations
            pool.margin = self.calculate_margin_semi_perfect_strategy(pool)
            margins.append(pool.margin)
            owned_pools[pool.id] = pool

        allocations = self.find_delegation_move_for_operator(pledges)

        return Strategy(pledges=pledges, margins=margins, stake_allocations=allocations,
                        is_pool_operator=True, num_pools=num_pools, owned_pools=owned_pools)

    def find_delegation_move_for_operator(self, pledges):
        allocations = dict()
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
        if stake_to_delegate is None:
            stake_to_delegate = self.stake

        pools = self.model.pools
        temp_pools = dict()

        allocations = dict()

        # Remove the player's stake from the pools in case it's being delegated
        for pool_id, allocation in self.strategy.stake_allocations.items():
            if allocation > 0:
                temp_pools[pool_id] = deepcopy(pools[pool_id])
                pools[pool_id].update_delegation(stake=-allocation, delegator_id=self.unique_id)

        # todo should we allow players to delegate to their own pools? makes sense only if we consider that
        #  pledge is locked and delegation is not
        pools_list = [pool for pool in pools.values() if pool.owner != self.unique_id and not pool.is_private]

        # Only proceed if there are active pools in the system that don't belong to the current player
        if len(pools_list) > 0:
            saturation_point = self.model.beta

            desirability_dict = {
                pool.id:
                    pool.calculate_myopic_desirability(self.model.alpha, saturation_point) if self.isMyopic
                    else pool.calculate_desirability()
                for pool in pools_list
            }
            pp_dict = {pool.id: pool.potential_profit for pool in pools_list}
            stake_dict = {pool.id: pool.stake for pool in pools_list}
            # Order the pools based on desirability and stake (to break ties in desirability) and delegate the stake to
            # the first non-saturated pool(s).
            pool_ranking = hlp.calculate_ranks(desirability_dict, pp_dict, stake_dict, rank_ids=True)
            allow_oversaturation = False
            while stake_to_delegate >= MIN_STAKE_UNIT:
                for pool_id, rank in sorted(pool_ranking.items(),  key=operator.itemgetter(1)):
                    stake_to_saturation = saturation_point - stake_dict[pool_id]
                    if stake_to_saturation >= MIN_STAKE_UNIT or allow_oversaturation:
                        allocation = stake_to_delegate if allow_oversaturation else min(stake_to_delegate,
                                                                                        stake_to_saturation)
                        stake_to_delegate -= allocation
                        allocations[pool_id] = allocation
                    if stake_to_delegate <= MIN_STAKE_UNIT:
                        break
                # there were not enough non-saturated pools for the player to delegate their stake to
                # so they have to choose a saturated pool
                allow_oversaturation = True

        for pool_id, pool in temp_pools.items():
            pools[pool_id] = pool

        return Strategy(stake_allocations=allocations, is_pool_operator=False)

    def execute_strategy(self):
        current_pools = self.model.pools

        old_allocations = self.strategy.stake_allocations
        new_allocations = self.new_strategy.stake_allocations
        # todo make simpler
        allocation_changes = dict()
        old_pool_ids = old_allocations.keys()
        new_pool_ids = new_allocations.keys()
        for pool_id in old_pool_ids - new_pool_ids:
            allocation_changes[pool_id] = -old_allocations[pool_id]
        for pool_id in old_pool_ids & new_pool_ids:
            allocation_changes[pool_id] = new_allocations[pool_id] - old_allocations[pool_id]
        for pool_id in new_pool_ids - old_pool_ids:
            allocation_changes[pool_id] = new_allocations[pool_id]

        old_owned_pools = set(self.strategy.owned_pools.keys())
        new_owned_pools = set(self.new_strategy.owned_pools.keys())

        for pool_id in old_owned_pools - new_owned_pools:
            # pools have closed
            self.close_pool(pool_id)
        for pool_id in new_owned_pools & old_owned_pools:
            # updates in old pools
            updated_pool = self.new_strategy.owned_pools[pool_id]
            if updated_pool is None:
                current_pools.pop(pool_id)  # todo can it ever be None?
                continue
            # todo alternatively keep delegators and stake(?) and set current_pools[pool_id] = updated_pool
            current_pools[pool_id].margin_change = updated_pool.margin - current_pools[pool_id].margin
            current_pools[pool_id].margin = updated_pool.margin
            pledge_diff = current_pools[pool_id].pledge - updated_pool.pledge
            current_pools[pool_id].stake -= pledge_diff
            current_pools[pool_id].pledge = updated_pool.pledge
            current_pools[pool_id].cost = updated_pool.cost
            current_pools[pool_id].is_private = updated_pool.is_private
            if current_pools[pool_id].is_private:
                # undelegate stake in case the pool turned from public to private
                self.remove_delegations(pool_id)
            current_pools[pool_id].set_potential_profit(self.model.alpha, self.model.beta, self.model.reward_function_option)

        self.strategy = self.new_strategy
        self.new_strategy = None
        for pool_id in new_owned_pools - old_owned_pools:
            self.open_pool(pool_id)

        for pool_id, allocation_change in allocation_changes.items():
            if current_pools[pool_id] is not None:  # todo can't really be none here, right?
                # add or subtract the relevant stake from the pool if it hasn't closed
                if allocation_change != 0:
                    current_pools[pool_id].update_delegation(stake=allocation_changes[pool_id],
                                                             delegator_id=self.unique_id)

    def open_pool(self, pool_id):
        self.model.pools[pool_id] = self.strategy.owned_pools[pool_id]
        self.remaining_min_steps_to_keep_pool = self.model.min_steps_to_keep_pool

    def close_pool(self, pool_id):
        pools = self.model.pools
        try:
            if pools[pool_id].owner != self.unique_id:
                raise ValueError("Player tried to close pool that belongs to another player.")
        except KeyError:
            raise ValueError("Given pool id is not valid.")
        # Undelegate delegators' stake
        self.remove_delegations(pool_id)
        pools.pop(pool_id)

    def remove_delegations(self, pool_id):  # todo potentially add in pool class (with changes)
        pool = self.model.pools[pool_id]
        players = self.model.get_players_dict()
        delegators = list(pool.delegators.keys())
        for player_id in delegators:
            pool.update_delegation(-pool.delegators[player_id], player_id)  # todo is that necessary here?
            player = players[player_id]
            player.strategy.stake_allocations.pop(pool_id)
        # Also remove pool from players' upcoming moves in case of (semi)simultaneous activation
        if "simultaneous" in self.model.player_activation_order.lower():
            for player in players.values():  # todo alternatively save potential delegators somewhere so that we don't go through the whole list of players here
                if player.new_strategy is not None:
                    player.new_strategy.stake_allocations.pop(pool_id, None)

    def get_status(self):
        print("Agent id: {}, is myopic: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.isMyopic, self.stake, self.cost))
        print("\n")
