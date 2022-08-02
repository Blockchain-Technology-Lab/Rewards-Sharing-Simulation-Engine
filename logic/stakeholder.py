# -*- coding: utf-8 -*-
from mesa import Agent
from copy import deepcopy
import heapq
import math

import logic.helper as hlp
from logic.pool import Pool
from logic.strategy import Strategy
from logic.helper import MIN_STAKE_UNIT

#todo move methods to children classes when necessary
class Stakeholder(Agent):

    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id, model)
        self.cost = cost  # the cost of running one pool for this agent
        self.stake = stake
        self.remaining_min_steps_to_keep_pool = 0
        self.new_strategy = None
        if strategy is None:
            # Initialise strategy to an "empty" strategy
            strategy = Strategy()
        self.strategy = strategy

    def step(self):
        self.update_strategy()
        if "simultaneous" not in self.model.agent_activation_order.lower():
            # When agents make moves simultaneously, "step() activates the agent and stages any necessary changes,
            # but does not apply them yet, and advance() then applies the changes". When they don't move simultaneously,
            # they can advance (i.e. execute their strategy) right after updating their strategy
            self.advance()
        if self.remaining_min_steps_to_keep_pool > 0:
            # For agents that have recently opened a pool
            self.remaining_min_steps_to_keep_pool -= 1

    def advance(self):
        if self.new_strategy is not None:
            # The agent has changed their strategy, so now they have to execute it
            self.execute_strategy()
            self.model.current_step_idle = False

    def update_strategy(self):
        current_utility = self.calculate_current_utility()
        current_move_expected_utility = self.calculate_expected_utility(self.strategy)
        augmented_current_move_utility = max(
            (1 + self.model.relative_utility_threshold) * current_utility,
            current_utility + self.model.absolute_utility_threshold,
            (1 + self.model.relative_utility_threshold) * current_move_expected_utility,
            current_move_expected_utility + self.model.absolute_utility_threshold
        )
        # hold the agent's potential moves in a dict, where the values are tuples of (utility, strategy)
        possible_moves = {"current": (augmented_current_move_utility, self.strategy)}

        # For all agents, find a possible delegation strategy and calculate its potential utility
        # unless they are pool operators with recently opened pools (we assume that they will keep them at least for a bit)
        if self.remaining_min_steps_to_keep_pool == 0:
            delegator_strategy = self.find_delegation_move()
            delegator_utility = self.calculate_expected_utility(delegator_strategy)
            possible_moves["delegator"] = delegator_utility, delegator_strategy
            pool_strategy = self.choose_pool_strategy()
            if pool_strategy[1] is not None:
                possible_moves["operator"] = pool_strategy
        else:
            raise ValueError("Pool opening process '{}' is not supported".format(self.model.pool_opening_process))

        # Compare the above with the utility of the current strategy and pick one of the 3
        # in case of a tie, the max function picks the element with the lowest index, so we have strategically ordered
        # them earlier so that the "easiest" move is preferred ( current -> delegator -> operator)
        max_utility_option = max(possible_moves, key=lambda key: possible_moves[key][0])
        #todo maybe discard temp pool ids here
        '''if max_utility_option == "operator":
            hlp.plot_margin_pools_heatmap(self)'''
        self.new_strategy = None if max_utility_option == "current" else possible_moves[max_utility_option][1]


    def discard_draft_pools(self, operator_strategy):
        # todo problem: if there are many different pool strategies developed, should we discard drafts from all of them?
        # Discard the pool ids that were used for the hypothetical operator move
        old_owned_pools = set(self.strategy.owned_pools.keys())
        hypothetical_owned_pools = set(operator_strategy.owned_pools.keys())
        self.model.rewind_pool_id_seq(step=len(hypothetical_owned_pools - old_owned_pools))

    def calculate_current_utility(self):
        utility = 0
        # Calculate current (not expected) utility of operating own pools
        for pool in self.strategy.owned_pools.values():
            utility += hlp.calculate_operator_utility_from_pool(pool_stake=pool.stake, pledge=pool.pledge, margin=pool.margin,
                                                                    cost=pool.cost, alpha=self.model.alpha, beta=self.model.beta, reward_function_option=self.model.reward_function_option, total_stake= self.model.total_stake)
        for pool_id, a in self.strategy.stake_allocations.items():
            utility += self.calculate_delegator_utility_from_pool(self.model.pools[pool_id], a)
        return utility

    def calculate_expected_utility(self, strategy):
        utility = 0

        # Calculate expected utility of operating own pools
        if len(strategy.owned_pools) > 0:
            utility += self.calculate_operator_utility_from_strategy(strategy)

        # Calculate expected utility of delegating to other pools
        pools = self.model.pools
        for pool_id, a in strategy.stake_allocations.items():
            if pool_id in pools:
                pool = pools[pool_id]
                utility += self.calculate_delegator_utility_from_pool(pool, a)
        return utility

    def choose_pool_strategy(self):
        """
        Find a suitable pool operation strategy by using the following process:
            - Start with an arbitrary number of pools t (we set this to (k+1)/2)
            - Calculate suitable margins so that all t pools end up in the top k (if not possible for some pools then set their margin to 0)
            - Calculate the agent's utility for this number of pools and margins
            - Do the same for the two neighbours of this strategy, i.e. operating t-1 pools and t+1 pools
            - Choose the direction with the highest utility and make a "jump" in t
            - If none of the neighbours have higher utility, solution found (strategy with t pools and calculated margins)
        This works because the utility of an agent as a function of the number of pools to operate has only one local max
        @return: a tuple with the utility of the chosen strategy and the strategy itself
        """
        t_min = 1
        t_max = self.model.k
        solution_found = False

        while not solution_found:
            t = math.floor((t_min + t_max) / 2)
            margins_t, utility_t = self.calculate_margins_and_utility(num_pools=t)
            if t > t_min:
                margins_t_minus, utility_t_minus = self.calculate_margins_and_utility(num_pools=t - 1)
                if utility_t_minus > utility_t:
                    t_max = t - 1
                    continue
            if t < t_max:
                margins_t_plus, utility_t_plus = self.calculate_margins_and_utility(num_pools=t + 1)
                if utility_t_plus > utility_t:
                    t_min = t + 1
                    continue # checking only one of them suffices under the assumption that the function has one local max and is otherwise monotonincally increasing/decreasing
            # none of the neighbours has higher utility (or there are no feasible neighbours), so we are at the local max
            solution_found = True

        num_pools, margins = t, margins_t
        utility = 0
        strategy = None
        if num_pools > 0:
            owned_pools_copies = self.determine_pools_to_keep(num_pools)
            strategy = self.find_operator_move(num_pools, owned_pools_copies, margins)
            utility = self.calculate_expected_utility(strategy) # recalculating utility to account for possible delegations
        return utility, strategy


    def calculate_margins_and_utility(self, num_pools):
        cost_per_pool = hlp.calculate_cost_per_pool(num_pools, self.cost, self.model.cost_factor)
        pledge_per_pool = hlp.determine_pledge_per_pool(self.stake, self.model.beta, num_pools)
        potential_profit_per_pool = hlp.calculate_potential_profit(pledge_per_pool, cost_per_pool, self.model.alpha,
                                                                   self.model.beta, self.model.reward_function_option,
                                                                   self.model.total_stake)
        boost = 1e-6  # to ensure that the new desirability will be higher than the target one #todo tune boost
        margins = [] # note that pools by the same agent may end up with different margins  because of the different pools they aim to outperform
        utility = 0
        for t in range(1, num_pools+1):
            target_pool = self.rankings[self.model.k - t] #todo what if that's own pool???
            target_desirability, target_pp =  (target_pool.desirability, target_pool.potential_profit) if target_pool is not None else (0, 0)
            target_desirability += boost
            if potential_profit_per_pool < target_desirability:
                # can't reach target desirability even with zero margin, so we can assume that the pool won't be in the top k
                margins.append(0)
                utility += hlp.calculate_operator_utility_from_pool(pool_stake=pledge_per_pool, pledge=pledge_per_pool,
                                                                    margin=0, cost=cost_per_pool, alpha=self.model.alpha,
                                                                    beta=self.model.beta,
                                                                    reward_function_option=self.model.reward_function_option,
                                                                    total_stake=self.model.total_stake)
            else:
                # potential profit > target desirability so we proceed by finding an appropriate margin
                max_target_desirability = max(target_desirability, target_pp)
                margins.append(hlp.calculate_suitable_margin(potential_profit=potential_profit_per_pool,
                                                      target_desirability=max_target_desirability))
                utility += hlp.calculate_operator_utility_from_pool(pool_stake=self.model.beta, pledge=pledge_per_pool,
                                                                    margin=margins[-1],
                                                                    cost=cost_per_pool, alpha=self.model.alpha,
                                                                    beta=self.model.beta,
                                                                    reward_function_option=self.model.reward_function_option,
                                                                    total_stake=self.model.total_stake)
        return margins, utility

    def calculate_margin(self, pool): #todo add test
        """
        The agent ranks all existing pools based on their potential
        profit and chooses a margin that can guarantee the pool's desirability (non-zero only if the pool
        ranks in the top k)
        :return: float, the margin that the agent will set for this pool
        """
        if pool.is_private:
            return 0

        reference_pool = self.rankings[self.model.k-1]
        reference_potential_profit = reference_pool.potential_profit if reference_pool is not None else 0
        margin = max(1 - (reference_potential_profit / pool.potential_profit), 0)
        return margin

    def determine_pools_to_keep(self, num_pools_to_keep):
        if num_pools_to_keep < len(self.strategy.owned_pools):
            # Only keep the pool(s) that rank best (based on desirability, potential profit, stake and "age")
            owned_pools_to_keep = dict()
            #todo just check pool rankings instead
            pool_properties = [(pool.desirability, pool.potential_profit, pool.stake, -pool_id) for pool_id, pool in self.strategy.owned_pools.items()]
            top_pools_ids = {-p[3] for p in heapq.nlargest(num_pools_to_keep, pool_properties)}
            for pool_id in top_pools_ids:
                owned_pools_to_keep[pool_id] = deepcopy(self.strategy.owned_pools[pool_id])
        else:
            owned_pools_to_keep = deepcopy(self.strategy.owned_pools)
        return owned_pools_to_keep

    def find_operator_moves(self):
        """
        When agents are allowed to operate multiple pools, they try out different options to decide
        exactly how many pools to operate. Specifically, they always consider running any number of pools up to their
        current number (i.e. closing some pools or keeping the same number) and they also consider opening one
        extra pool. In case an operator has recently opened a new pool, they are not allowed to close any currently open
        pools, so they only consider keeping the same number of pools or adding one to it.
        :return:
        """
        moves = dict()
        max_new_pools_per_round = 1
        current_num_pools = len(self.strategy.owned_pools)
        if self.model.pool_splitting:
            if current_num_pools == 0:
                # If the agent hasn't made any pools yet, consider operating up to as many pools as they can saturate with pledge + 1
                num_pools_options = {i for i in range(1, math.ceil(self.stake / self.model.beta) + 1)}
            else:
                num_pools_options = {max(i, 1) for i in
                                     range(current_num_pools, current_num_pools + max_new_pools_per_round + 1)}
                if self.remaining_min_steps_to_keep_pool <= 0:
                    num_pools_options.update(range(1, current_num_pools))
        else:
            # If pool splitting is not allowed by the model, there are no options
            num_pools_options = {1}

        for num_pools in num_pools_options:
            owned_pools_copies = self.determine_pools_to_keep(num_pools)
            moves[num_pools] = self.find_operator_move(num_pools, owned_pools_copies)
        return moves

    def find_operator_move(self, num_pools, owned_pools, margins=[]):
        pledge = hlp.determine_pledge_per_pool(agent_stake=self.stake, beta=self.model.beta, num_pools=num_pools)
        cost_per_pool = hlp.calculate_cost_per_pool(num_pools=num_pools, initial_cost=self.cost, cost_factor=self.model.cost_factor)

        for i, (pool_id, pool) in enumerate(owned_pools.items()):
            # For pools that already exist, modify them to match the new strategy
            pool.stake -= pool.pledge - pledge
            pool.pledge = pledge
            pool.is_private = pool.pledge >= self.model.beta
            pool.cost = cost_per_pool
            pool.set_profit(self.model.alpha, self.model.beta, self.model.reward_function_option, self.model.total_stake)
            pool.margin = margins[i] if len(margins) > i  else self.calculate_margin(pool)

        existing_pools_num = len(owned_pools)
        for i in range(existing_pools_num, num_pools):
            # For pools under consideration of opening, create according to the strategy
            pool_id = self.model.get_next_pool_id()
            # todo maybe use a temp pool id here and assign final id at execution
            pool = Pool(pool_id=pool_id, cost=cost_per_pool,
                        pledge=pledge, owner=self.unique_id, alpha=self.model.alpha,
                        beta=self.model.beta, is_private=pledge >= self.model.beta,
                        reward_function_option=self.model.reward_function_option, total_stake=self.model.total_stake)
            # private pools have margin 0 but don't allow delegations
            pool.margin = margins[i] if len(margins) > i else self.calculate_margin(pool)
            owned_pools[pool_id] = pool

        allocations = self.find_delegation_for_operator(pledge*num_pools)

        return Strategy(stake_allocations=allocations, owned_pools=owned_pools)

    def find_delegation_for_operator(self, total_pledge):
        allocations = dict()
        remaining_stake = self.stake - total_pledge
        if remaining_stake > 0:
            # in some cases agents may not want to allocate their entire stake to their pool (e.g. when stake > β)
            delegation_strategy = self.find_delegation_move(stake_to_delegate=remaining_stake)
            allocations = delegation_strategy.stake_allocations
        return allocations

    def determine_stake_allocations(self, stake_to_delegate):
        """
        Choose a delegation move based on the desirability of the existing pools. If two or more pools are tied,
        choose the one with the highest potential profit, as it promises higher potential rewards (further ties are
        broken using ids, so that older pools are preferred).
        :return:
        """
        all_pools_dict = self.model.pools
        saturation_point = self.model.beta
        eligible_pools_ranked = [
            pool
            for pool in self.rankings
            if pool is not None and pool.owner != self.unique_id and not pool.is_private
        ]
        # Only proceed if there are public pools in the system that don't belong to the current agent
        if len(eligible_pools_ranked) == 0:
            return None

        # Remove the agent's stake from the pools in case it's being delegated
        for pool_id, allocation in self.strategy.stake_allocations.items():
            pool = all_pools_dict[pool_id]
            self.model.pool_rankings_myopic.remove(pool) #todo tmp find better way
            pool.update_delegation(new_delegation=0, delegator_id=self.unique_id)
            self.model.pool_rankings_myopic.add(pool)

        allocations = dict()
        best_saturated_pool = None
        while len(eligible_pools_ranked) > 0:
            # first attempt to delegate to unsaturated pools
            best_pool = eligible_pools_ranked.pop(0)
            stake_to_saturation = saturation_point - best_pool.stake
            if stake_to_saturation < MIN_STAKE_UNIT:
                if best_saturated_pool is None:
                    best_saturated_pool = best_pool
                continue
            allocation = min(stake_to_delegate, stake_to_saturation)
            stake_to_delegate -= allocation
            allocations[best_pool.id] = allocation
            if stake_to_delegate < MIN_STAKE_UNIT:
                break
        if stake_to_delegate >= MIN_STAKE_UNIT and best_saturated_pool is not None:
            # if the stake to delegate does not fit in unsaturated pools, delegate to the saturated one with the highest desirability
            allocations[best_saturated_pool.id] = stake_to_delegate

        # Return the agent's stake to the pools it was delegated to
        for pool_id, allocation in self.strategy.stake_allocations.items():
            pool = all_pools_dict[pool_id]
            self.model.pool_rankings_myopic.remove(pool)  # todo tmp find better way
            pool.update_delegation(new_delegation=allocation, delegator_id=self.unique_id)
            self.model.pool_rankings_myopic.add(pool)
        return allocations

    def find_delegation_move(self, stake_to_delegate=None):
        if stake_to_delegate is None:
            stake_to_delegate = self.stake
        if stake_to_delegate < MIN_STAKE_UNIT:
            return Strategy()

        allocations = self.determine_stake_allocations(stake_to_delegate)
        return Strategy(stake_allocations=allocations)

    def execute_strategy(self):
        """
        Execute the updated strategy of the agent
        @return:
        """
        current_pools = self.model.pools
        old_allocations = self.strategy.stake_allocations
        new_allocations = self.new_strategy.stake_allocations
        for pool_id in old_allocations.keys() - new_allocations.keys():
            pool = current_pools[pool_id]
            if pool is not None:  # todo can't really be none here, right? maybe in simultaneous act? check if else clause is needed (e.g. to update strategy)
                # remove delegation
                self.model.pool_rankings_myopic.remove(pool)
                pool.update_delegation(new_delegation=0, delegator_id=self.unique_id)
                self.model.pool_rankings_myopic.add(pool)
        for pool_id in new_allocations.keys() :
            pool = current_pools[pool_id]
            if pool is not None:  # todo can't really be none here, right? maybe in simultaneous act? check if else clause is needed (e.g. to update strategy)
                # add / modify delegation
                self.model.pool_rankings_myopic.remove(pool)
                pool.update_delegation(new_delegation=new_allocations[pool_id], delegator_id=self.unique_id)
                self.model.pool_rankings_myopic.add(pool)

        old_owned_pools = set(self.strategy.owned_pools.keys())
        new_owned_pools = set(self.new_strategy.owned_pools.keys())

        for pool_id in old_owned_pools - new_owned_pools:
            # pools have closed
            self.close_pool(pool_id)
        for pool_id in new_owned_pools & old_owned_pools:
            # updates in old pools
            current_pools[pool_id] = self.update_pool(pool_id)

        self.strategy = self.new_strategy
        self.new_strategy = None
        for pool_id in new_owned_pools - old_owned_pools:
            self.open_pool(pool_id)

    def update_pool(self, pool_id):
        updated_pool = self.new_strategy.owned_pools[pool_id]
        if updated_pool.is_private and updated_pool.stake > updated_pool.pledge:
            # undelegate stake in case the pool turned from public to private
            self.remove_delegations(updated_pool)
        self.model.pools[pool_id] = updated_pool
        # update pool rankings
        old_pool = self.strategy.owned_pools[pool_id]
        self.model.pool_rankings.remove(old_pool)
        self.model.pool_rankings.add(updated_pool)
        self.model.pool_rankings_myopic.remove(old_pool)
        self.model.pool_rankings_myopic.add(updated_pool)
        return updated_pool

    def open_pool(self, pool_id):
        pool = self.strategy.owned_pools[pool_id]
        self.model.pools[pool_id] = pool
        self.remaining_min_steps_to_keep_pool = self.model.min_steps_to_keep_pool
        # include in pool rankings
        self.model.pool_rankings.add(pool)
        self.model.pool_rankings_myopic.add(pool)

    def close_pool(self, pool_id):
        pools = self.model.pools
        pool = pools[pool_id]
        # remove from top k desirabilities
        self.model.pool_rankings.remove(pool)
        self.model.pool_rankings_myopic.remove(pool)
        # Undelegate delegators' stake
        self.remove_delegations(pool)
        pools.pop(pool_id)

    def remove_delegations(self, pool):
        agents = self.model.get_agents_dict()
        delegators = list(pool.delegators.keys())
        for agent_id in delegators:
            agent = agents[agent_id]
            agent.strategy.stake_allocations.pop(pool.id)
            pool.update_delegation(new_delegation=0, delegator_id=agent_id)

        # Also remove pool from agents' upcoming moves in case of (semi)simultaneous activation
        if "simultaneous" in self.model.agent_activation_order.lower():
            for agent in agents.values():  # todo alternatively save potential delegators somewhere so that we don't go through the whole list of agents here
                if agent.new_strategy is not None:
                    agent.new_strategy.stake_allocations.pop(pool.id, None)

    def get_status(self):
        print("Agent id: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.stake, self.cost))
        print("\n")
