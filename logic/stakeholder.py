# -*- coding: utf-8 -*-
from mesa import Agent
from copy import deepcopy
import heapq
import math

import logic.helper as hlp
from logic.pool import Pool
from logic.strategy import Strategy
from logic.helper import MIN_STAKE_UNIT
from logic.custom_exceptions import PoolNotFoundError


class Stakeholder(Agent):

    def __init__(self, unique_id, model, stake, cost, is_myopic=False, is_abstainer=False, strategy=None):
        super().__init__(unique_id, model)
        self.cost = cost  # the cost of running one pool for this agent
        self.stake = stake
        self.isMyopic = is_myopic
        self.abstains = is_abstainer #todo determine if still necessary with current definition of abstention rate
        self.remaining_min_steps_to_keep_pool = 0
        self.new_strategy = None

        if not is_abstainer:
            if strategy is None:
                # Initialise strategy to an "empty" strategy for agents that don't abstain
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

    def update_strategy(self):
        if not self.abstains:
            self.make_move()

    def advance(self):
        if self.new_strategy is not None:
            # The agent has changed their strategy, so now they have to execute it
            self.execute_strategy()
            self.model.current_step_idle = False

    def make_move(self):
        current_utility = self.calculate_expected_utility(self.strategy)
        current_utility_with_inertia = max(
            (1 + self.model.relative_utility_threshold) * current_utility,
            current_utility + self.model.absolute_utility_threshold
        )
        # hold the agent's potential moves in a dict, where the values are tuples of (utility, strategy)
        possible_moves = {"current": (current_utility_with_inertia, self.strategy)}

        # For all agents, find a possible delegation strategy and calculate its potential utility
        # unless they are pool operators with recently opened pools (we assume that they will keep them at least for a bit)
        if self.remaining_min_steps_to_keep_pool == 0:
            delegator_strategy = self.find_delegation_move()
            delegator_utility = self.calculate_expected_utility(delegator_strategy)
            possible_moves["delegator"] = delegator_utility, delegator_strategy
        if self.model.pool_opening_process == 'plus-one':
            if len(self.strategy.owned_pools) > 0 or self.has_potential_for_pool():
                # agent is considering opening a pool, so he has to find the most suitable pool params
                # and calculate the potential utility of operating a pool with these params
                possible_moves["operator"] = self.choose_pool_strategy_plus_one()
        elif self.model.pool_opening_process == 'local-search':
            pool_strategy = self.choose_pool_strategy_local_search()
            if pool_strategy[1] is not None:
                possible_moves["operator"] = pool_strategy
        else:
            #todo change error handling
            print('helllooooo')
            print(self.model.pool_opening_process)

        # Compare the above with the utility of the current strategy and pick one of the 3
        # in case of a tie, the max function picks the element with the lowest index, so we have strategically ordered
        # them earlier so that the "easiest" move is preferred ( current -> delegator -> operator)
        max_utility_option = max(possible_moves, key=lambda key: possible_moves[key][0])
        #todo maybe discard temp pool ids here
        '''if max_utility_option == "operator":
            hlp.plot_margin_pools_heatmap(self)'''
        self.new_strategy = None if max_utility_option == "current" else possible_moves[max_utility_option][1]


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
            target_desirability, target_pp = self.model.pool_desirabilities_n_pps[self.model.k - t] #todo what if that's own pool???
            target_desirability += boost
            if potential_profit_per_pool < target_desirability:
                # can't reach target desirability even with zero margin, so we can assume that the pool won't be in the top k
                margins.append(0)
                utility += hlp.calculate_operator_utility_from_pool(non_myopic_pool_stake=pledge_per_pool, pledge=pledge_per_pool,
                                                                       margin=0, cost=cost_per_pool, alpha=self.model.alpha,
                                                                       beta=self.model.beta,
                                                                       reward_function_option=self.model.reward_function_option,
                                                                       total_stake=self.model.total_stake)
            else:
                # pp > target desirability so we proceed by finding an appropriate margin
                max_target_desirability = max(target_desirability, target_pp)
                margins.append(hlp.calculate_suitable_margin(potential_profit=potential_profit_per_pool,
                                                      target_desirability=max_target_desirability))
                utility += hlp.calculate_operator_utility_from_pool(non_myopic_pool_stake=self.model.beta, pledge=pledge_per_pool,
                                                                               margin=margins[-1],
                                                                               cost=cost_per_pool, alpha=self.model.alpha,
                                                                               beta=self.model.beta,
                                                                               reward_function_option=self.model.reward_function_option,
                                                                               total_stake=self.model.total_stake)
        return margins, utility


    def choose_pool_strategy_local_search(self):
        """
        Find a suitable pool operation strategy by using the following process:
            - Start with an arbitrary number of pools t (we set this to k/2)
            - Calculate a suitable margin so that all t pools end up in the top k (if not possible then set margin to 0)
            - Calculate the agent's utility for this number of pools and margin
            - Do the same for the two neighbours of this strategy, i.e. operating t-1 pools and t+1 pools
            - Choose the direction with the highest utility and make a jump in t
            - If none of the neighbours have higher utility, solution found (strategy with t pools)
        This works because the utility of an agent as a function of the number of pools to operate has only one local max
        @return:
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


    def choose_pool_strategy_plus_one(self):
        operator_strategies = self.find_operator_moves()
        max_operator_utility = 0
        max_operator_strategy = None
        for num_pools in sorted(operator_strategies.keys()):
            operator_strategy = operator_strategies[num_pools]
            operator_utility = self.calculate_expected_utility(operator_strategy)
            if operator_utility > max_operator_utility:
                max_operator_utility = operator_utility
                max_operator_strategy = operator_strategy
        return max_operator_utility, max_operator_strategy

    def discard_draft_pools(self, operator_strategy):
        # todo problem: if there are many different pool strategies developed, should we discard drafts from all of them?
        # Discard the pool ids that were used for the hypothetical operator move
        old_owned_pools = set(self.strategy.owned_pools.keys())
        hypothetical_owned_pools = set(operator_strategy.owned_pools.keys())
        self.model.rewind_pool_id_seq(step=len(hypothetical_owned_pools - old_owned_pools))

    def calculate_expected_utility(self, strategy):
        utility = 0

        # Calculate expected utility of operating own pools
        if len(strategy.owned_pools) > 0:
            utility += self.calculate_operator_utility_from_strategy(strategy)

        # Calculate expected utility of delegating to other pools
        pools = self.model.pools
        for pool_id, a in strategy.stake_allocations.items():
            if a <= 0:
                continue
            if pool_id in pools:
                pool = pools[pool_id]
                utility += self.calculate_delegator_utility_from_pool(pool, a)
            else:
                raise PoolNotFoundError("Agent {} considered delegating to a non-existing pool ({})!"
                                        .format(self.unique_id, pool_id))
        return utility

    def calculate_operator_utility_from_strategy(self, strategy):
        utility = 0
        potential_pools = strategy.owned_pools
        fixed_pools = {pool_id: pool for pool_id, pool in self.model.pools.items() if pool.owner != self.unique_id}
        all_considered_pools = fixed_pools | potential_pools
        for pool in potential_pools.values():
            pool_utility = self.calculate_operator_utility_from_pool(pool, all_considered_pools)
            utility += pool_utility
        return utility

    def calculate_operator_utility_from_pool(self, pool, all_pools):
        '''pool_stake = pool.stake if self.isMyopic else hlp.calculate_pool_stake_NM(pool.id,
                                                                                  all_pools,
                                                                                  beta,
                                                                                  self.model.k
                                                                                  )'''  # assuming there is no myopic play for pool owners
        pool_stake = hlp.calculate_pool_stake_NM(
            pool.id,
            all_pools,
            self.model.beta,
            self.model.k
        )

        return hlp.calculate_operator_utility_from_pool(non_myopic_pool_stake=pool_stake, pledge=pool.pledge, margin=pool.margin,
                                                        cost=pool.cost, alpha=self.model.alpha, beta=self.model.beta, reward_function_option=self.model.reward_function_option, total_stake= self.model.total_stake)

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
        relative_pool_stake = pool_stake / self.model.total_stake
        relative_pledge = pool.pledge / self.model.total_stake
        r = hlp.calculate_pool_reward(relative_pool_stake, relative_pledge, alpha, beta, self.model.reward_function_option, self.model.total_stake)

        q = stake_allocation / pool_stake
        return hlp.calculate_delegator_reward_from_pool(pool_margin=pool.margin, pool_cost=pool.cost, pool_reward=r, delegator_stake_fraction=q)

    # how does a myopic agent decide whether to open a pool or not? -> for now we assume that all agents play non-myopically when it comes to pool moves
    def has_potential_for_pool(self):
        """
        Determine whether the agent is at a good position to open a pool, using the following rules:
        If the current pools are not enough to cover the total active stake of the system without getting oversaturated
        then having a positive potential profit is a necessary and sufficient condition for a agent to open a pool
        (as it means that there is stake that is forced to remain undelegated
        or delegated to an oversaturated pool that yields suboptimal rewards)
        Note that the total active stake is estimated by the stakeholders at predetermined intervals and may not be
        100% accurate

        If there are enough pools to cover all active agents' stake without causing oversaturation,
        then the agent only opens a pool if the maximum possible desirability of their pool
        (aka the potential profit) is higher than the desirability of at least one currently active pool
        (with the perspective of "stealing" the delegators from that pool)

        :return: bool true if the agent has potential to open pool based on the above rules, false otherwise
        """
        saturation_point = self.model.beta
        alpha = self.model.alpha
        current_pools = self.model.get_pools_list()

        potential_profit = hlp.calculate_potential_profit(self.stake, self.cost, alpha, saturation_point, self.model.reward_function_option, self.model.total_stake)
        if len(current_pools) * saturation_point < self.model.perceived_active_stake:  # note that we use active stake instead of total stake
            return potential_profit > 0

        existing_desirabilities = [pool.desirability for pool in current_pools]
        # Note that the potential profit is equal to the desirability of a pool with 0 margin,
        # so, effectively, the agent is comparing his best-case desirability with the desirabilities of the current pools
        return potential_profit > 0 and any(desirability < potential_profit for desirability in existing_desirabilities)

    def calculate_margin(self, pool):
        """
        Inspired by "perfect strategies", the agent ranks all existing pools based on their potential
        profit and chooses a margin that can guarantee the pool's desirability (non-zero only if the pool
        ranks in the top k)
        :return: float, the margin that the agent will set for this pool
        """
        all_pools = self.model.pools
        temp_pool = None
        if pool.id in all_pools:
            temp_pool = all_pools[pool.id]
        all_pools[pool.id] = pool

        potential_profits = [pool.potential_profit for pool in all_pools.values()]
        k = self.model.k
        top_potential_profits = heapq.nlargest(k+1, potential_profits)#todo is this a bottleneck?

        # find the pool that is ranked at position k+1, if such pool exists
        reference_potential_profit = top_potential_profits[-1]
        margin = max(1 - (reference_potential_profit / pool.potential_profit), 0)

        # make sure that model fields are left intact
        if temp_pool is None:
            all_pools.pop(pool.id)
        else:
            all_pools[pool.id] = temp_pool

        return margin

    def calculate_margin_perfect_strategy(self):
        """
        Based on "perfect strategies", the agent ranks all agents based on their potential
        profit of operating one pool and chooses a margin that can keep his pool competitive
        :return: float, the margin that the agent will use for their pool(s)
        """
        # first calculate the potential profits of all agents
        agents = self.model.get_agents_list()
        potential_profits = [hlp.calculate_potential_profit(agent.stake, agent.cost, self.model.alpha, self.model.beta,
                                                            self.model.reward_function_option, self.model.total_stake)
                             for agent in agents]
        current_agent_pp = hlp.calculate_potential_profit(self.stake, self.cost, self.model.alpha, self.model.beta,
                                                           self.model.reward_function_option, self.model.total_stake)
        k = self.model.k
        n = len(agents)
        if k < n:
            top_potential_profits = heapq.nlargest(k + 1, potential_profits)
            # find the pool that is ranked at position k+1, if such pool exists
            reference_potential_profit = top_potential_profits[-1]
        else:
            reference_potential_profit = 0

        # todo possible improvement: keep track of all agents who have opened pools and their potential profits
        margin = max(1 - (reference_potential_profit / current_agent_pp), 0)
        return margin

    def determine_pools_to_keep(self, num_pools_to_keep):
        if num_pools_to_keep < len(self.strategy.owned_pools):
            # Only keep the pool(s) that rank best (based on desirability, potential profit, stake and "age")
            owned_pools_to_keep = dict()
            # todo do I need to calculate myopic desirability in case of myopic agent?
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
        max_new_pools_per_round = 10
        current_num_pools = len(self.strategy.owned_pools)
        if self.model.pool_splitting:
            num_pools_options = {max(i,1) for i in range(current_num_pools, current_num_pools + max_new_pools_per_round + 1)}
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
            pool.set_potential_profit(self.model.alpha, self.model.beta, self.model.reward_function_option, self.model.total_stake)
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

    def find_delegation_move(self, stake_to_delegate=None):
        """
        Choose a delegation move based on the desirability of the existing pools. If two or more pools are tied,
        choose the one with the highest (current) stake, as it offers higher short-term rewards.
        :return:
        """
        if stake_to_delegate is None:
            stake_to_delegate = self.stake
        if stake_to_delegate < MIN_STAKE_UNIT:
            return Strategy()

        all_pools_dict = self.model.pools
        saturation_point = self.model.beta
        # todo do I need to calculate myopic desirability in case of myopic agent?
        relevant_pools_properties = [
            (
                -pool.desirability,
                -pool.potential_profit,
                -pool.stake,
                pool.id
            )
               for pool in all_pools_dict.values()
               if pool.owner != self.unique_id and not pool.is_private
        ]
        # Only proceed if there are public pools in the system that don't belong to the current agent
        if len(relevant_pools_properties) == 0:
            return Strategy()

        heapq.heapify(relevant_pools_properties) # turn list into (min) heap

        # Remove the agent's stake from the pools in case it's being delegated
        for pool_id, allocation in self.strategy.stake_allocations.items():
            all_pools_dict[pool_id].update_delegation(new_delegation=0, delegator_id=self.unique_id)

        allocations = dict()
        best_saturated_pool = None
        while len(relevant_pools_properties) > 0:
            # first attempt to delegate to unsaturated pools
            best_pool_id = heapq.heappop(relevant_pools_properties)[3]
            best_pool = all_pools_dict[best_pool_id]
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
            all_pools_dict[pool_id].update_delegation(new_delegation=allocation, delegator_id=self.unique_id)

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
            if current_pools[pool_id] is not None:  # todo can't really be none here, right? maybe in simultaneous act? check if else clause is needed (e.g. to update strategy)
                # remove delegation
                current_pools[pool_id].update_delegation(new_delegation=0, delegator_id=self.unique_id)
        for pool_id in new_allocations.keys() :
            if current_pools[pool_id] is not None:  # todo can't really be none here, right? maybe in simultaneous act? check if else clause is needed (e.g. to update strategy)
                # add / modify delegation
                current_pools[pool_id].update_delegation(new_delegation=new_allocations[pool_id], delegator_id=self.unique_id)


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
        # update top k desirability list if needed
        pool_desirability =updated_pool.desirability
        pool_pp = updated_pool.potential_profit
        old_pool = self.strategy.owned_pools[pool_id]
        # remove the desirability of the old pool
        self.model.pool_desirabilities_n_pps.remove((old_pool.desirability, old_pool.potential_profit))
        # add desirability of updated pool
        self.model.pool_desirabilities_n_pps.add((updated_pool.desirability, updated_pool.potential_profit))
        return updated_pool

    def open_pool(self, pool_id):
        self.model.pools[pool_id] = self.strategy.owned_pools[pool_id]
        self.remaining_min_steps_to_keep_pool = self.model.min_steps_to_keep_pool
        # update top k desirability list if needed
        pool_desirability = self.strategy.owned_pools[pool_id].desirability
        pool_pp = self.strategy.owned_pools[pool_id].potential_profit
        self.model.pool_desirabilities_n_pps.add((pool_desirability, pool_pp))

    def close_pool(self, pool_id):
        pools = self.model.pools
        try:
            pool = pools[pool_id]
            if pool.owner != self.unique_id:
                raise ValueError("agent tried to close pool that belongs to another agent.")
        except KeyError:
            raise ValueError("Given pool id is not valid.")
        # Undelegate delegators' stake
        self.remove_delegations(pool)
        pools.pop(pool_id)
        # remove from top k desirabilities
        self.model.pool_desirabilities_n_pps.remove((pool.desirability, pool.potential_profit))
        if len(self.model.pool_desirabilities_n_pps) < self.model.k:
            self.model.pool_desirabilities_n_pps.add((0,0))

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
        print("Agent id: {}, is myopic: {}, stake: {}, cost:{}"
              .format(self.unique_id, self.isMyopic, self.stake, self.cost))
        print("\n")
