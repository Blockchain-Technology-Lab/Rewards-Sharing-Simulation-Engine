# -*- coding: utf-8 -*-
from mesa import Agent
from copy import deepcopy
import heapq

import logic.helper as hlp
from logic.pool import Pool
from logic.strategy import Strategy
from logic.helper import MIN_STAKE_UNIT
from logic.custom_exceptions import PoolNotFoundError


class Stakeholder(Agent):

    def __init__(self, unique_id, model, stake, is_myopic=False, is_abstainer=False, cost=0, strategy=None):
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

        if len(self.strategy.owned_pools) > 0 or self.has_potential_for_pool():
            # agent is considering opening a pool, so he has to find the most suitable pool params
            # and calculate the potential utility of operating a pool with these params
            possible_moves["operator"] = self.choose_pool_strategy()

        # Compare the above with the utility of the current strategy and pick one of the 3
        # in case of a tie, the max function picks the element with the lowest index, so we have strategically ordered
        # them earlier so that the "easiest" move is preferred ( current -> delegator -> operator)
        max_utility_option = max(possible_moves, key=lambda key: possible_moves[key][0])

        #todo maybe discard temp pool ids here

        self.new_strategy = None if max_utility_option == "current" else possible_moves[max_utility_option][1]

    def choose_pool_strategy(self):
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
        relative_pool_stake = pool_stake / self.model.total_stake
        relative_pledge = pledge / self.model.total_stake
        r = hlp.calculate_pool_reward(relative_pool_stake, relative_pledge, alpha, beta, self.model.reward_function_option, self.model.total_stake)
        stake_allocation = pool.pledge
        q = stake_allocation / pool_stake
        return hlp.calculate_operator_reward_from_pool(pool_margin=pool.margin, pool_cost=pool.cost, pool_reward=r, operator_stake_fraction=q)

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

    def calculate_margin_semi_perfect_strategy(self, pool):
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
        top_potential_profits = heapq.nlargest(k+1, potential_profits)

        # find the pool that is ranked at position k+1, if such pool exists
        reference_potential_profit = top_potential_profits[-1]
        # todo possible improvement: keep track of all agents who have opened pools and their potential profits
        margin = 1 - (reference_potential_profit / pool.potential_profit) if pool.potential_profit in top_potential_profits else 0

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
        max_new_pools_per_round = 1
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

    def find_operator_move(self, num_pools, owned_pools):
        pledges = hlp.determine_pledge_per_pool(agent_stake=self.stake, beta=self.model.beta, num_pools=num_pools)

        cost_per_pool = hlp.calculate_cost_per_pool_fixed_fraction(num_pools=num_pools, initial_cost=self.cost, cost_factor=self.model.cost_factor) if \
            self.model.extra_cost_type == 'fixed_fraction' else hlp.calculate_cost_per_pool(num_pools=num_pools, initial_cost=self.cost, cost_factor=self.model.cost_factor)
        for i, (pool_id, pool) in enumerate(owned_pools.items()):
            # For pools that already exist, modify them to match the new strategy
            pool.stake -= pool.pledge - pledges[i]
            pool.pledge = pledges[i]
            pool.is_private = pool.pledge >= self.model.beta
            pool.cost = cost_per_pool
            pool.set_potential_profit(self.model.alpha, self.model.beta, self.model.reward_function_option, self.model.total_stake)
            pool.margin = self.calculate_margin_semi_perfect_strategy(pool)

        existing_pools_num = len(owned_pools)
        for i in range(existing_pools_num, num_pools):
            # For pools under consideration of opening, create according to the strategy
            pool_id = self.model.get_next_pool_id()
            # todo maybe use a temp pool id here and assign final id at execution
            pool = Pool(pool_id=pool_id, cost=cost_per_pool,
                        pledge=pledges[i], owner=self.unique_id, alpha=self.model.alpha,
                        beta=self.model.beta, is_private=pledges[i] >= self.model.beta,
                        reward_function_option=self.model.reward_function_option, total_stake=self.model.total_stake)
            # private pools have margin 0 but don't allow delegations
            pool.margin = self.calculate_margin_semi_perfect_strategy(pool)
            owned_pools[pool_id] = pool

        allocations = self.find_delegation_for_operator(pledges)

        return Strategy(stake_allocations=allocations, owned_pools=owned_pools)

    def find_delegation_for_operator(self, pledges):
        allocations = dict()
        remaining_stake = self.stake - sum(pledges)
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
            updated_pool = self.new_strategy.owned_pools[pool_id]
            current_pools[pool_id] = updated_pool
            if updated_pool.is_private and updated_pool.stake > updated_pool.pledge:
                # undelegate stake in case the pool turned from public to private
                self.remove_delegations(updated_pool)

        self.strategy = self.new_strategy
        self.new_strategy = None
        for pool_id in new_owned_pools - old_owned_pools:
            self.open_pool(pool_id)

    def open_pool(self, pool_id):
        self.model.pools[pool_id] = self.strategy.owned_pools[pool_id]
        self.remaining_min_steps_to_keep_pool = self.model.min_steps_to_keep_pool

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
