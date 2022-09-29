from logic.stakeholder import Stakeholder
import logic.helper as hlp

from sortedcontainers import SortedList

#todo add required methods to parent class (and raise NotImplementedError)
class NonMyopicStakeholder(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, strategy=strategy)
        self.rankings = self.model.pool_rankings

    def calculate_operator_utility_from_strategy(self, strategy): #todo can also have in parent class if I define a self.comparison_key like the rankings
        potential_pools = strategy.owned_pools.values()
        temp_rankings = SortedList([pool for pool in self.rankings if pool is not None and pool.owner != self.unique_id], key=hlp.pool_comparison_key) #todo maybe more efficient way to copy / slice sorted list
        temp_rankings.update(potential_pools)

        utility = 0
        for pool in potential_pools:
            pool_utility = self.calculate_operator_utility_from_pool(pool, temp_rankings)
            utility += pool_utility
        return utility

    def calculate_operator_utility_from_pool(self, pool, all_pool_rankings):
        non_myopic_pool_stake = hlp.calculate_non_myopic_pool_stake(
            pool=pool,
            pool_rankings=all_pool_rankings,
            reward_scheme=self.model.reward_scheme,
            total_stake=self.model.total_stake
        )

        return hlp.calculate_operator_utility_from_pool(
            pool_stake=non_myopic_pool_stake, pledge=pool.pledge, margin=pool.margin, cost=pool.cost,
            reward_scheme=self.model.reward_scheme
        )


    def calculate_delegator_utility_from_pool(self, pool, stake_allocation):
        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id] \
            if pool.id in self.strategy.stake_allocations else 0
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        pool_stake = max(
            hlp.calculate_non_myopic_pool_stake(
                pool=pool,
                pool_rankings=self.rankings,
                reward_scheme=self.model.reward_scheme,
                total_stake=self.model.total_stake
            ),
            current_stake
        )
        return hlp.calculate_delegator_utility_from_pool(
            stake_allocation, pool_stake, pool.pledge, pool.margin, pool.cost, self.model.reward_scheme
        )
    def calculate_margins_and_utility(self, num_pools):
        cost_per_pool = self.calculate_cost_per_pool(num_pools)
        pledge_per_pool = self.determine_pledge_per_pool(num_pools)
        pool_saturation_threshold = self.model.reward_scheme.get_pool_saturation_threshold(pledge_per_pool)
        potential_profit_per_pool = hlp.calculate_potential_profit(
            reward_scheme=self.model.reward_scheme, pledge=pledge_per_pool, cost=cost_per_pool
        )
        boost = 1e-6  # to ensure that the new desirability will be higher than the target one #todo tune boost
        margins = [] # note that pools by the same agent may end up with different margins  because of the different pools they aim to outperform
        utility = 0

        fixed_pools_ranked = [
            pool
            for pool in self.rankings
            if pool is None or pool.owner != self.unique_id
        ]

        for t in range(1, num_pools+1):
            target_pool = fixed_pools_ranked[self.model.reward_scheme.k - t] # todo remove dependency from k to accommodate broader class of reward schemes
            target_desirability, target_pp =  (target_pool.desirability, target_pool.potential_profit) \
                if target_pool is not None else (0, 0)
            target_desirability += boost
            if potential_profit_per_pool < target_desirability:
                # can't reach target desirability even with zero margin, so we can assume that the pool won't be in the top k
                margins.append(0)
                utility += hlp.calculate_operator_utility_from_pool(
                    pool_stake=pledge_per_pool, pledge=pledge_per_pool, margin=0, cost=cost_per_pool,
                    reward_scheme=self.model.reward_scheme
                )
            else:
                # the pool has potential to surpass the target desirability so we proceed by finding an appropriate margin
                # as the agent is non myopic they try to surpass the target pool's potential profit
                # (which is its expected desirability in the long-term) rather than its current desirability
                max_target_desirability = max(target_desirability, target_pp)
                margins.append(hlp.calculate_suitable_margin(potential_profit=potential_profit_per_pool,
                                                      target_desirability=max_target_desirability))
                utility += hlp.calculate_operator_utility_from_pool(
                    pool_stake=pool_saturation_threshold, pledge=pledge_per_pool, margin=margins[-1],
                    cost=cost_per_pool, reward_scheme=self.model.reward_scheme
                )
        return margins, utility

class MyopicStakeholder(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, strategy=strategy)
        self.rankings = self.model.pool_rankings_myopic

    def calculate_operator_utility_from_strategy(self, strategy):  # todo can also have in parent class if I define a self.comparison_key like the rankings
        potential_pools = strategy.owned_pools.values()
        utility = 0
        for pool in potential_pools:
            pool_utility = hlp.calculate_operator_utility_from_pool(
                pool_stake=pool.stake, pledge=pool.pledge, margin=pool.margin, cost=pool.cost, reward_scheme=self.model.reward_scheme
            )
            utility += pool_utility
        return utility

    def calculate_delegator_utility_from_pool(self, pool, stake_allocation):
        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id] \
            if pool.id in self.strategy.stake_allocations else 0
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        return hlp.calculate_delegator_utility_from_pool(
            stake_allocation, current_stake, pool.pledge, pool.margin, pool.cost, self.model.reward_scheme
        )

    def calculate_margins_and_utility(self, num_pools):
        cost_per_pool = self.calculate_cost_per_pool(num_pools)
        pledge_per_pool = self.determine_pledge_per_pool(num_pools)
        pool_saturation_threshold = self.model.reward_scheme.get_pool_saturation_threshold(pledge_per_pool)

        agent_total_delegated_stake = max(sum([pool.stake for pool in self.strategy.owned_pools.values()]), self.stake)
        expected_stake_per_pool = agent_total_delegated_stake / num_pools
        profit_per_pool = hlp.calculate_current_profit(
            expected_stake_per_pool, pledge_per_pool, cost_per_pool, self.model.reward_scheme
        )
        boost = 1e-6  # to ensure that the new desirability will be higher than the target one #todo tune boost
        margins = [] # note that pools by the same agent may end up with different margins  because of the different pools they aim to outperform
        utility = 0

        fixed_pools_ranked = [
            pool
            for pool in self.rankings
            if pool is None or pool.owner != self.unique_id
        ]

        for t in range(1, num_pools+1):
            target_pool = fixed_pools_ranked[self.model.reward_scheme.k - t] # todo remove dependency from k to accommodate broader class of reward schemes
            if target_pool is None:
                target_desirability = 0
            else:
                target_pool_current_profit = hlp.calculate_current_profit(
                    target_pool.stake, target_pool.pledge, target_pool.cost, self.model.reward_scheme
                )
                target_desirability = hlp.calculate_myopic_pool_desirability(target_pool.margin, target_pool_current_profit)
            target_desirability += boost

            margins.append(
                hlp.calculate_suitable_margin(
                    potential_profit=profit_per_pool, target_desirability=target_desirability
                )
            )
            utility += hlp.calculate_operator_utility_from_pool(
                pool_stake=pool_saturation_threshold, pledge=pledge_per_pool, margin=margins[-1],
                cost=cost_per_pool, reward_scheme=self.model.reward_scheme
            )
        return margins, utility

class Abstainer(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, strategy=strategy)
        self.strategy = None

    def update_strategy(self):
        return

PROFILE_MAPPING = {
    0: NonMyopicStakeholder,
    1: MyopicStakeholder,
    2: Abstainer
}
