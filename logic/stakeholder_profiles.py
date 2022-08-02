from logic.stakeholder import Stakeholder
import logic.helper as hlp

from sortedcontainers import SortedList

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
        pool_stake_nm = hlp.calculate_pool_stake_NM(
            pool,
            all_pool_rankings,
            self.model.beta,
            self.model.k
        )

        return hlp.calculate_operator_utility_from_pool(pool_stake=pool_stake_nm, pledge=pool.pledge, margin=pool.margin,
                                                        cost=pool.cost, alpha=self.model.alpha, beta=self.model.beta, reward_function_option=self.model.reward_function_option, total_stake= self.model.total_stake)


    def calculate_delegator_utility_from_pool(self, pool, stake_allocation):
        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id] \
            if pool.id in self.strategy.stake_allocations else 0
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        pool_stake = max(
            hlp.calculate_pool_stake_NM(
                pool,
                self.rankings,
                self.model.beta,
                self.model.k
            ),
            current_stake
        )
        return hlp.calculate_delegator_utility_from_pool(stake_allocation, pool_stake, pool.pledge, pool.margin, pool.cost,
                                                         self.model.alpha, self.model.beta, self.model.reward_function_option, self.model.total_stake)


class MyopicStakeholder(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, strategy=strategy)
        self.rankings = self.model.pool_rankings_myopic

    def calculate_operator_utility_from_strategy(self, strategy):  # todo can also have in parent class if I define a self.comparison_key like the rankings
        potential_pools = strategy.owned_pools.values()
        utility = 0
        for pool in potential_pools:
            pool_utility = hlp.calculate_operator_utility_from_pool(pool_stake=pool.stake, pledge=pool.pledge,
                                                        margin=pool.margin, cost=pool.cost, alpha=self.model.alpha,
                                                        beta=self.model.beta,
                                                        reward_function_option=self.model.reward_function_option,
                                                        total_stake=self.model.total_stake)
            utility += pool_utility
        return utility

    def calculate_delegator_utility_from_pool(self, pool, stake_allocation):
        previous_allocation_to_pool = self.strategy.stake_allocations[pool.id] \
            if pool.id in self.strategy.stake_allocations else 0
        current_stake = pool.stake - previous_allocation_to_pool + stake_allocation
        return hlp.calculate_delegator_utility_from_pool(stake_allocation, current_stake, pool.pledge, pool.margin,
                                                         pool.cost, self.model.alpha, self.model.beta,
                                                         self.model.reward_function_option, self.model.total_stake)


class Abstainer(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, strategy=strategy)
        self.strategy = None

    def update_strategy(self):
        return

profile_mapping = {
    0: NonMyopicStakeholder,
    1: MyopicStakeholder,
    2: Abstainer
}
