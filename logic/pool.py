# -*- coding: utf-8 -*-
import logic.helper as hlp
from logic.helper import MIN_STAKE_UNIT

class Pool:

    def __init__(self, pool_id, cost, pledge, owner, alpha, beta, reward_function_option, total_stake, margin=-1, is_private=False):
        self.id = pool_id
        self.cost = cost
        self.pledge = pledge
        self.stake = pledge
        self.owner = owner
        self.is_private = is_private
        self.delegators = dict()
        self.set_profit(alpha, beta, reward_function_option, total_stake)
        self.margin = margin

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, m):
        self._margin = m
        # whenever the margin changes, the pool's desirability gets automatically re-calculated
        #todo shouldn't it also change when pledge, cost / pp is changed? -> not an issue in practice because whenever pledge changes margin also changes but maybe can make it better
        self.set_desirability()

    def set_profit(self, alpha, beta, reward_function_option, total_stake):
        self.potential_profit = hlp.calculate_potential_profit(self.pledge, self.cost, alpha, beta, reward_function_option, total_stake)
        self.current_profit = hlp.calculate_current_profit(self.stake, self.pledge, self.cost, alpha, beta, reward_function_option, total_stake)

    def set_desirability(self):
        self.desirability = hlp.calculate_pool_desirability(margin=self.margin, potential_profit=self.potential_profit)
        self.myopic_desirability = hlp.calculate_myopic_pool_desirability(margin=self.margin, current_profit=self.current_profit)

    def update_delegation(self, new_delegation, delegator_id):
        if delegator_id in self.delegators:
            self.stake -= self.delegators[delegator_id]
        self.stake += new_delegation
        self.delegators[delegator_id] = new_delegation
        if self.delegators[delegator_id] < MIN_STAKE_UNIT:
            self.delegators.pop(delegator_id)



