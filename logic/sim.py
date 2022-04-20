# -*- coding: utf-8 -*-

import csv
import time
import pathlib
import math
import collections
import sys
import random

#import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler, SimultaneousActivation, RandomActivation

from logic.stakeholder import Stakeholder
from logic.activations import SemiSimultaneousActivation
import logic.helper as hlp
from logic.model_reporters import *

AdjustableParams = collections.namedtuple("AdjustableParams", [
    'k',
    'alpha',
    'myopic_fraction',
    'relative_utility_threshold',
    'absolute_utility_threshold',
    'cost_factor'
])


class Simulation(Model):
    """
    Simulation of staking behaviour in Proof-of-Stake Blockchains.
    """

    agent_activation_orders = {
        "Random": RandomActivation,
        "Sequential": BaseScheduler,
        "Simultaneous": SimultaneousActivation, # note that during simultaneous activation agents apply their moves sequentially which may not be the expected behaviour
        "Semisimultaneous": SemiSimultaneousActivation

    }

    def __init__(self, n=1000, k=100, alpha=0.3, stake_distr_source='Pareto', myopic_fraction=0, abstention_rate=0,
                 abstention_known=False, relative_utility_threshold=0, absolute_utility_threshold=1e-9,
                 min_steps_to_keep_pool=5, pool_splitting=True, seed=None, pareto_param=2.0, max_iterations=1000,
                 cost_min=1e-4, cost_max=1e-3, cost_factor=0.7, agent_activation_order="Random", total_stake=-1, ms=10,
                 extra_cost_type='fixed_fraction', reward_function_option=0, execution_id=''):
        # todo make sure that the input is valid? n > 0, 0 < k <= n
        self.arguments = locals()  # only used for naming the output files appropriately

        if execution_id == '' or execution_id == 'temp':
            # No identifier was provided by the user, so we construct one based on the simulation's parameter values
            execution_id = hlp.generate_execution_id(self.arguments)
        seed = str(seed) # to maintain consistency among seeds, because command line arguments are parsed as strings
        if seed == 'None':
            seed = str(random.randint(0, 9999999))
        execution_id += '-seed-' + seed

        super().__init__(seed=seed)
        #print([f"{k}, {v}" for k, v in locals().items()])

        # An era is defined as a time period during which the parameters of the model don't change
        self.current_era = 0
        total_eras = 1

        adjustable_params = AdjustableParams(
            k=[int(k_value) for k_value in k] if isinstance(k, list) else [int(k)],
            alpha=alpha if isinstance(alpha, list) else [alpha],
            cost_factor=cost_factor if isinstance(cost_factor, list) else [cost_factor],
            relative_utility_threshold=relative_utility_threshold if isinstance(relative_utility_threshold, list)else [
                relative_utility_threshold],
            absolute_utility_threshold=absolute_utility_threshold if isinstance(absolute_utility_threshold, list) else [
                absolute_utility_threshold],
            myopic_fraction=myopic_fraction if isinstance(myopic_fraction, list) else [myopic_fraction]
        )

        for attr_name, attr_values_list in zip(adjustable_params._fields, adjustable_params):
            setattr(self, attr_name, attr_values_list[self.current_era])
            if len(attr_values_list) > total_eras:
                total_eras = len(attr_values_list)
        self.total_eras = total_eras
        self.adjustable_params = adjustable_params

        self.n = int(n)
        self.abstention_rate = abstention_rate
        self.min_steps_to_keep_pool = min_steps_to_keep_pool
        self.pool_splitting = pool_splitting
        self.max_iterations = max_iterations
        self.agent_activation_order = agent_activation_order
        self.extra_cost_type = extra_cost_type
        self.reward_function_option = reward_function_option

        if abstention_known:
            # The system is aware of the abstention rate of the system, so it inflates k (and subsequently lowers beta)
            # to make it possible to end up with the original desired number of pools
            self.k = int(self.k / (1 - self.abstention_rate))

        self.running = True  # for batch running and visualisation purposes
        self.schedule = self.agent_activation_orders[agent_activation_order](self)

        total_stake = self.initialize_agents(cost_min, cost_max, pareto_param, stake_distr_source.lower(), imposed_total_stake=total_stake, seed=seed)
        self.total_stake = total_stake / (1 - abstention_rate)
        print("Total stake (including abstaining fraction): ", self.total_stake)

        if self.total_stake <= 0:
            raise ValueError('Total stake must be > 0')
        self.perceived_active_stake = self.total_stake
        self.beta = self.total_stake / self.k
        self.execution_id = execution_id

        # generate file that describes the state of the system at step 0
        filename = "initial-states.csv" # aka system prior?
        '''data = {
            'n': [self.n],
            'k': [self.k],
            'alpha': [self.alpha],
            'Total stake': [self.total_stake],
            'Nakamoto coeff': [get_nakamoto_coefficient(self)],
            '# cost efficient': [get_cost_efficient_count(self)],
            'execution id': [self.execution_id]
        }
        df = pd.DataFrame(data)'''
        # n, k, alpha, total stake, NC, #cost efficient agents,  execution_id
        row = [self.n, self.k, self.alpha, self.total_stake, get_nakamoto_coefficient(self), get_cost_efficient_count(self), self.execution_id]
        with open(filename, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        self.consecutive_idle_steps = 0  # steps towards convergence
        self.current_step_idle = True
        self.min_consecutive_idle_steps_for_convergence = max(min_steps_to_keep_pool + 1, ms)
        self.pools = dict()
        self.revision_frequency = 10  # defines how often active stake and expected #pools are revised
        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run

        # only include reporters that are needed for every STEP here
        self.datacollector = DataCollector(
            model_reporters={
                "#Pools": get_number_of_pools,
                "Stake per entity": get_pool_stakes_by_agent,
                "PoolSizes": get_pool_sizes,
                "MaxPoolsPerAgent": get_max_pools_per_operator,
                #"PoolSizesByAgent": get_pool_sizes_by_agent,
                #"PoolSizesByPool": get_pool_sizes_by_pool,
                #"DesirabilitiesByAgent": get_desirabilities_by_agent,
                #"DesirabilitiesByPool": get_desirabilities_by_pool,
                "StakePairs": get_stakes_n_margins,
                "AvgPledge": get_avg_pledge,
                "TotalPledge": get_total_pledge,
                #"MedianPledge": get_median_pledge,
                #"MeanAbsDiff": get_controlled_stake_mean_abs_diff,
                #"StatisticalDistance": get_controlled_stake_distr_stat_dist,
                "NakamotoCoefficient": get_nakamoto_coefficient,
                # "NCR": get_NCR,
                # "MinAggregatePledge": get_min_aggregate_pledge,
                # "PledgeRate": get_pledge_rate,
                #"AreaCoverage": get_homogeneity_factor,
                #"AvgMargin": get_avg_margin,
                #"MedianMargin": get_median_margin,
                "AvgStkRank": get_avg_stk_rnk,
                "AvgCostRank": get_avg_cost_rnk
            })

        self.start_time = time.time()
        self.equilibrium_steps = []
        self.pivot_steps = []

    def initialize_agents(self, cost_min, cost_max, pareto_param, stake_distr_source, imposed_total_stake, seed):
        if stake_distr_source == 'file':
            stake_distribution = hlp.read_stake_distr_from_file(num_agents=self.n)
        elif stake_distr_source == 'pareto':
            # Allocate stake to the agents, sampling from a Pareto distribution
            stake_distribution = hlp.generate_stake_distr_pareto(num_agents=self.n, pareto_param=pareto_param, seed=seed,
                                                                 total_stake=imposed_total_stake)#, truncation_factor=self.k)
        elif stake_distr_source == 'flat':
            # Distribute the total stake of the system evenly to all agents
            stake_distribution = hlp.generate_stake_distr_flat(num_agents=self.n, total_stake= self.n)#max(imposed_total_stake, 1))
        else:
            raise ValueError("Unsupported stake distribution source '{}'.".format(stake_distr_source))
        total_stake = sum(stake_distribution)
        print("Total stake: ", total_stake)
        print("Max stake: ", max(stake_distribution))

        # Allocate cost to the agents, sampling from a uniform distribution
        cost_distribution = hlp.generate_cost_distr_unfrm(num_agents=self.n, low=cost_min, high=cost_max, seed=seed)
        #cost_distribution = hlp.generate_cost_distr_bands(num_agents=self.n, low=cost_min, high=cost_max, num_bands=10)
        #cost_distribution = hlp.generate_cost_distr_nrm(num_agents=self.n, low=cost_min, high=cost_max, mean=5e-6, stddev=5e-1)

        num_myopic_agents = int(self.myopic_fraction * self.n)
        unique_ids = [i for i in range(self.n)]
        self.random.shuffle(unique_ids)
        # Create agents
        for i, unique_id in enumerate(unique_ids):
            agent = Stakeholder(
                unique_id=unique_id,
                model=self,
                is_abstainer=False,
                is_myopic=(i < num_myopic_agents),
                cost=cost_distribution[i],
                stake=stake_distribution[i]
            )
            self.schedule.add(agent)
        return total_stake

    def initialise_pool_id_seq(self):
        self.id_seq = 0

    def get_next_pool_id(self):
        self.id_seq += 1
        return self.id_seq

    def rewind_pool_id_seq(self, step=1):
        self.id_seq -= step

    def step(self):
        """
        Execute one step of the simulation
        """
        self.get_status()
        self.datacollector.collect(self)

        current_step = self.schedule.steps
        if current_step >= self.max_iterations:
            self.wrap_up_execution()
            return
        if current_step % self.revision_frequency == 0 and current_step > 0:
            self.revise_beliefs()

        # Activate all agents (in the order specified by self.schedule) to perform all their actions for one time step
        self.schedule.step()
        if self.current_step_idle:
            self.consecutive_idle_steps += 1
            if self.has_converged():
                self.equilibrium_steps.append(current_step - self.min_consecutive_idle_steps_for_convergence + 1)
                if self.current_era < self.total_eras - 1:
                    self.adjust_params()
                else:
                    self.wrap_up_execution()
                    return
        else:
            self.consecutive_idle_steps = 0
        self.current_step_idle = True

    def run_model(self):
        """
        Execute multiple steps of the simulation, until it converges or a maximum number of iterations is reached
        :return:
        """
        self.start_time = time.time()
        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run
        while self.schedule.steps <= self.max_iterations and self.running:
            self.step()

    def has_converged(self):
        """
        Check whether the system has reached a state of equilibrium,
        where no agent wants to change their strategy
        """
        return self.consecutive_idle_steps >= self.min_consecutive_idle_steps_for_convergence

    def export_agents_file(self):
        row_list = [["Agent id", "Stake", "Cost", "Potential Profit","Status"]]
        agents = self.get_agents_dict()
        decimals = 5
        row_list.extend([
            [agent_id, round(agents[agent_id].stake, decimals), round(agents[agent_id].cost, decimals),
             round(hlp.calculate_potential_profit(agents[agent_id].stake, agents[agent_id].cost, self.alpha, self.beta, self.reward_function_option, self.total_stake), decimals),
             "Abstainer" if agents[agent_id].strategy is None else "Operator" if len(agents[agent_id].strategy.owned_pools) > 0 else "Delegator"
             ] for agent_id in range(len(agents))
        ])

        suffix = '-final_configuration_stakeholders.csv' if self.has_converged() else '-intermediate_configuration_stakeholders.csv'
        filename = self.execution_id + suffix

        hlp.export_csv_file(row_list, filename)
        
    def export_pools_file(self):
        row_list = [["Pool id", "Owner id", "Owner stake", "Pool Pledge", "Pool stake", "Owner cost", "Pool cost", "Pool margin"]]
        agents = self.get_agents_dict()
        pools = self.get_pools_list()
        decimals = 12
        row_list.extend(
            [[pool.id, pool.owner, round(agents[pool.owner].stake, decimals), round(pool.pledge, decimals),
              round(pool.stake, decimals), round(agents[pool.owner].cost, decimals), round(pool.cost, decimals),
              round(pool.margin, decimals)] for pool in pools])
        suffix = '-final_configuration_pools.csv' if self.has_converged() else '-intermediate_configuration_pools.csv'
        filename = self.execution_id + suffix

        hlp.export_csv_file(row_list, filename)

    def get_pools_list(self):
        return list(self.pools.values())

    def get_agents_dict(self):
        return {agent.unique_id: agent for agent in self.schedule.agents}

    def get_agents_list(self):
        return self.schedule.agents

    def get_status(self):
        print("Step {}: {} pools"
              .format(self.schedule.steps, len(self.pools)))

    def revise_beliefs(self):
        """
        Revise the perceived active stake and expected number of pools,
        to reflect the current state of the system
        The value for the active stake is calculated based on the currently delegated stake
        Note that this value is an estimate that the agents can easily calculate and use with the knowledge they have,
        it's not necessarily equal to the sum of all active agents' stake
        """
        # Revise active stake
        active_stake = sum([pool.stake for pool in self.pools.values()])
        self.perceived_active_stake = active_stake
        # Revise expected number of pools, k  (note that the value of beta, which is used to calculate rewards, does not change in this case)
        self.k = math.ceil(round(active_stake / self.beta, 12))  # first rounding to 12 decimal digits to avoid floating point errors

    def adjust_params(self):
        self.current_era += 1
        change_occured = False
        for attr_name, attr_values_list in zip(self.adjustable_params._fields, self.adjustable_params):
            if len(attr_values_list) > self.current_era:
                setattr(self, attr_name, attr_values_list[self.current_era])
                change_occured = True
                if attr_name == 'k':
                    self.k = int(self.k)
                    # update beta in case the value of k changes
                    self.beta = self.total_stake / self.k
        if change_occured:
            self.pivot_steps.append(self.schedule.steps)

    def wrap_up_execution(self):
        self.running = False
        print("Execution {} took  {:.2f} seconds to run.".format(self.execution_id, time.time() - self.start_time))
        self.export_pools_file()
        self.export_agents_file()