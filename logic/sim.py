# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:59:49 2021

@author: chris
"""
import csv
import time
import pathlib
import math
import collections
import sys

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
    'abstention_rate',
    'relative_utility_threshold',
    'absolute_utility_threshold',
    'common_cost'
])


class Simulation(Model):
    """
    Simulation of staking behaviour in Proof-of-Stake Blockchains.
    """

    player_activation_orders = {
        "Random": RandomActivation,
        "Sequential": BaseScheduler,
        "Simultaneous": SimultaneousActivation, # note that during simultaneous activation players apply their moves sequentially which may not be the expected behaviour
        "Semisimultaneous": SemiSimultaneousActivation

    }

    def __init__(
            self, n=1000, k=100, alpha=0.3, myopic_fraction=0, abstention_rate=0,
            relative_utility_threshold=0, absolute_utility_threshold=1e-9,
            min_steps_to_keep_pool=5, pool_splitting=True, seed=None, pareto_param=2.0, max_iterations=1000,
            common_cost=1e-6, cost_min=1.1e-6, cost_max=2e-5, player_activation_order="Random", total_stake=1,
            ms=10, execution_id=''
    ):
        # todo make sure that the input is valid? n > 0, 0 < k <= n

        print(sys.version)
        print([f"{k}, {v}" for k, v in locals().items()])

        self.arguments = locals()  # only used for naming the output files appropriately

        # An era is defined as a time period during which the parameters of the model don't change
        self.current_era = 0
        total_eras = 1

        adjustable_params = AdjustableParams(
            k=k if isinstance(k, list) else [k],
            alpha=alpha if isinstance(alpha, list) else [alpha],
            common_cost=common_cost if isinstance(common_cost,list) else [common_cost],
            relative_utility_threshold=relative_utility_threshold if isinstance(relative_utility_threshold, list)else [
                relative_utility_threshold],
            absolute_utility_threshold=absolute_utility_threshold if isinstance(absolute_utility_threshold, list) else [
                absolute_utility_threshold],
            myopic_fraction=myopic_fraction if isinstance(myopic_fraction, list) else [myopic_fraction],
            abstention_rate=abstention_rate if isinstance(abstention_rate, list) else [abstention_rate]
        )

        for attr_name, attr_values_list in zip(adjustable_params._fields, adjustable_params):
            setattr(self, attr_name, attr_values_list[self.current_era]) #todo maybe self.abstention_rate not necessary?
            if len(attr_values_list) > total_eras:
                total_eras = len(attr_values_list)
        self.total_eras = total_eras
        self.adjustable_params = adjustable_params

        self.n = n
        self.min_steps_to_keep_pool = min_steps_to_keep_pool
        self.pool_splitting = pool_splitting
        self.max_iterations = max_iterations
        self.total_stake = total_stake
        self.player_activation_order = player_activation_order

        self.perceived_active_stake = total_stake
        self.beta = total_stake / self.k
        self.execution_id = execution_id if execution_id != '' else self.generate_execution_id()

        self.running = True  # for batch running and visualisation purposes
        self.schedule = self.player_activation_orders[player_activation_order](self)
        self.consecutive_idle_steps = 0  # steps towards convergence
        self.current_step_idle = True
        self.min_consecutive_idle_steps_for_convergence = max(min_steps_to_keep_pool + 1, ms)
        self.pools = dict()
        self.revision_frequency = 10  # defines how often active stake and expected #pools are revised
        # self.initial_states = {"inactive":0, "maximally_decentralised":1, "nicely_decentralised":2} todo maybe support different initial states

        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run
        self.initialize_players(cost_min, cost_max, pareto_param)

        self.datacollector = DataCollector(
            model_reporters={
                "#Pools": get_number_of_pools,
                "PoolSizes": get_pool_sizes,
                "PoolSizesByAgent": get_pool_sizes_by_agent,
                "PoolSizesByPool": get_pool_sizes_by_pool,
                "DesirabilitiesByAgent": get_desirabilities_by_agent,
                "DesirabilitiesByPool": get_desirabilities_by_pool,
                "StakePairs": get_stakes_n_margins,
                "AvgPledge": get_avg_pledge,
                "TotalPledge": get_total_pledge,
                "MedianPledge": get_median_pledge,
                "MeanAbsDiff": get_controlled_stake_mean_abs_diff,
                "StatisticalDistance": get_controlled_stake_distr_stat_dist,
                "NakamotoCoefficient": get_nakamoto_coefficient,
                # "NCR": get_NCR,
                # "MinAggregatePledge": get_min_aggregate_pledge,
                # "PledgeRate": get_pledge_rate,
                "AreaCoverage": get_homogeneity_factor,
                "MarginChanges": get_margin_changes,
                "AvgMargin": get_avg_margin,
                "MedianMargin": get_median_margin,
                "Iterations": get_convergence_iterations
            })

        self.pool_owner_id_mapping = dict()
        self.start_time = None
        self.equilibrium_steps = []
        self.pivot_steps = []

    def initialize_players(self, cost_min, cost_max, pareto_param):

        # Allocate stake to the players, sampling from a Pareto distribution
        stake_distribution = hlp.generate_stake_distr(num_agents=self.n, total_stake=self.total_stake,
                                                      pareto_param=pareto_param)

        # Allocate cost to the players, sampling from a uniform distribution
        #cost_distribution = hlp.generate_cost_distr_unfrm(num_agents=self.n, low=cost_min, high=cost_max)
        cost_distribution = hlp.generate_cost_distr_bands(num_agents=self.n, low=cost_min, high=cost_max, num_bands=3)
        #cost_distribution = hlp.generate_cost_distr_nrm(num_agents=self.n, low=cost_min, high=cost_max, mean=5e-6, stddev=5e-1)

        num_myopic_agents = int(self.myopic_fraction * self.n)
        num_abstaining_agents = int(self.abstention_rate * self.n)
        unique_ids = [i for i in range(self.n)]
        self.random.shuffle(unique_ids)
        # Create agents
        for i, unique_id in enumerate(unique_ids):
            agent = Stakeholder(
                unique_id=unique_id,
                model=self,
                is_abstainer=(i < num_abstaining_agents),
                is_myopic=(num_abstaining_agents <= i < num_abstaining_agents + num_myopic_agents),
                cost=cost_distribution[i],
                stake=stake_distribution[i]
            )
            self.schedule.add(agent)

    def initialise_pool_id_seq(self):
        self.id_seq = 0

    def get_next_pool_id(self):
        self.id_seq += 1
        return self.id_seq

    def rewind_pool_id_seq(self, step=1):
        self.id_seq -= step

    # One step of the model
    def step(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.datacollector.collect(self)

        current_step = self.schedule.steps

        if current_step > 0 and current_step % self.revision_frequency == 0:
            self.revise_beliefs()

        if current_step >= self.max_iterations:
            self.running = False
            print("Model took  {:.2f} seconds to run.".format(time.time() - self.start_time))
            self.dump_state_to_csv()
            return

        # Activate all agents (in the order specified by self.schedule) to perform all their actions for one time step
        self.schedule.step()
        if self.current_step_idle:
            self.consecutive_idle_steps += 1
            if self.has_converged():
                self.equilibrium_steps.append(current_step - self.min_consecutive_idle_steps_for_convergence + 1)
                if self.current_era < self.total_eras - 1:
                    self.adjust_params()
                else:
                    self.running = False
                    print("Model took  {:.2f} seconds to run.".format(time.time() - self.start_time))
                    self.dump_state_to_csv()
        else:
            self.consecutive_idle_steps = 0
        self.current_step_idle = True
        self.get_status()

    # Run multiple steps
    def run_model(self):
        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run
        while self.schedule.steps <= self.max_iterations and self.running:
            self.step()

    def has_converged(self):
        """
            Check whether the system has reached a state of equilibrium,
            where no player wants to change their strategy
        """
        return self.consecutive_idle_steps >= self.min_consecutive_idle_steps_for_convergence

    def dump_state_to_csv(self):
        row_list = [
            ["Owner id", "Pool id", "Pool stake", "Margin", "Perfect margin", "Pledge",
             "Owner stake", "Owner stake rank", "Pool cost", "Owner cost rank", "Pool desirability",
             "Pool potential profit", "Owner PP rank", "Pool desirability rank", "Pool status"]]
        players = self.get_players_dict()
        pools = self.get_pools_list()
        player_potential_profits = {
            player.unique_id: hlp.calculate_potential_profit(player.stake, player.cost, self.alpha, self.beta) for
            player in players.values()}
        pool_potential_profits = {pool.id: pool.potential_profit for pool in pools}
        pool_potential_profit_ranks = hlp.calculate_ranks(pool_potential_profits)
        player_potential_profit_ranks = hlp.calculate_ranks(player_potential_profits)
        desirabilities = {pool.id: pool.calculate_desirability() for pool in pools}
        desirability_ranks = hlp.calculate_ranks(desirabilities, pool_potential_profits)
        stakes = {player_id: player.stake for player_id, player in players.items()}
        stake_ranks = hlp.calculate_ranks(stakes)
        negative_cost_ranks = hlp.calculate_ranks({player_id: -player.cost for player_id, player in players.items()})
        decimals = 8
        row_list.extend(
            [[pool.owner, pool.id, round(pool.stake, decimals), round(pool.margin, decimals),
              round(players[pool.owner].calculate_margin_perfect_strategy(), decimals),
              round(pool.pledge, decimals), round(players[pool.owner].stake, decimals), stake_ranks[pool.owner],
              round(pool.cost, decimals),
              negative_cost_ranks[pool.owner], round(pool.calculate_desirability(), 10),
              round(pool.potential_profit, decimals), player_potential_profit_ranks[pool.owner],
              desirability_ranks[pool.id],
              "Private" if pool.is_private else "Public"]
             for pool in pools])

        today = time.strftime("%d-%m-%Y")
        output_dir = "output/" + today
        path = pathlib.Path.cwd() / output_dir
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = (path / (self.execution_id + '-final_configuration.csv')) \
            if self.has_converged() else (path / (self.execution_id + '-intermediate-configuration.csv'))
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        # temporary, used to extract results in latex format for easier reporting
        latex_dir = output_dir + "/latex/"
        hlp.to_latex(row_list, self.execution_id, latex_dir)

    def get_pools_list(self):
        return list(self.pools.values())

    def get_players_dict(self):
        return {player.unique_id: player for player in self.schedule.agents}

    def get_players_list(self):
        return self.schedule.agents

    def get_status(self):
        print("Step {}: {} pools".format(self.schedule.steps, len(self.pools)))

    def generate_execution_id(self):
        return "".join(['-' + str(key) + '=' + str(value) for key, value in self.arguments.items()
                        if type(value) == bool or type(value) == int or type(value) == float])[:147]

    def revise_beliefs(self):
        """
        Revise the perceived active stake and expected number of pools,
        to reflect the current state of the system
        The value for the active stake is calculated based on the currently delegated stake
        Note that this value is an estimate that the players can easily calculate and use with the knowledge they have,
        it's not necessarily equal to the sum of all active players' stake
        """
        # Revise active stake
        active_stake = sum([pool.stake for pool in self.pools.values()])
        self.perceived_active_stake = active_stake
        # Revise expected number of pools, k
        self.k = math.ceil(round(active_stake / self.beta, 12))  # first rounding to 12 decimal digits to avoid floating point errors

    def adjust_params(self):
        self.current_era += 1
        change_occured = False
        for attr_name, attr_values_list in zip(self.adjustable_params._fields, self.adjustable_params):
            if len(attr_values_list) > self.current_era:
                setattr(self, attr_name, attr_values_list[self.current_era])
                change_occured = True
                if attr_name == 'k':
                    # update beta in case the value of k changes
                    self.beta = self.total_stake / self.k
                elif attr_name == 'abstention_rate':
                    # update agent properties in case the abstention rate changes
                    abstention_change = attr_values_list[self.current_era] - attr_values_list[self.current_era - 1]
                    new_abstaining_agents = int(abstention_change * self.n)
                    all_agents = self.schedule.agents
                    if new_abstaining_agents > 0:
                        # abstention increased
                        # todo fix issue when abstention increases
                        for i, agent in enumerate(all_agents):
                            if not agent.abstains and agent.remaining_min_steps_to_keep_pool == 0:
                                agent.abstains = True
                            new_abstaining_agents -= 1
                            if new_abstaining_agents == 0:
                                break
                    else:
                        # abstention decreased
                        for i, agent in enumerate(all_agents):
                            if agent.abstains:
                                agent.abstains = False
                            new_abstaining_agents += 1
                            if new_abstaining_agents == 0:
                                break
        if change_occured:
            self.pivot_steps.append(self.schedule.steps)
