# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:59:49 2021

@author: chris
"""
import random
import csv
import time
from collections import defaultdict
from pathlib import Path

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler, SimultaneousActivation, RandomActivation

from logic.stakeholder import Stakeholder
import logic.helper as hlp

MAX_NUM_POOLS = 1000


def get_number_of_pools(model):
    return len(model.get_pools_list())


def get_pool_sizes(model):
    max_pools = MAX_NUM_POOLS - 1  # must be < max defined for the chart
    pool_sizes = {i: 0 for i in range(max_pools)}
    current_pools = model.pools
    for pool_id in current_pools:
        pool_sizes[pool_id] = current_pools[pool_id].stake
    return dict(sorted(pool_sizes.items()))


def get_pool_sizes_by_agent(model):  # !! attention: only works when one pool per agent!
    return {pool.owner: pool.stake for pool in model.get_pools_list()}


def get_pool_sizes_by_pool(model):
    pools = defaultdict(lambda: 0)
    pools.update({pool_id: pool.stake for pool_id, pool in model.pools.items()})
    return [pools[i] for i in range(MAX_NUM_POOLS)] if len(pools) > 0 else [0]*MAX_NUM_POOLS


def get_desirabilities_by_agent(model):
    desirabilities = defaultdict(lambda: 0)
    for pool in model.get_pools_list():
        desirabilities[pool.owner] = pool.calculate_desirability()
    return [desirabilities[i] for i in range(model.num_agents)]


def get_desirabilities_by_pool(model):
    desirabilities = defaultdict(lambda: 0)
    for id, pool in model.pools.items():
        desirabilities[id] = pool.calculate_desirability()
    return [desirabilities[i] for i in range(MAX_NUM_POOLS)] if len(desirabilities) > 0 else [0]*MAX_NUM_POOLS


def get_avg_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return sum(current_pool_pledges) / len(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_stakes_n_margins(model):
    players = model.schedule.agents
    pools = model.get_pools_list()
    return {'x': [players[pool.owner].stake for pool in pools],
            'y': [pool.stake for pool in pools],
            'r': [pool.margin for pool in pools],
            'id': [pool.id for pool in pools]}


def get_total_delegated_stake(model):
    stake_from_pools = sum([pool.stake for pool in model.get_pools_list()])
    stake_from_players = sum([sum([a for a in player.strategy.stake_allocations.values()])
                              for player in model.schedule.agents]) + \
                         sum([sum([pledge for pledge in player.strategy.pledges])
                               for player in model.schedule.agents])
    return stake_from_pools, stake_from_players


class Simulation(Model):
    """
    Simulation of staking behaviour in Proof-of-Stake Blockchains.
    """

    player_activation_orders = {
        "Sequential": BaseScheduler,
        "Random": RandomActivation,
        "Simultaneous": SimultaneousActivation
        # todo check if during simultaneous activation players apply their moves sequentially or randomly (sequential may not be fair)
    }

    def __init__(self, n=100, k=10, alpha=0.3, total_stake=1, max_iterations=100, seed=42,
                 cost_min=0.001, cost_max=0.002, utility_threshold=1e-9, pareto_param=2.0, player_activation_order="Random",
                 idle_steps_after_pool=10, myopic_fraction=0, pool_splitting=True, common_cost=1e-5):
        if seed is not None:
            random.seed(seed)

        self.num_agents = n
        self.k = k
        self.beta = 1 / k
        self.alpha = alpha
        self.total_stake = total_stake
        self.max_iterations = max_iterations
        self.player_activation_order = player_activation_order
        self.idle_steps_after_pool = idle_steps_after_pool
        self.myopic_fraction = myopic_fraction
        self.pool_splitting = pool_splitting
        self.common_cost = common_cost
        self.utility_threshold = utility_threshold

        self.running = True  # for batch running and visualisation purposes
        self.schedule = self.player_activation_orders[player_activation_order](self)
        self.idle_steps = 0  # steps towards convergence
        self.current_step_idle = True
        self.min_consecutive_idle_steps_for_convergence = idle_steps_after_pool + 1
        self.pools = defaultdict(lambda: None)
        # self.initial_states = {"inactive":0, "maximally_decentralised":1, "nicely_decentralised":2} todo support different initial states

        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run
        self.initialize_players(cost_min, cost_max, pareto_param)

        self.datacollector = DataCollector(
            model_reporters={"#Pools": get_number_of_pools, "PoolSizes": get_pool_sizes,
                             "PoolSizesByAgent": get_pool_sizes_by_agent,
                             "PoolSizesByPool": get_pool_sizes_by_pool,
                             "DesirabilitiesByAgent": get_desirabilities_by_agent,
                             "DesirabilitiesByPool": get_desirabilities_by_pool,
                             "StakePairs": get_stakes_n_margins, "AvgPledge": get_avg_pledge})

    def initialize_players(self, cost_min, cost_max, pareto_param):

        # Allocate stake to the players, sampling from a Pareto distribution
        stake_distribution = hlp.generate_stake_distr(self.num_agents, self.total_stake, pareto_param)

        # Allocate cost to the players, sampling from a uniform distribution
        cost_distribution = hlp.generate_cost_distr(num_agents=self.num_agents, low=cost_min, high=cost_max)

        num_myopic_agents = int(self.myopic_fraction * self.num_agents)
        # Create agents
        for i in range(self.num_agents):
            agent = Stakeholder(i, self, is_myopic=(i < num_myopic_agents), cost=cost_distribution[i],
                                stake=stake_distribution[i])
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
        self.datacollector.collect(self)

        if self.schedule.steps >= self.max_iterations:
            self.running = False
            self.dump_state_to_csv()

        # Activate all agents (in the order specified by self.schedule) to perform all their actions for one time step
        self.schedule.step()
        if self.current_step_idle:
            self.idle_steps += 1
            if self.has_converged():
                self.running = False
                self.dump_state_to_csv()
        else:
            self.idle_steps = 0
        self.current_step_idle = True
        # self.get_status()

    # Run multiple steps
    def run_model(self, max_steps=300):
        i = 0
        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run
        while i < max_steps and self.running:
            self.step()
            i += 1

    def has_converged(self):
        """
            Check whether the system has reached a state of equilibrium,
            where no player wants to change their strategy
        """
        return self.idle_steps >= self.min_consecutive_idle_steps_for_convergence

    def dump_state_to_csv(self):
        row_list = [
            ["Owner id", "Pool id", "Owner stake", "Pool stake", "Pool pledge", "Pool cost", "Pool margin",
             "Perfect margin", "Pool potential profit", "Pool desirability", "Owner potential profit rank",
             "Private pool"]]
        players = self.schedule.agents
        potential_profits = {
            player.unique_id: hlp.calculate_potential_profit(player.stake, player.cost, self.alpha, self.beta) for
            player in players}
        row_list.extend(
            [[pool.owner, pool.id, players[pool.owner].stake, pool.stake, pool.pledge, pool.cost, pool.margin,
              players[pool.owner].calculate_margin_perfect_strategy(), pool.potential_profit,
              pool.calculate_desirability(), hlp.calculate_rank(potential_profits, pool.owner), pool.is_private]
             for pool in self.get_pools_list()])
        current_datetime = time.strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / "output"
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = path / ('final_configuration' + current_datetime + '.csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        sim_df = self.datacollector.get_model_vars_dataframe()
        pool_sizes_by_step = sim_df["PoolSizesByPool"]
        filename = path / ('pool_sizes_by_step' + current_datetime + '.csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            header_row = [i for i in range(MAX_NUM_POOLS)]
            header_row.append("sum")
            writer.writerow(header_row)
            for row in pool_sizes_by_step:
                row.append(sum(row))
                writer.writerow(row)

        desirabilities = sim_df["DesirabilitiesByPool"]
        filename = path / ('desirabilities_by_step' + current_datetime + '.csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i for i in range(MAX_NUM_POOLS)])
            writer.writerows(desirabilities)

    def get_pools_list(self):
        return list(self.pools.values())

    def get_status(self):
        print("Step {}".format(self.schedule.steps))
        print("Number of agents: {} \n Number of pools: {} \n"
              .format(self.num_agents, len([1 for p in self.pools if p is not None])))
