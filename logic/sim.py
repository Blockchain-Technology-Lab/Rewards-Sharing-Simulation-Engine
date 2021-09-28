# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:59:49 2021

@author: chris
"""
import random
import csv
import time
from pathlib import Path

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler, SimultaneousActivation, RandomActivation

from logic.stakeholder import Stakeholder
from logic.activations import SemiSimultaneousActivation
import logic.helper as hlp

MAX_NUM_POOLS = 1000


def get_number_of_pools(model):
    return len(model.pools)


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
    pool_stakes = {pool_id: pool.stake for pool_id, pool in model.pools.items()}
    return [pool_stakes[i] if i in pool_stakes else 0 for i in range(1, MAX_NUM_POOLS)] \
        if len(pool_stakes) > 0 else [0] * MAX_NUM_POOLS


def get_desirabilities_by_agent(model):
    desirabilities = dict()
    for pool in model.get_pools_list():
        desirabilities[pool.owner] = pool.calculate_desirability()
    return [desirabilities[i] if i in desirabilities else 0 for i in range(model.n)]


def get_desirabilities_by_pool(model):
    desirabilities = dict()
    for id, pool in model.pools.items():
        desirabilities[id] = pool.calculate_desirability()
    return [desirabilities[i] if i in desirabilities else 0 for i in range(1, MAX_NUM_POOLS)] \
        if len(desirabilities) > 0 else [0] * MAX_NUM_POOLS


def get_avg_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return sum(current_pool_pledges) / len(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_avg_pools_per_operator(model):
    current_pools = model.pools
    current_num_pools = len(current_pools)
    current_num_operators = len(set([pool.owner for pool in current_pools.values()]))
    return current_num_pools / current_num_operators


def get_stakes_n_margins(model):
    players = model.get_players_dict()
    pools = model.get_pools_list()
    return {'x': [players[pool.owner].stake for pool in pools],
            'y': [pool.stake for pool in pools],
            'r': [pool.margin for pool in pools],
            'pool_id': [pool.id for pool in pools],
            'owner_id': [pool.owner for pool in pools]
            }


def get_total_delegated_stake(model):
    players = model.get_players_list()
    stake_from_pools = sum([pool.stake for pool in model.get_pools_list()])
    stake_from_players = sum([sum([a for a in player.strategy.stake_allocations.values()])
                              for player in players]) + \
                         sum([sum([pledge for pledge in player.strategy.pledges])
                              for player in players])
    return stake_from_pools, stake_from_players


class Simulation(Model):
    """
    Simulation of staking behaviour in Proof-of-Stake Blockchains.
    """

    player_activation_orders = {
        "Random": RandomActivation,
        "Sequential": BaseScheduler,
        "Simultaneous": SimultaneousActivation,
        "Semisimultaneous": SemiSimultaneousActivation
        # todo during simultaneous activation players apply their moves sequentially which may not be the expected behaviour

    }

    def __init__(self, n=100, k=10, alpha=0.3, myopic_fraction=0.1, abstaining_fraction=0.1,
                 relative_utility_threshold=0.1, absolute_utility_threshold=1e-9,
                 min_steps_to_keep_pool=5, pool_splitting=True, seed=42, pareto_param=2.0, max_iterations=1000,
                 common_cost=1e-4, cost_min=0.001, cost_max=0.002, player_activation_order="Random", total_stake=1,
                 ms=10, simulation_id=''
                 ):

        # todo make sure that the input is valid? n > 0, 0 < k <= n

        self.arguments = locals()  # only used for naming the output files appropriately
        if seed is not None:
            random.seed(seed)

        self.n = n
        self.k = k
        self.alpha = alpha
        self.myopic_fraction = myopic_fraction
        self.abstaining_fraction = abstaining_fraction
        self.absolute_utility_threshold = absolute_utility_threshold
        self.relative_utility_threshold = relative_utility_threshold
        self.min_steps_to_keep_pool = min_steps_to_keep_pool
        self.pool_splitting = pool_splitting
        self.common_cost = common_cost
        self.max_iterations = max_iterations
        self.beta = 1 / k
        self.total_stake = total_stake
        self.player_activation_order = player_activation_order
        self.simulation_id = simulation_id if simulation_id != '' else \
            "".join(['-' + str(key) + '=' + str(value) for key, value in self.arguments.items()
                     if type(value) == bool or type(value) == int or type(value) == float])[:147]

        self.running = True  # for batch running and visualisation purposes
        self.schedule = self.player_activation_orders[player_activation_order](self)
        self.consecutive_idle_steps = 0  # steps towards convergence
        self.current_step_idle = True
        self.min_consecutive_idle_steps_for_convergence = max(min_steps_to_keep_pool + 1, ms)
        self.pools = dict()
        # self.initial_states = {"inactive":0, "maximally_decentralised":1, "nicely_decentralised":2} todo maybe support different initial states

        self.initialise_pool_id_seq()  # initialise pool id sequence for the new model run
        self.initialize_players(cost_min, cost_max, pareto_param)

        self.datacollector = DataCollector(
            model_reporters={"#Pools": get_number_of_pools, "PoolSizes": get_pool_sizes,
                             "PoolSizesByAgent": get_pool_sizes_by_agent,
                             "PoolSizesByPool": get_pool_sizes_by_pool,
                             "DesirabilitiesByAgent": get_desirabilities_by_agent,
                             "DesirabilitiesByPool": get_desirabilities_by_pool,
                             "StakePairs": get_stakes_n_margins, "AvgPledge": get_avg_pledge})

        self.pool_owner_id_mapping = {}
        self.start_time = None

    def initialize_players(self, cost_min, cost_max, pareto_param):

        # Allocate stake to the players, sampling from a Pareto distribution
        stake_distribution = hlp.generate_stake_distr(self.n, total_stake=self.total_stake,
                                                      pareto_param=pareto_param)

        # Allocate cost to the players, sampling from a uniform distribution
        cost_distribution = hlp.generate_cost_distr(num_agents=self.n, low=cost_min, high=cost_max)

        num_myopic_agents = int(self.myopic_fraction * self.n)
        num_abstaining_agents = int(self.abstaining_fraction * self.n)
        unique_ids = [i for i in range(self.n)]
        random.shuffle(unique_ids)
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

        if self.schedule.steps >= self.max_iterations:
            self.running = False
            print("Model took  {:.2f} seconds to run.".format(time.time() - self.start_time))
            self.dump_state_to_csv()
            return

        # Activate all agents (in the order specified by self.schedule) to perform all their actions for one time step
        self.schedule.step()
        if self.current_step_idle:
            self.consecutive_idle_steps += 1
            if self.has_converged():
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
             "Pool potential profit", "Owner PP rank", "Pool status"]]
        players = self.get_players_dict()
        pools = self.get_pools_list()
        potential_profits = {
            player.unique_id: hlp.calculate_potential_profit(player.stake, player.cost, self.alpha, self.beta) for
            player in players.values()}
        potential_profit_ranks = hlp.calculate_ranks(potential_profits)
        stakes = {player_id: player.stake for player_id, player in players.items()}
        stake_ranks = hlp.calculate_ranks(stakes)
        negative_cost_ranks = hlp.calculate_ranks({player_id: -player.cost for player_id, player in players.items()})
        decimals = 4
        row_list.extend(
            [[pool.owner, pool.id, round(pool.stake, decimals), round(pool.margin, decimals),
              round(players[pool.owner].calculate_margin_perfect_strategy(), decimals),
              round(pool.pledge, decimals), round(players[pool.owner].stake, decimals), stake_ranks[pool.owner],
              round(pool.cost, decimals),
              negative_cost_ranks[pool.owner], round(pool.calculate_desirability(), decimals),
              round(pool.potential_profit, decimals), potential_profit_ranks[pool.owner],
              "Private" if pool.is_private else "Public"]
             for pool in pools])

        path = Path.cwd() / "output"
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = (path / (self.simulation_id + '-final_configuration.csv')) \
            if self.has_converged() else (path / (self.simulation_id + '-intermediate-configuration.csv'))
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

    def get_pools_list(self):
        return list(self.pools.values())

    def get_players_dict(self):
        return {player.unique_id: player for player in self.schedule.agents}

    def get_players_list(self):
        return self.schedule.agents

    def get_status(self):
        print("Step {}: {} pools".format(self.schedule.steps, len(self.pools)))
