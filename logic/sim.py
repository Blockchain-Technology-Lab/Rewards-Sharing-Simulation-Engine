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
import logic.helper as hlp

MIN_CONSECUTIVE_IDLE_STEPS_FOR_CONVERGENCE = 11


def get_number_of_pools(model):
    return len([1 for pool in model.pools if pool is not None and pool != []])


def get_pool_sizes(model):
    return [pool.stake if pool is not None else 0 for pool in model.pools]


def get_avg_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.pools if pool is not None]
    return sum(current_pool_pledges) / len(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_stakes_n_margins(model):
    players = model.schedule.agents
    pools = model.pools
    return {'x': [players[pool.owner].stake if pool is not None else 0 for pool in pools],
            'y': [pool.stake if pool is not None else 0 for pool in pools],
            'r': [pool.margin if pool is not None else 0 for pool in pools]}


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

    def __init__(self, n=100, k=10, alpha=0.3, total_stake=1, max_iterations=100, seed=None,
                 cost_min=0.001, cost_max=0.002, pareto_param=1.5, player_activation_order="Random",
                 idle_steps_after_pool=10, myopic_fraction=0):
        if seed is not None:
            random.seed(seed)

        self.num_agents = n
        self.k = k
        self.beta = 1 / k
        self.alpha = alpha
        self.total_stake = total_stake
        self.max_iterations = max_iterations
        self.cost_min = cost_min
        self.cost_max = cost_max
        self.pareto_param = pareto_param
        self.player_activation_order = player_activation_order
        self.idle_steps_after_pool = idle_steps_after_pool
        self.myopic_fraction = myopic_fraction

        self.running = True  # for batch running and visualisation purposes
        self.schedule = self.player_activation_orders[player_activation_order](self)
        self.current_step = 0
        self.idle_steps = 0  # steps towards convergence
        self.current_step_idle = True
        self.pools = [None] * self.num_agents
        # self.initial_states = {"inactive":0, "maximally_decentralised":1, "nicely_decentralised":2} todo support different initial states
        # todo add aggregate values as fields? (e.g. total delegated stake)

        self.initialize_players()

        self.datacollector = DataCollector(
            model_reporters={"#Pools": get_number_of_pools, "Pool": get_pool_sizes,  # change "Pool" label?
                             "StakePairs": get_stakes_n_margins, "AvgPledge": get_avg_pledge})

    def initialize_players(self):

        # Allocate stake to the players, sampling from a Pareto distribution
        stake_distribution = hlp.generate_stake_distr(self.num_agents, self.total_stake, self.pareto_param)

        # Allocate cost to the players, sampling from a uniform distribution
        cost_distribution = hlp.generate_cost_distr(num_agents=self.num_agents, low=self.cost_min, high=self.cost_max)

        num_myopic_agents = int(self.myopic_fraction * self.num_agents)
        # Create agents
        for i in range(self.num_agents):
            agent = Stakeholder(i, self, is_myopic=(i < num_myopic_agents), cost=cost_distribution[i],
                                stake=stake_distribution[i])
            self.schedule.add(agent)

    # One step of the model
    def step(self):
        self.datacollector.collect(self)

        if self.current_step >= self.max_iterations:
            self.running = False

        # Activate all agents (in the order specified by self.schedule) to perform all their actions for one time step
        self.schedule.step()
        if self.current_step_idle:
            self.idle_steps += 1
            if self.has_converged():
                self.running = False
                self.dump_state_to_csv()
        else:
            self.idle_steps = 0
        self.current_step += 1
        self.current_step_idle = True
        # self.get_status()

    # Run multiple steps
    def run_model(self, max_steps=1):
        i = 0
        while i < max_steps and self.running:
            self.step()
            i += 1

    def has_converged(self):
        """
            Check whether the system has reached a state of equilibrium,
            where no player wants to change their strategy
        """
        return self.idle_steps >= MIN_CONSECUTIVE_IDLE_STEPS_FOR_CONVERGENCE

    def dump_state_to_csv(self):
        # todo add potential profit rank column
        row_list = [["Pool owner id", "Pool owner stake", "Pool stake", "Pool cost", "Pool pledge", "Pool margin",
                     "Perfect margin", "Pool potential profit", "Pool desirability"]]
        players = self.schedule.agents
        row_list.extend([[pool.owner, players[pool.owner].stake, pool.stake, pool.cost, pool.pledge, pool.margin,
                          players[pool.owner].calculate_margin_perfect_strategy(), pool.potential_profit, pool.calculate_desirability()]
                         for pool in self.pools if pool is not None])
        current_datetime = time.strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / "output"
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = path / ('final_configuration' + current_datetime + '.csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

    def get_status(self):
        print("Step {}".format(self.current_step))
        print("Number of agents: {} \n Number of pools: {} \n"
              .format(self.num_agents, len([1 for p in self.pools if p is not None])))
