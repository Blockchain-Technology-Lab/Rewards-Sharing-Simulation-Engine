# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:59:49 2021

@author: chris
"""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from logic.stakeholder import Stakeholder
from logic import helper as hlp

MIN_CONSECUTIVE_IDLE_STEPS_FOR_CONVERGENCE = 5


def get_number_of_pools(model):
    return len([1 for pool in model.pools if pool is not None and pool != []])


def get_pool_sizes(model):
    return [pool.stake if pool is not None else 0 for pool in hlp.flatten_list(model.pools)]


# Only valid when poolSplitting = True
def get_pool_group_sizes(model):
    return [sum([pool.stake for pool in group]) for group in model.pools]


class Simulation(Model):
    """
    Simulation of staking behaviour in Proof-of-Stake Blockchains.
    """

    def __init__(self, n=100, k=10, alpha=0.3, total_stake=1, max_iterations=100, seed=None,
                 pool_splitting=False, cost_min=0.001, cost_max=0.002):
        if seed is not None:
            pass #todo set random seed (bur for random or numpy?)

        self.num_agents = n
        self.k = k
        self.beta = 1/k
        self.alpha = alpha
        self.total_stake = total_stake
        self.max_iterations = max_iterations
        self.pool_splitting = pool_splitting  # True if players are allowed to operate multiple pools
        self.cost_min = cost_min
        self.cost_max = cost_max

        self.current_step = 0
        self.running = True  # for batch running and visualisation purposes
        self.schedule = BaseScheduler(
            self)  # RandomActivation(self)  <- use base if you want them to get activated in specific order

        self.initialize_system()
        self.initialize_players()

        self.datacollector = DataCollector(
            model_reporters={"#Pools": get_number_of_pools, "Pool": get_pool_sizes},
            agent_reporters={"Utility": "utility"})

    def initialize_players(self):
        # initialize system
        agent_types = ['M', 'NM']  # myopic, non-myopic

        # Allocate stake to the players, sampling from a Pareto distribution
        stake_distribution = hlp.generate_stake_distr(self.num_agents, self.total_stake)

        # Allocate cost to the players, sampling from a uniform distribution
        cost_distribution = hlp.generate_cost_distr(num_agents=self.num_agents, low=self.cost_min, high=self.cost_max)
        # todo cost distribution for pool splitting? mc1 + c2 for each player (where m = #player's pools)

        # Create agents
        for i in range(self.num_agents):
            # for now only non-myopic agents, in the future we can mix them
            agent_type = 'NM'#random.choice(agent_types)
            agent = Stakeholder(i, self, agent_type, cost=cost_distribution[i],
                                stake=stake_distribution[i], can_split_pools=self.pool_splitting)
            self.schedule.add(agent)

    def initialize_system(self):
        # self.initial_states = {"inactive":0, "maximally_decentralised":1, "nicely_decentralised":2}
        element = [] if self.pool_splitting else None
        self.pools = [element] * self.num_agents
        self.idle_steps = 0
        self.current_step_idle = True
        # todo add aggregate values as fields? (e.g. total delegated stake)

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
        else:
            self.idle_steps = 0
        self.current_step += 1
        self.current_step_idle = True
        self.get_status()

    # Run multiple steps
    def run_model(self, max_steps=1):
        i = 0
        while (i < max_steps and self.running):
            self.step()
            i += 1


    def has_converged(self):
        """
            Check whether the system has reached a state of equilibrium,
            where no player wants to change their strategy
        """
        return self.idle_steps >= MIN_CONSECUTIVE_IDLE_STEPS_FOR_CONVERGENCE

    def get_status(self):
        return
        print("Step {}".format(self.current_step))
        '''print("Number of agents: {} \n Number of pools: {} \n"
              .format(self.num_agents, len([1 for p in self.pools if p != None])))'''
        '''print("Pools: ")'''
        #for i, agent in enumerate(self.schedule.agents):
            #agent.get_status()
            #stake = self.pools[i].stake if self.pools[i] is not None else 0
            #print("Agent {}: Stake: {:.3f}, Cost: {:.3f}, Pool stake: {:.3f} \n"
            #     .format(agent.unique_id, agent.stake, agent.cost, stake))
        '''for pool in self.pools:
            if pool is not None:
                print(pool)'''
