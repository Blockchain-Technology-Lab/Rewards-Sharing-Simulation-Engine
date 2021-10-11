# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:59:49 2021

@author: chris
"""
import random
import csv
import time
import pathlib
import math
import collections
import statistics
import itertools

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


def get_max_pools_per_operator(model):
    current_pools = model.get_pools_list()
    current_owners = [pool.owner for pool in current_pools]
    max_frequency_owner, max_pool_count_per_owner = collections.Counter(current_owners).most_common(1)[0]
    return max_pool_count_per_owner


def get_median_pools_per_operator(model):
    current_pools = model.get_pools_list()
    current_owners = [pool.owner for pool in current_pools]
    sorted_frequencies = sorted(collections.Counter(current_owners).values())
    return statistics.median(sorted_frequencies)


def get_avg_sat_rate(model):
    sat_point = model.beta
    current_pools = model.pools
    sat_rates = [pool.stake / sat_point for pool in current_pools.values()]
    return sum(sat_rates) / len(current_pools)


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


# todo deal with undelegated stake
def get_controlled_stake_mean_abs_diff(model):
    """

    :param model:
    :return: the mean value of the absolute differences of the stake the players control
                (how they started vs how they ended up)
    """
    if not model.has_converged():
        return -1
    players = model.get_players_dict()
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    initial_controlled_stake = {player_id: players[player_id].stake for player_id in players}
    current_controlled_stake = {player_id: 0 for player_id in players}
    for pool in pools:
        current_controlled_stake[pool.owner] += pool.stake
    abs_diff = [abs(current_controlled_stake[player_id] - initial_controlled_stake[player_id])
                for player_id in players]
    return sum(abs_diff) / len(abs_diff)


def get_nakamoto_coefficient(model):
    """
    The Nakamoto coefficient is defined as the minimum number of entities that control more than 50% of the system
    (and can therefore launch a 51% attack against it). This function returns the nakamoto coefficient for a given
    simulation instance.
    :param model: the instance of the simulation
    :return: the number of players that control more than 50% of the total active stake through their pools
    """
    if not model.has_converged():
        return -1
    players = model.get_players_dict()
    active_players = {player_id: players[player_id] for player_id in players if not players[player_id].abstains}
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0

    final_controlled_stake = {player_id: 0 for player_id in active_players}
    for pool in pools:
        final_controlled_stake[pool.owner] += pool.stake

    final_stake = [final_controlled_stake[player_id] for player_id in active_players.keys()]
    total_active_stake = sum(final_stake)

    sorted_final_stake = sorted(final_stake, reverse=True)
    majority_control_players = 0
    majority_control_stake = 0
    index = 0

    while majority_control_stake <= total_active_stake / 2:
        majority_control_stake += sorted_final_stake[index]
        majority_control_players += 1
        index += 1

    return majority_control_players


def get_NCR(model):
    if not model.has_converged():
        return -1
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    independent_pool_owners = {pool.owner for pool in pools}
    nakamoto_coefficient = get_nakamoto_coefficient(model)
    return nakamoto_coefficient / len(independent_pool_owners)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))


def get_min_aggregate_pledge(model):
    if not model.has_converged():
        return -1
    pools = model.pools
    if len(pools) == 0:
        return 0

    pool_stakes = {pool_id: pool.stake for pool_id, pool in pools.items()}
    total_active_stake = sum(pool_stakes.values())

    # todo find more efficient way that doesn't require calculating the entire powerset
    pool_subsets = list(powerset(pool_stakes))
    majority_pool_subsets = []
    for subset in pool_subsets:
        controlled_stake = 0
        for pool_id in subset:
            controlled_stake += pool_stakes[pool_id]
        if controlled_stake >= total_active_stake / 2:
            majority_pool_subsets.append(subset)

    aggregate_pledges = []
    for subset in majority_pool_subsets:
        aggregate_pledge = 0
        for pool_id in subset:
            pledge = pools[pool_id].pledge
            aggregate_pledge += pledge
        aggregate_pledges.append(aggregate_pledge)

    return min(aggregate_pledges)


def get_pledge_rate(model):
    """
    Pledge rate is defined as: total_pledge / total_active_stake
    and can be used as an indication of the system's degree of decentralisation
    :param model: instance of the simulation
    :return: the pledge rate of the final configuration (otherwise -1)
    """
    if not model.has_converged():
        return -1
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    total_active_stake = sum([pool.stake for pool in pools])
    total_pledge = sum([pool.pledge for pool in pools])
    return total_pledge / total_active_stake


def get_homogeneity_factor(model):
    """
    Shows how homogeneous the pools are
    :param model:
    :return:
    """
    if not model.has_converged():
        return -1
    pools = model.get_pools_list()
    pool_count = len(pools)
    if pool_count == 0:
        return 0
    pool_stakes = [pool.stake for pool in pools]
    max_stake = max(pool_stakes)

    ideal_area = pool_count * max_stake
    actual_area = sum(pool_stakes)

    return actual_area / ideal_area


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

    def __init__(self, n=100, k=10, alpha=0.3, myopic_fraction=0.1, abstention_rate=0.1,
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
        self.abstention_rate = abstention_rate
        self.absolute_utility_threshold = absolute_utility_threshold
        self.relative_utility_threshold = relative_utility_threshold
        self.min_steps_to_keep_pool = min_steps_to_keep_pool
        self.pool_splitting = pool_splitting
        self.common_cost = common_cost
        self.max_iterations = max_iterations
        self.total_stake = total_stake
        self.perceived_active_stake = total_stake
        self.beta = total_stake / k
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
                "MeanAbsDiff": get_controlled_stake_mean_abs_diff,
                "NakamotoCoefficient": get_nakamoto_coefficient,
                "NCR": get_NCR,
                "MinAggregatePledge": get_min_aggregate_pledge,
                "PledgeRate": get_pledge_rate,
                "AreaCoverage": get_homogeneity_factor
            })

        self.pool_owner_id_mapping = {}
        self.start_time = None

    def initialize_players(self, cost_min, cost_max, pareto_param):

        # Allocate stake to the players, sampling from a Pareto distribution
        stake_distribution = hlp.generate_stake_distr(self.n, total_stake=self.total_stake,
                                                      pareto_param=pareto_param)

        # Allocate cost to the players, sampling from a uniform distribution
        cost_distribution = hlp.generate_cost_distr(num_agents=self.n, low=cost_min, high=cost_max)

        num_myopic_agents = int(self.myopic_fraction * self.n)
        num_abstaining_agents = int(self.abstention_rate * self.n)
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

        if self.schedule.steps > 0 and self.schedule.steps % self.revision_frequency == 0:
            self.revise_beliefs()

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
             "Pool potential profit", "Owner PP rank", "Pool desirability rank", "Pool status"]]
        players = self.get_players_dict()
        pools = self.get_pools_list()
        potential_profits = {
            player.unique_id: hlp.calculate_potential_profit(player.stake, player.cost, self.alpha, self.beta) for
            player in players.values()}
        potential_profit_ranks = hlp.calculate_ranks(potential_profits)
        desirabilities = {pool.id: pool.calculate_desirability() for pool in pools}
        desirability_ranks = hlp.calculate_ranks(desirabilities)
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
              round(pool.potential_profit, decimals), potential_profit_ranks[pool.owner], desirability_ranks[pool.id],
              "Private" if pool.is_private else "Public"]
             for pool in pools])

        path = pathlib.Path.cwd() / "output"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        filename = (path / (self.simulation_id + '-final_configuration.csv')) \
            if self.has_converged() else (path / (self.simulation_id + '-intermediate-configuration.csv'))
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        # temporary, used to extract results in latex format for easier reporting
        hlp.to_latex(row_list)

    def get_pools_list(self):
        return list(self.pools.values())

    def get_players_dict(self):
        return {player.unique_id: player for player in self.schedule.agents}

    def get_players_list(self):
        return self.schedule.agents

    def get_status(self):
        print("Step {}: {} pools".format(self.schedule.steps, len(self.pools)))

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
        self.k = math.ceil(active_stake / self.beta)
