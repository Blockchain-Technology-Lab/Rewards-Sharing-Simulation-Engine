import statistics
import collections
from gekko import GEKKO
import numpy as np

from logic.helper import MAX_NUM_POOLS
import logic.helper as hlp

def get_number_of_pools(model):
    return len(model.pools)


def get_avg_margin(model):
    pools = model.get_pools_list()
    margins = [pool.margin for pool in pools]
    return statistics.mean(margins) if len(margins) > 0 else 0


def get_median_margin(model):
    pools = model.get_pools_list()
    margins = [pool.margin for pool in pools]
    return statistics.median(margins) if len(margins) > 0 else 0

def get_avg_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return statistics.mean(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_total_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return sum(current_pool_pledges)


def get_median_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return statistics.median(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_avg_pools_per_operator(model):
    current_pools = model.get_pools_list()
    current_num_pools = len(current_pools)
    if current_num_pools == 0:
        return 0
    current_num_operators = len(set([pool.owner for pool in current_pools]))
    return current_num_pools / current_num_operators


def get_max_pools_per_operator(model):
    current_pools = model.get_pools_list()
    if len(current_pools) == 0:
        return 0
    current_owners = [pool.owner for pool in current_pools]
    max_frequency_owner, max_pool_count_per_owner = collections.Counter(current_owners).most_common(1)[0]
    return max_pool_count_per_owner


def get_median_pools_per_operator(model):
    current_pools = model.get_pools_list()
    if len(current_pools) == 0:
        return 0
    current_owners = [pool.owner for pool in current_pools]
    sorted_frequencies = sorted(collections.Counter(current_owners).values())
    return statistics.median(sorted_frequencies)


def get_avg_sat_rate(model):
    sat_point = model.beta
    current_pools = model.get_pools_list()
    if len(current_pools) == 0:
        return 0
    sat_rates = [pool.stake / sat_point for pool in current_pools]
    return statistics.mean(sat_rates)


def get_stakes_n_margins(model):
    agents = model.get_agents_dict()
    pools = model.get_pools_list()
    return {
        'x': [agents[pool.owner].stake for pool in pools],
        'y': [pool.stake for pool in pools],
        'r': [pool.margin for pool in pools],
        'pool_id': [pool.id for pool in pools],
        'owner_id': [pool.owner for pool in pools]
    }


def get_controlled_stake_distr_stat_dist(model):
    """
    :param model:
    :return: the statistical distance of the distributions of the stake that agents control
                (how they started vs how they ended up)
    """
    active_agents = {
        agent_id: agent
        for agent_id, agent in model.get_agents_dict().items()
    }
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    initial_controlled_stake = {
        agent_id: active_agents[agent_id].stake
        for agent_id in active_agents
    }
    current_controlled_stake = {
        agent_id: 0
        for agent_id in active_agents
    }
    for pool in pools:
        current_controlled_stake[pool.owner] += pool.stake
    abs_diff = [
        abs(current_controlled_stake[agent_id] - initial_controlled_stake[agent_id])
        for agent_id in active_agents
    ]
    return sum(abs_diff) / 2


def get_nakamoto_coefficient(model):
    """
    The Nakamoto coefficient is defined as the minimum number of entities that control more than 50% of the system
    (and can therefore launch a 51% attack against it). This function returns the nakamoto coefficient for a given
    simulation instance.
    :param model: the instance of the simulation
    :return: the number of agents that control more than 50% of the total active stake through their pools
    """
    agents = model.get_agents_dict()
    active_agents = {agent_id: agents[agent_id] for agent_id in agents}
    try:
        pools = model.get_pools_list()
    except AttributeError:
        # no pools have been created at this point
        # todo merge in one mechanism for pools or agents
        agent_stakes = [agent.stake for agent in agents.values()]
        sorted_final_stake = sorted(agent_stakes, reverse=True)
        majority_control_agents = 0
        majority_control_stake = 0
        index = 0
        total_stake = sum(sorted_final_stake)
        while majority_control_stake <= total_stake / 2:
            majority_control_stake += sorted_final_stake[index]
            majority_control_agents += 1
            index += 1

        return majority_control_agents

    if len(pools) == 0:
        return 0

    controlled_stake = {agent_id: 0 for agent_id in active_agents}
    for pool in pools:
        controlled_stake[pool.owner] += pool.stake

    final_stake = [controlled_stake[agent_id] for agent_id in active_agents.keys()]
    total_active_stake = sum(final_stake)

    sorted_final_stake = sorted(final_stake, reverse=True)
    # final_stake.sort(reverse=True)
    majority_control_agents = 0
    majority_control_stake = 0
    index = 0
    #todo make simpler
    while majority_control_stake <= total_active_stake / 2:
        majority_control_stake += sorted_final_stake[index]
        majority_control_agents += 1
        index += 1

    return majority_control_agents


def get_min_aggregate_pledge(model):
    """
    Solve optimisation problem using solver
    """

    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0

    ids = [pool.id for pool in pools]
    pledges = [pool.pledge for pool in pools]
    stakes = [pool.stake for pool in pools]
    items = len(ids)

    # Create model
    m = GEKKO()
    # Variables
    x = m.Array(m.Var, len(ids), lb=0, ub=1, integer=True)
    # Objective
    m.Minimize(m.sum([pledges[i] * x[i] for i in range(items)]))
    # Constraint
    lower_bound = sum(stakes) / 2
    m.Equation(m.sum([stakes[i] * x[i] for i in range(items)]) >= lower_bound)
    # Optimize with APOPT
    m.options.SOLVER = 1

    try:
        m.solve(disp=False) # choose disp = True to print details while running
    except Exception:
        print("Min aggregate pledge not found")
        return -2

    min_aggr_pledge = m.options.objfcnval
    return min_aggr_pledge


def get_pledge_rate(model):
    """
    Pledge rate is defined as: total_pledge / total_active_stake
    :param model: instance of the simulation
    :return: the pledge rate of the model at its current state
    """
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
    pools = model.get_pools_list()
    pool_count = len(pools)
    if pool_count == 0:
        return 0
    pool_stakes = [pool.stake for pool in pools]
    max_stake = max(pool_stakes)

    ideal_area = pool_count * max_stake
    actual_area = sum(pool_stakes)

    return actual_area / ideal_area


def get_iterations(model):
    return model.schedule.steps


def get_avg_stk_rnk(model):
    pools = model.get_pools_list()
    all_agents = model.get_agents_dict()
    stakes = {agent_id: agent.stake for agent_id, agent in all_agents.items()}
    stake_ranks = hlp.calculate_ranks(stakes)
    pool_owner_stk_ranks = [stake_ranks[pool.owner] for pool in pools]
    return round(statistics.mean(pool_owner_stk_ranks)) if len(pool_owner_stk_ranks) > 0 else 0


def get_avg_cost_rnk(model):
    pools = model.get_pools_list()
    all_agents = model.get_agents_dict()
    negative_cost_ranks = hlp.calculate_ranks({agent_id: -agent.cost for agent_id, agent in all_agents.items()})
    pool_owner_cost_ranks = [negative_cost_ranks[pool.owner] for pool in pools]
    return round(statistics.mean(pool_owner_cost_ranks)) if len(pool_owner_cost_ranks) > 0 else 0


def get_median_stk_rnk(model):
    pools = model.get_pools_list()
    all_agents = model.get_agents_dict()
    stakes = {agent_id: agent.stake for agent_id, agent in all_agents.items()}
    stake_ranks = hlp.calculate_ranks(stakes)
    pool_owner_stk_ranks = [stake_ranks[pool.owner] for pool in pools]
    return round(statistics.median(pool_owner_stk_ranks)) if len(pool_owner_stk_ranks) > 0 else 0


def get_median_cost_rnk(model):
    pools = model.get_pools_list()
    all_agents = model.get_agents_dict()
    negative_cost_ranks = hlp.calculate_ranks({agent_id: -agent.cost for agent_id, agent in all_agents.items()})
    pool_owner_cost_ranks = [negative_cost_ranks[pool.owner] for pool in pools]
    return round(statistics.median(pool_owner_cost_ranks)) if len(pool_owner_cost_ranks) > 0 else 0


def get_pool_splitter_count(model):
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    pool_operators = [pool.owner for pool in pools]
    cnt = collections.Counter(pool_operators)
    pool_splitters = [k for k, v in cnt.items() if v > 1]

    return len(pool_splitters)


def get_cost_efficient_count(model):
    all_agents = model.get_agents_list()
    potential_profits = [
        hlp.calculate_potential_profit(agent.stake, agent.cost, model.a0, model.beta, model.reward_function)
        for agent in all_agents
    ]
    positive_potential_profits = [pp for pp in potential_profits if pp > 0]
    return len(positive_potential_profits)


def get_pool_stakes_by_agent(model):
    num_agents = model.n
    pool_stakes = [0 for _ in range(num_agents)]
    current_pools = model.get_pools_list()
    for pool in current_pools:
        pool_stakes[pool.owner] += pool.stake
    return pool_stakes


def get_pool_stakes_by_agent_id(model):
    num_agents = model.n
    pool_stakes = {i: 0 for i in range(num_agents)}
    current_pools = model.get_pools_list()
    for pool in current_pools:
        pool_stakes[pool.owner] += pool.stake
    return pool_stakes


def gini_coefficient(np_array):
    """Compute Gini coefficient of array of values
    using the fact that their Gini coefficient is half their relative mean absolute difference,
    as noted here: https://en.wikipedia.org/wiki/Mean_absolute_difference#Relative_mean_absolute_difference """
    diffsum = 0 # sum of absolute differences
    for i, xi in enumerate(np_array[:-1], 1):
        diffsum += np.sum(np.abs(xi - np_array[i:]))
    return diffsum / (len(np_array) * sum(np_array)) if sum(np_array) != 0 else -1


def get_gini_id_coeff_pool_count(model):
    # gather data
    pools = model.get_pools_list()
    #todo check later if you can abstract this to a function that serves this one, NC and others
    pools_owned = collections.defaultdict(lambda: 0)
    for pool in pools:
        pools_owned[pool.owner] += 1
    pools_per_agent = np.fromiter(pools_owned.values(), dtype=int)
    return gini_coefficient(pools_per_agent)


def get_gini_id_coeff_pool_count_k_agents(model):
    # use at least k agents (if there aren't k pool operators, pad with non-pool operators)
    pools = model.get_pools_list()
    pools_owned = collections.defaultdict(lambda: 0)
    for pool in pools:
        pools_owned[pool.owner] += 1
    pools_per_agent = np.fromiter(pools_owned.values(), dtype=int)
    if pools_per_agent.size < model.k:
        missing_values = model.k - pools_per_agent.size
        pools_per_agent = np.append(pools_per_agent, np.zeros(missing_values, dtype=int))
    return gini_coefficient(pools_per_agent)


def get_gini_id_coeff_stake(model):
    pools = model.get_pools_list()
    stake_controlled = collections.defaultdict(lambda: 0)
    for pool in pools:
        stake_controlled[pool.owner] += pool.stake
    stake_per_agent = np.fromiter(stake_controlled.values(), dtype=float)
    return gini_coefficient(stake_per_agent)


def get_gini_id_coeff_stake_k_agents(model):
    pools = model.get_pools_list()
    stake_controlled = collections.defaultdict(lambda: 0)
    for pool in pools:
        stake_controlled[pool.owner] += pool.stake
    stake_per_agent = np.fromiter(stake_controlled.values(), dtype=float)
    if stake_per_agent.size < model.k:
        missing_values = model.k - stake_per_agent.size
        stake_per_agent = np.append(stake_per_agent, np.zeros(missing_values, dtype=int))
    return gini_coefficient(stake_per_agent)


def get_total_delegated_stake(model):
    pools = model.get_pools_list()
    del_stake = sum([pool.stake for pool in pools])
    return del_stake


def get_active_stake_agents(model):
    return sum([agent.stake for agent in model.schedule.agents])


def get_stake_distr_stats(model):
    stake_distribution = np.array([agent.stake for agent in model.schedule.agents])
    return stake_distribution.max(), stake_distribution.min(), stake_distribution.mean(), np.median(stake_distribution), stake_distribution.std()


def get_operator_count(model):
    return len({pool.owner for pool in model.get_pools_list()})


all_model_reporters = {
    "Pool count": get_number_of_pools,
    "Total pledge": get_total_pledge,
    "Mean pledge": get_avg_pledge,
    "Median pledge": get_median_pledge,
    "Average pools per operator": get_avg_pools_per_operator,
    "Max pools per operator": get_max_pools_per_operator,
    "Median pools per operator": get_median_pools_per_operator,
    "Average saturation rate": get_avg_sat_rate,
    "Nakamoto coefficient": get_nakamoto_coefficient,
    "Statistical distance": get_controlled_stake_distr_stat_dist,
    "Min-aggregate pledge": get_min_aggregate_pledge,
    "Pledge rate": get_pledge_rate,
    "Pool homogeneity factor": get_homogeneity_factor,
    "Iterations": get_iterations,
    "Mean stake rank": get_avg_stk_rnk,
    "Mean cost rank": get_avg_cost_rnk,
    "Median stake rank": get_median_stk_rnk,
    "Median cost rank": get_median_cost_rnk,
    "Number of pool splitters": get_pool_splitter_count,
    "Cost efficient stakeholders": get_cost_efficient_count,
    "Gini-id": get_gini_id_coeff_pool_count,
    "Gini-id stake": get_gini_id_coeff_stake,
    "Gini-id (k)": get_gini_id_coeff_pool_count_k_agents,
    "Gini-id stake (k)": get_gini_id_coeff_stake_k_agents,
    "Mean margin": get_avg_margin,
    "Median margin": get_median_margin,
    "Stake per agent": get_pool_stakes_by_agent,
    "Stake per agent id": get_pool_stakes_by_agent_id,
    "StakePairs": get_stakes_n_margins,
    "Total delegated stake": get_total_delegated_stake,
    "Total agent stake": get_active_stake_agents,
    "Operator count": get_operator_count
}

reporter_ids = {
    1: "Pool count",
    2: "Total pledge",
    3: "Mean pledge",
    4: "Median pledge",
    5: "Average pools per operator",
    6: "Max pools per operator",
    7: "Median pools per operator",
    8: "Average saturation rate",
    9: "Nakamoto coefficient",
    10: "Statistical distance",
    11: "Min-aggregate pledge",
    12: "Pledge rate",
    13: "Pool homogeneity factor",
    14: "Iterations",
    15: "Mean stake rank",
    16: "Mean cost rank",
    17: "Median stake rank",
    18: "Median cost rank",
    19: "Number of pool splitters",
    20: "Cost efficient stakeholders",
    21: "StakePairs",
    22: "Gini-id",
    23: "Gini-id stake",
    24: "Mean margin",
    25: "Median margin",
    26: "Stake per agent",
    27: "Stake per agent id",
    28: "Total delegated stake",
    29: "Total agent stake",
    30: "Operator count"
}
