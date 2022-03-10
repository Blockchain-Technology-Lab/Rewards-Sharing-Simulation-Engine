import statistics
import collections
from gekko import GEKKO

from logic.helper import MAX_NUM_POOLS
import logic.helper as hlp

def get_number_of_pools(model):
    return len(model.pools)


def get_final_number_of_pools(model):
    if not model.has_converged():
        return -1
    return len(model.pools)


def get_margin_changes(model):
    pools = model.get_pools_list()
    margin_increase, margin_decrease, margin_abs_change = 0, 0, 0
    for pool in pools:
        margin_change = pool.margin_change
        if margin_change > 0:
            margin_increase += margin_change
            margin_abs_change += margin_change
        else:
            margin_decrease -= margin_change
            margin_abs_change -= margin_change
    return margin_increase, margin_decrease, margin_abs_change


def get_avg_margin(model):
    pools = model.get_pools_list()
    margins = [pool.margin for pool in pools]
    return statistics.mean(margins) if len(margins) > 0 else 0


def get_median_margin(model):
    pools = model.get_pools_list()
    margins = [pool.margin for pool in pools]
    return statistics.median(margins) if len(margins) > 0 else 0


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
    return statistics.mean(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_total_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return sum(current_pool_pledges)


def get_median_pledge(model):
    current_pool_pledges = [pool.pledge for pool in model.get_pools_list()]
    return statistics.median(current_pool_pledges) if len(current_pool_pledges) > 0 else 0


def get_avg_pools_per_operator(model):
    current_pools = model.pools
    current_num_pools = len(current_pools)
    current_num_operators = len(set([pool.owner for pool in current_pools.values()]))
    return current_num_pools / current_num_operators if current_num_operators > 0 else 0


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
    current_pools = model.pools
    if len(current_pools) == 0:
        return 0
    sat_rates = [pool.stake / sat_point for pool in current_pools.values()]
    return statistics.mean(sat_rates) if len(sat_rates) > 0 else 0


def get_stakes_n_margins(model):
    players = model.get_players_dict()
    pools = model.get_pools_list()
    return {
        'x': [players[pool.owner].stake for pool in pools],
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


def get_controlled_stake_mean_abs_diff(model):
    """

    :param model:
    :return: the mean value of the absolute differences of the stake the players control
                (how they started vs how they ended up)
    """
    active_players = {player_id: player for player_id, player in model.get_players_dict().items() if
                      not player.abstains}
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    initial_controlled_stake = {player_id: active_players[player_id].stake for player_id in active_players}
    current_controlled_stake = {player_id: 0 for player_id in active_players}
    for pool in pools:
        current_controlled_stake[pool.owner] += pool.stake
    abs_diff = [abs(current_controlled_stake[player_id] - initial_controlled_stake[player_id])
                for player_id in active_players]
    return statistics.mean(abs_diff)


def get_controlled_stake_distr_stat_dist(model):
    """
    :param model:
    :return: the statistical distance of the distributions of the stake that players control
                (how they started vs how they ended up)
    """
    active_players = {
        player_id: player
        for player_id, player in model.get_players_dict().items()
        if not player.abstains
    }
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0
    initial_controlled_stake = {
        player_id: active_players[player_id].stake
        for player_id in active_players
    }
    current_controlled_stake = {
        player_id: 0
        for player_id in active_players
    }
    for pool in pools:
        current_controlled_stake[pool.owner] += pool.stake
    abs_diff = [
        abs(current_controlled_stake[player_id] - initial_controlled_stake[player_id])
        for player_id in active_players
    ]
    return sum(abs_diff) / 2


def get_nakamoto_coefficient(model):
    """
    The Nakamoto coefficient is defined as the minimum number of entities that control more than 50% of the system
    (and can therefore launch a 51% attack against it). This function returns the nakamoto coefficient for a given
    simulation instance.
    :param model: the instance of the simulation
    :return: the number of players that control more than 50% of the total active stake through their pools
    """
    players = model.get_players_dict()
    active_players = {player_id: players[player_id] for player_id in players if not players[player_id].abstains}
    try:
        pools = model.get_pools_list()
    except AttributeError:
        # no pools have been created at this point
        # todo merge in one mechanism for pools or players
        player_stakes = [player.stake for player in players.values()]
        sorted_final_stake = sorted(player_stakes, reverse=True)
        majority_control_players = 0
        majority_control_stake = 0
        index = 0
        total_stake = sum(sorted_final_stake)
        while majority_control_stake <= total_stake / 2:
            majority_control_stake += sorted_final_stake[index]
            majority_control_players += 1
            index += 1

        return majority_control_players

    if len(pools) == 0:
        return 0

    controlled_stake = {player_id: 0 for player_id in active_players}
    for pool in pools:
        controlled_stake[pool.owner] += pool.stake

    final_stake = [controlled_stake[player_id] for player_id in active_players.keys()]
    total_active_stake = sum(final_stake)

    sorted_final_stake = sorted(final_stake, reverse=True)
    # final_stake.sort(reverse=True)
    majority_control_players = 0
    majority_control_stake = 0
    index = 0
    #todo make simpler
    while majority_control_stake <= total_active_stake / 2:
        majority_control_stake += sorted_final_stake[index]
        majority_control_players += 1
        index += 1

    return majority_control_players


def get_NCR(model):
    nakamoto_coefficient = get_nakamoto_coefficient(model)
    return nakamoto_coefficient / model.n if nakamoto_coefficient >= 0 else -1

def get_optimal_min_aggregate_pledge(model):
    """
    In the optimal scenario (pledge-wise) there are k pools
    operated by the k richest agents (the ones who can pledge the highest amounts)
    and having 1/k stake each.
    In that case, any k/2 pools control half of the total stake,
    therefore the k/2 pools controlled by the k/2 (or k/2 + 1 for odd k) "poorest" agents (among the k richest)
    should define the optimal min-aggregate pledge.
    Potential caveat: maybe doesn't generalise to when we have abstention
    """
    k = model.k
    all_stakes = [player.stake for player in model.get_players_list()]
    relevant_stakes = sorted(all_stakes, reverse=True)[int(k/2):k]
    return sum(relevant_stakes)

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
    and can be used as an indication of the system's degree of decentralisation
    :param model: instance of the simulation
    :return: the pledge rate of the final configuration (otherwise -1)
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
    all_players = model.get_players_dict()
    pool_owner_ids = {pool.owner for pool in pools}
    stakes = {player_id: player.stake for player_id, player in all_players.items()}
    stake_ranks = hlp.calculate_ranks(stakes)
    pool_owner_stk_ranks = [stake_ranks[pool_owner] for pool_owner in pool_owner_ids]
    return statistics.mean(pool_owner_stk_ranks) if len(pool_owner_stk_ranks) > 0 else 0


def get_avg_cost_rnk(model):
    pools = model.get_pools_list()
    all_players = model.get_players_dict()
    pool_owner_ids = {pool.owner for pool in pools}
    negative_cost_ranks = hlp.calculate_ranks({player_id: -player.cost for player_id, player in all_players.items()})
    pool_owner_cost_ranks = [negative_cost_ranks[pool_owner] for pool_owner in pool_owner_ids]

    return statistics.mean(pool_owner_cost_ranks) if len(pool_owner_cost_ranks) > 0 else 0


def get_median_stk_rnk(model):
    pools = model.get_pools_list()
    all_players = model.get_players_dict()
    pool_owner_ids = {pool.owner for pool in pools}
    stakes = {player_id: player.stake for player_id, player in all_players.items()}
    stake_ranks = hlp.calculate_ranks(stakes)
    pool_owner_stk_ranks = [stake_ranks[pool_owner] for pool_owner in pool_owner_ids]
    return statistics.median(pool_owner_stk_ranks) if len(pool_owner_stk_ranks) > 0 else 0


def get_median_cost_rnk(model):
    pools = model.get_pools_list()
    all_players = model.get_players_dict()
    pool_owner_ids = {pool.owner for pool in pools}
    negative_cost_ranks = hlp.calculate_ranks({player_id: -player.cost for player_id, player in all_players.items()})
    pool_owner_cost_ranks = [negative_cost_ranks[pool_owner] for pool_owner in pool_owner_ids]

    return statistics.median(pool_owner_cost_ranks) if len(pool_owner_cost_ranks) > 0 else 0


def get_pool_splitter_count(model):
    pools = model.get_pools_list()
    if len(pools) == 0:
        return 0

    pool_operators = [pool.owner for pool in pools]

    cnt = collections.Counter(pool_operators)
    pool_splitters = [k for k, v in cnt.items() if v > 1]

    return len(pool_splitters)

def get_cost_efficient_count(model):
    all_players = model.get_players_list()
    potential_profits = [
        hlp.calculate_potential_profit(player.stake, player.cost, model.alpha, model.beta, model.reward_function_option, model.total_stake)
        for player in all_players
    ]
    positive_potential_profits = [pp for pp in potential_profits if pp > 0]
    return len(positive_potential_profits)


def get_pool_stake_distribution_snapshot(model):
    return [pool.stake for pool in model.get_pools_list()]


def get_pool_stakes_by_agent(model):
    num_agents = model.n
    pool_stakes = [0 for _ in range(num_agents)]
    current_pools = model.get_pools_list()
    for pool in current_pools:
        pool_stakes[pool.owner] += pool.stake
    return pool_stakes


# note that any new model reporters should be added to the end, to maintain the colour allocation
all_model_reporters = {
    "Pool count": get_final_number_of_pools,
    "Average pledge": get_avg_pledge,
    "Total pledge": get_total_pledge,
    "Average pools per operator": get_avg_pools_per_operator,
    "Max pools per operator": get_max_pools_per_operator,
    "Median pools per operator": get_median_pools_per_operator,
    "Average saturation rate": get_avg_sat_rate,
    "Nakamoto coefficient": get_nakamoto_coefficient,
    "Statistical distance": get_controlled_stake_distr_stat_dist,
    "Nakamoto coefficient rate": get_NCR,
    "Min-aggregate pledge": get_min_aggregate_pledge,
    "Pledge rate": get_pledge_rate,
    "Homogeneity factor": get_homogeneity_factor,
    "Iterations": get_iterations,
    "Average stake rank": get_avg_stk_rnk,
    "Average cost rank": get_avg_cost_rnk,
    "Median stake rank": get_median_stk_rnk,
    "Median cost rank": get_median_cost_rnk,
    "Opt min aggr pledge": get_optimal_min_aggregate_pledge,
    "Number of pool splitters": get_pool_splitter_count,
    "Cost efficient stakeholders": get_cost_efficient_count
}