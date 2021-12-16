import statistics
import itertools
import collections
from gekko import GEKKO

from logic.helper import MAX_NUM_POOLS


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
    if not model.has_converged():
        return -1
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
    # final_stake.sort(reverse=True) todo figure out if (and why) we get different results with this sorting
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


def get_min_aggregate_pledge(model):
    """
    Solve optimisation problem using solver
    @param model:
    @return:
    """
    if not model.has_converged():
        return -1

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


def get_iterations(model):
    return model.schedule.steps