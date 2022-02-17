import pytest

from logic.model_reporters import *
from logic.pool import Pool
from logic.stakeholder import Stakeholder
import logic.sim


# Note: need to 'pip install pytest-mock' to run some of these tests


def test_get_number_of_pools():
    assert True


def test_get_controlled_stake_mean_abs_diff(mocker):
    model = logic.sim.Simulation()

    players_dict = {
        1: Stakeholder(unique_id=1, model=model, stake=0.01),
        2: Stakeholder(unique_id=2, model=model, stake=0.04)
    }

    pool1 = Pool(owner=1, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=555, reward_function_option=0)
    pool1.stake = 0.08
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=557, reward_function_option=0)
    pool3.stake = 0.05
    pools_list = [pool1, pool2, pool3]

    mocker.patch('logic.sim.Simulation.get_players_dict', return_value=players_dict)
    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)

    mean_abs_diff = get_controlled_stake_mean_abs_diff(model)

    assert pytest.approx(mean_abs_diff) == 0.09  # use approximation because of floating point operation error


def test_get_controlled_stake_distr_stat_dist(mocker):
    model = logic.sim.Simulation()

    players_dict = {
        1: Stakeholder(unique_id=1, model=model, stake=0.01),
        2: Stakeholder(unique_id=2, model=model, stake=0.04),
        3: Stakeholder(unique_id=3, model=model, stake=0.01)
    }

    pool1 = Pool(owner=1, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=555, reward_function_option=0)
    pool1.stake = 0.08
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=557, reward_function_option=0)
    pool3.stake = 0.05
    pools_list = [pool1, pool2, pool3]

    mocker.patch('logic.sim.Simulation.get_players_dict', return_value=players_dict)
    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)
    mocker.patch('logic.sim.Simulation.has_converged', return_value=True)

    stat_dist = get_controlled_stake_distr_stat_dist(model)

    assert pytest.approx(stat_dist) == 0.095  # use approximation because of floating point operation error


def test_get_min_aggregate_pledge(mocker):
    model = logic.sim.Simulation()

    pool1 = Pool(owner=1, cost=0.001, pledge=0.001, margin=0.1, alpha=0.3, beta=0.1, pool_id=555, reward_function_option=0)
    pool1.stake = 0.09
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.002, margin=0.1, alpha=0.3, beta=0.1, pool_id=557, reward_function_option=0)
    pool3.stake = 0.05
    pools_list = [pool1, pool2, pool3]

    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)
    mocker.patch('logic.sim.Simulation.has_converged', return_value=True)

    min_aggr_pledge = get_min_aggregate_pledge(model)

    assert min_aggr_pledge == 0.003

    pools_list = []
    num_pools = 500
    stake_per_pool = 0.001
    for i in range(num_pools):
        pools_list.append(
            Pool(owner=i, cost=0.001, pledge=stake_per_pool, margin=0.1, alpha=0.3, beta=0.1, pool_id=100 + i, reward_function_option=0))
    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)
    min_aggr_pledge = get_min_aggregate_pledge(model)
    assert min_aggr_pledge == num_pools / 2 * stake_per_pool


def test_get_optimal_min_aggregate_pledge(mocker):
    model = logic.sim.Simulation(k=10)
    players = [Stakeholder(unique_id=x, model=model, stake=x) for x in range(1, 101)]
    mocker.patch('logic.sim.Simulation.get_players_list', return_value=players)
    optimal_min_aggr_pledge = get_optimal_min_aggregate_pledge(model)
    assert optimal_min_aggr_pledge == (95 + 94 + 93 + 92 + 91)

    model = logic.sim.Simulation(k=11)
    optimal_min_aggr_pledge = get_optimal_min_aggregate_pledge(model)
    assert optimal_min_aggr_pledge == (95 + 94 + 93 + 92 + 91 + 90)


def test_get_pool_splitter_count(mocker):
    model = logic.sim.Simulation()

    pools_list = [Pool(owner=i, cost=0.001, pledge=0.001, margin=0.1, alpha=0.3, beta=0.1, pool_id=555, reward_function_option=0)
                  for i in range(1,11)]
    pools_list.append(Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0))
    pools_list.append(Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0))
    pools_list.append(Pool(owner=5, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0))

    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)

    pool_splitter_count = get_pool_splitter_count(model)
    assert pool_splitter_count == 2