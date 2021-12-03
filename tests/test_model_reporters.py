import pytest

from logic.model_reporters import *
from logic.pool import Pool
from logic.stakeholder import Stakeholder
import logic.sim


# Note: need to 'pip install pytest-mock' to run some of these tests


def test_get_number_of_pools():
    assert False


def test_get_controlled_stake_mean_abs_diff(mocker):
    model = logic.sim.Simulation()

    players_dict = {
        1: Stakeholder(unique_id=1, model=model, stake=0.01),
        2: Stakeholder(unique_id=2, model=model, stake=0.04)
    }

    pool1 = Pool(owner=1, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=555)
    pool1.stake = 0.08
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=557)
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

    pool1 = Pool(owner=1, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=555)
    pool1.stake = 0.08
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=557)
    pool3.stake = 0.05
    pools_list = [pool1, pool2, pool3]

    mocker.patch('logic.sim.Simulation.get_players_dict', return_value=players_dict)
    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)
    mocker.patch('logic.sim.Simulation.has_converged', return_value=True)

    stat_dist = get_controlled_stake_distr_stat_dist(model)

    assert pytest.approx(stat_dist) == 0.095  # use approximation because of floating point operation error


def test_get_min_aggregate_pledge(mocker):
    model = logic.sim.Simulation()

    pool1 = Pool(owner=1, cost=0.001, pledge=0.001, margin=0.1, alpha=0.3, beta=0.1, pool_id=555)
    pool1.stake = 0.09
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01, margin=0.1, alpha=0.3, beta=0.1, pool_id=556)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.002, margin=0.1, alpha=0.3, beta=0.1, pool_id=557)
    pool3.stake = 0.05
    pools_list = [pool1, pool2, pool3]

    mocker.patch('logic.sim.Simulation.get_pools_list', return_value=pools_list)

    min_aggr_pledge = get_min_aggregate_pledge(model)

    assert min_aggr_pledge == 0.003
