import pytest

from pool import Pool
from stakeholder import Stakeholder

import sim


def test_get_controlled_stake_mean_abs_diff(mocker):
    model = sim.Simulation()

    players_dict = {
        1: Stakeholder(unique_id=1, model=model, stake=0.01),
        2: Stakeholder(unique_id=2, model=model, stake=0.04)
    }

    pool1 = Pool(owner=1, cost=0.001, pledge=0.01,  margin=0.1, alpha=0.3, beta=0.1, pool_id=555)
    pool1.stake = 0.08
    pool2 = Pool(owner=2, cost=0.001, pledge=0.01,  margin=0.1, alpha=0.3, beta=0.1, pool_id=556)
    pool2.stake = 0.1
    pool3 = Pool(owner=2, cost=0.001, pledge=0.01,  margin=0.1, alpha=0.3, beta=0.1, pool_id=557)
    pool3.stake = 0.05
    pools_list = [pool1, pool2, pool3]

    mocker.patch('sim.Simulation.get_players_dict', return_value=players_dict)
    mocker.patch('sim.Simulation.get_pools_list', return_value=pools_list)

    mean_abs_diff = sim.get_controlled_stake_mean_abs_diff(model)

    assert pytest.approx(mean_abs_diff) == 0.09 # use approximation because of floating point operation error
