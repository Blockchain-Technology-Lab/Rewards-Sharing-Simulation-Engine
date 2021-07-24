import pytest

from pool import Pool
from sim import Simulation
from stakeholder import Stakeholder


def test_step():
    stakeholder = Stakeholder()
    assert False


def test_advance():
    assert False


def test_make_move():
    assert False


def test_update_strategy():
    assert False


def test_calculate_utility():
    assert False


def test_calculate_operator_utility():
    model = Simulation()
    pool = Pool(cost=0.001, pledge=0.01, owner=156, margin=0.1, alpha=0.3, beta=0.1, pool_id=555)
    model.pools[555] = pool
    player = Stakeholder(unique_id=156, model=model, cost=0.001)
    utility = player.calculate_operator_utility(pool)
    assert utility == 0.0148638461538461537


# add test for (over)saturated pools
def test_calculate_delegator_utility():
    assert False


def test_has_potential_for_pool():
    assert False


def test_calculate_pledges():
    assert False


def test_calculate_margin_simple():
    assert False


def test_calculate_margin_perfect_strategy():
    assert False


def test_find_operator_move():
    assert False


def test_find_delegation_move_desirability():
    assert False


def test_execute_strategy():
    assert False


def test_open_pool():
    assert False


def test_close_pool():
    model = Simulation()
    pool = Pool(cost=0.001, pledge=0.001, owner=156, margin=0.2, alpha=0.3, beta=0.1, pool_id=555)
    model.pools[555] = pool
    player = Stakeholder(156, model)
    player.close_pool(555)
    assert model.pools[555] is None

    # try to close the same pool again but get an exception because it doesn't exist anymore
    with pytest.raises(ValueError) as e_info:
        player.close_pool(55)
    assert str(e_info.value) == 'Given pool id is not valid.'

    # try to close another player's pool
    with pytest.raises(ValueError) as e_info:
        model.pools[555] = pool
        player = Stakeholder(157, model)
        player.close_pool(555)
    assert str(e_info.value) == "Player tried to close pool that belongs to another player."
