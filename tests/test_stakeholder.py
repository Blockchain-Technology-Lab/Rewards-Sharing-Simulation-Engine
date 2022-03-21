import pytest

from logic.pool import Pool
from logic.sim import Simulation
from logic.stakeholder import Stakeholder
from logic.strategy import Strategy
import logic.helper as hlp


# todo add more tests

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


#todo review failing test
def test_calculate_operator_utility():
    model = Simulation(total_stake=1)
    pool = Pool(cost=0.001, pledge=0.1, owner=156, margin=0.1, alpha=0.3, beta=0.1, pool_id=555,
                reward_function_option=0, total_stake=1)
    model.pools[555] = pool
    player = Stakeholder(unique_id=156, model=model, stake=0.1, cost=0.001)
    strategy = Strategy(owned_pools={555: pool})

    utility = player.calculate_operator_utility_from_strategy(strategy)

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


# todo test going from 2 pools to 1 pool, 2 pools to 3 pools, 1 pool to 1 pool
def test_find_operator_move():
    assert False


def test_find_delegation_move_desirability():
    assert False


def test_execute_strategy():
    assert False


def test_open_pool():
    assert False


def test_close_pool():
    total_stake = 1
    model = Simulation(total_stake=total_stake)
    player = Stakeholder(156, model, 0.001)
    pool = Pool(cost=0.001, pledge=0.001, owner=156, margin=0.2, alpha=0.3, beta=0.1, pool_id=555, reward_function_option=0, total_stake=total_stake)
    model.pools[555] = pool

    player.close_pool(555)

    assert 555 not in model.pools.keys()

    # try to close the same pool again but get an exception because it doesn't exist anymore
    with pytest.raises(ValueError) as e_info:
        player.close_pool(555)
    assert str(e_info.value) == 'Given pool id is not valid.'

    # try to close another player's pool
    with pytest.raises(ValueError) as e_info:
        model.pools[555] = pool
        player = Stakeholder(157, model, 0.003)
        player.close_pool(555)
    assert str(e_info.value) == "Player tried to close pool that belongs to another player."


def test_calculate_margin_semi_perfect_strategy():
    total_stake = 1
    model = Simulation(k=2, total_stake=total_stake)
    player156 = Stakeholder(156, model, stake=0.001)
    player157 = Stakeholder(157, model, stake=0.002)
    player158 = Stakeholder(158, model, stake=0.003)
    player159 = Stakeholder(159, model, stake=0.0001)
    pool555 = Pool(cost=0.001, pledge=0.001, owner=156, alpha=0.3, beta=0.1, pool_id=555, reward_function_option=0, total_stake=total_stake)
    model.pools[555] = pool555
    pool556 = Pool(cost=0.001, pledge=0.002, owner=157, alpha=0.3, beta=0.1, pool_id=556, reward_function_option=0, total_stake=total_stake)
    model.pools[556] = pool556
    pool557 = Pool(cost=0.001, pledge=0.003, owner=158, alpha=0.3, beta=0.1, pool_id=557, reward_function_option=0, total_stake=total_stake)
    model.pools[557] = pool557
    pool558 = Pool(cost=0.001, pledge=0.0001, owner=159, alpha=0.3, beta=0.1, pool_id=558, reward_function_option=0, total_stake=total_stake)
    model.pools[558] = pool558

    pool555.margin = player156.calculate_margin_semi_perfect_strategy(pool555)
    pool556.margin = player157.calculate_margin_semi_perfect_strategy(pool556)
    pool557.margin = player158.calculate_margin_semi_perfect_strategy(pool557)
    pool558.margin = player159.calculate_margin_semi_perfect_strategy(pool558)

    assert pool555.margin == pool558.margin == 0
    assert pool557.margin > pool556.margin > 0

    desirability555 = hlp.calculate_pool_desirability(pool555.margin, pool555.potential_profit)
    desirability556 = hlp.calculate_pool_desirability(pool556.margin, pool556.potential_profit)
    desirability557 = hlp.calculate_pool_desirability(pool557.margin, pool557.potential_profit)
    desirability558 = hlp.calculate_pool_desirability(pool558.margin, pool558.potential_profit)
    assert desirability555 == desirability556 == desirability557 > desirability558 > 0

    #todo test tie breaking

