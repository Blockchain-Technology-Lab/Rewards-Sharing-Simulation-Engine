import random

import pytest

import logic.helper as hlp


# todo add more tests

def test_generate_stake_distr():
    stk_distr = hlp.generate_stake_distr_pareto(num_agents=100, pareto_param=2)

    assert len(stk_distr) == 100
    assert sum(stk_distr) == 1

    stk_distr = hlp.generate_stake_distr_pareto(num_agents=1001, pareto_param=1.5, total_stake=21527)

    assert len(stk_distr) == 1001
    assert sum(stk_distr) == 21527


def test_generate_stake_distr_flat():
    stk_distr = hlp.generate_stake_distr_flat(num_agents=100, total_stake=1)

    rnd_idx = random.randint(1, 100)

    assert pytest.approx(stk_distr[rnd_idx]) == 0.01
    assert len(stk_distr) == 100
    assert pytest.approx(sum(stk_distr)) == 1


def test_generate_cost_distr():
    assert True


def test_normalize_distr():
    sample_dstr = [10, 8, 5, 5, 1, 0.5, 0.1]

    nrm_dstr = hlp.normalize_distr(sample_dstr)
    assert sum(nrm_dstr) == 1

    nrm_dstr = hlp.normalize_distr(sample_dstr, normal_sum=156)
    assert sum(nrm_dstr) == 156


def test_calculate_potential_profit():
    assert True


def test_calculate_pool_reward_variable_stake():
    # GIVEN
    alpha = 0.3
    saturation_point = 0.1
    stakes = [0.01, 0.1, 0.2]
    pledges = [0.01, 0.01, 0.01]

    # WHEN
    results = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=0
    ) for i in range(len(stakes))]

    # THEN
    assert results[0] < results[1] == results[2]


def test_calculate_pool_reward_variable_pledge():
    alpha = 0.3
    saturation_point = 0.1
    stakes = [0.1, 0.1, 0.1]
    pledges = [0.01, 0.05, 0.1]

    results = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=0
    ) for i in range(len(stakes))]

    assert results[0] < results[1] < results[2]


def test_calculate_pool_reward_variable_stake_alpha_zero():
    # GIVEN
    alpha = 0
    saturation_point = 0.1
    stakes = [0.01, 0.1, 0.2]
    pledges = [0.01, 0.01, 0.01]

    # WHEN
    results = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=0
    ) for i in range(len(stakes))]

    # THEN
    assert results[0] < results[1] == results[2]


def test_calculate_pool_reward_variable_pledge_alpha_zero():
    alpha = 0
    saturation_point = 0.1
    stakes = [0.1, 0.1, 0.1]
    pledges = [0.01, 0.05, 0.1]

    results = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=0
    ) for i in range(len(stakes))]

    assert results[0] == results[1] == results[2]


def test_calculate_pool_saturation_prob():
    assert True


def test_calculate_pool_stake_nm_my_way():
    assert True


def test_calculate_pool_stake_nm():
    # define pool, pools, pool_index, alpha, beta, k
    assert True


def test_calculate_ranks():
    desirabilities = {5: 0.2, 3: 0.3, 1: 0.1, 12: 0.9, 8: 0.8}
    ranks = {5: 4, 3: 3, 1: 5, 12: 1, 8: 2}
    assert hlp.calculate_ranks(desirabilities) == ranks


def test_calculate_ranks_with_tie_breaking():
    desirabilities = {5: 0.2, 3: 0.2, 1: 0.1, 12: 0.9, 8: 0.9}
    potential_profits = {5: 0.8, 3: 0.7, 1: 0.99, 12: 0.8, 8: 0.9}
    ranks = {5: 3, 3: 4, 1: 5, 12: 2, 8: 1}
    assert hlp.calculate_ranks(desirabilities, potential_profits) == ranks

    desirabilities = {5: 0.2, 3: 0.2, 1: 0.1, 12: 0.9, 8: 0.9}
    potential_profits = {5: 0.8, 3: 0.7, 1: 0.99, 12: 0.8, 8: 0.8}
    stakes = {5: 0.8, 3: 0.7, 1: 0.99, 12: 0.8, 8: 0.9}
    ranks = {5: 3, 3: 4, 1: 5, 12: 2, 8: 1}
    assert hlp.calculate_ranks(desirabilities, potential_profits, stakes) == ranks

    desirabilities = {5: 0.2, 3: 0.2, 1: 0.1, 12: 0.9, 8: 0.9}
    potential_profits = {5: 0.8, 3: 0.7, 1: 0.99, 12: 0.8, 8: 0.8}
    stakes = {5: 0.8, 3: 0.7, 1: 0.99, 12: 0.8, 8: 0.8}
    ranks = {5: 3, 3: 4, 1: 5, 12: 2, 8: 1}
    assert hlp.calculate_ranks(desirabilities, potential_profits, stakes, rank_ids=True) == ranks


def test_calculate_cost_per_pool():
    num_pools = 4
    initial_cost = 1
    cost_factor = 0.6
    expected_cost_per_pool = 0.544
    expected_total_cost = 2.176

    cost_per_pool = hlp.calculate_cost_per_pool(num_pools, initial_cost, cost_factor)

    assert cost_per_pool == expected_cost_per_pool
    assert cost_per_pool * num_pools == expected_total_cost


def test_calculate_pool_reward_curve_pledge_benefit():
    # results of options 0 and 4 must be the same when curve_root = 1
    alpha = 0.3
    saturation_point = 0.1
    stakes = [0.01, 0.1, 0.2]
    pledges = [0.001, 0.0069, 0.012]

    results_0 = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=0
    ) for i in range(len(stakes))]

    results_4 = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=4,
        curve_root=1
    ) for i in range(len(stakes))]

    assert results_0 == results_4

    results_4_ = [hlp.calculate_pool_reward(
        stake=stakes[i],
        pledge=pledges[i],
        alpha=alpha,
        beta=saturation_point,
        reward_function_option=4,
        curve_root=3
    ) for i in range(len(stakes))]

    assert results_0 != results_4_
