import random

import pytest
import logic.helper as hlp

from logic.pool import Pool


def test_generate_stake_distr():
    stk_distr = hlp.generate_stake_distr_pareto(num_agents=100, pareto_param=2)

    assert len(stk_distr) == 100

    stk_distr = hlp.generate_stake_distr_pareto(num_agents=1001, pareto_param=1.5, total_stake=1)

    assert len(stk_distr) == 1001
    assert sum(stk_distr) == 1


def test_generate_stake_distr_flat():
    stk_distr = hlp.generate_stake_distr_flat(num_agents=100, total_stake=1)

    rnd_idx = random.randint(1, 100)

    assert pytest.approx(stk_distr[rnd_idx]) == 0.01
    assert len(stk_distr) == 100
    assert pytest.approx(sum(stk_distr)) == 1


def test_normalize_distr():
    sample_dstr = [10, 8, 5, 5, 1, 0.5, 0.1]

    nrm_dstr = hlp.normalize_distr(sample_dstr)
    assert sum(nrm_dstr) == 1

    nrm_dstr = hlp.normalize_distr(sample_dstr, normal_sum=156)
    assert sum(nrm_dstr) == 156


def test_calculate_pool_reward_variable_stake():
    # GIVEN
    a0 = 0.3
    saturation_point = 0.1
    stakes = [0.01, 0.1, 0.2]
    pledges = [0.01, 0.01, 0.01]
    total_stake = 1

    # WHEN
    results = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=0,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    # THEN
    assert results[0] < results[1] == results[2]


def test_calculate_pool_reward_variable_pledge():
    a0 = 0.3
    saturation_point = 0.1
    stakes = [0.1, 0.1, 0.1]
    pledges = [0.01, 0.05, 0.1]
    total_stake = 1

    results = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=0,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    assert results[0] < results[1] < results[2]


def test_calculate_pool_reward_variable_stake_a0_zero():
    # GIVEN
    a0 = 0
    saturation_point = 0.1
    stakes = [0.01, 0.1, 0.2]
    pledges = [0.01, 0.01, 0.01]
    total_stake = 1

    # WHEN
    results = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=0,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    # THEN
    assert results[0] < results[1] == results[2]


def test_calculate_pool_reward_variable_pledge_a0_zero():
    a0 = 0
    saturation_point = 0.1
    stakes = [0.1, 0.1, 0.1]
    pledges = [0.01, 0.05, 0.1]
    total_stake = 1

    results = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=0,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    assert results[0] == results[1] == results[2]


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
    extra_pool_cost_fraction = 0.6
    expected_cost_per_pool = 0.544
    expected_total_cost = 2.176

    cost_per_pool = hlp.calculate_cost_per_pool(num_pools, initial_cost, extra_pool_cost_fraction)

    assert cost_per_pool == expected_cost_per_pool
    assert cost_per_pool * num_pools == expected_total_cost


def test_calculate_pool_reward_curve_pledge_benefit():
    # results of options 0 and 4 must be the same when curve_root = 1
    a0 = 0.3
    saturation_point = 0.1
    stakes = [0.01, 0.1, 0.2]
    pledges = [0.001, 0.0069, 0.012]
    total_stake = 1

    results_0 = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=0,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    results_4 = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=4,
        curve_root=1,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    assert results_0 == results_4

    results_4_ = [hlp.calculate_pool_reward(
        relative_stake=stakes[i] / total_stake,
        relative_pledge=pledges[i] / total_stake,
        a0=a0,
        beta=saturation_point,
        reward_function=4,
        curve_root=3,
        total_stake=total_stake
    ) for i in range(len(stakes))]

    assert results_0 != results_4_


def test_calculate_pool_stake_nm():
    a0 = 0.3
    beta = 0.1
    k = 10
    reward_func = 0
    total_stake = 1

    pools = {i: Pool(pool_id=i, cost=0.0001, pledge=0.001, owner=i, a0=a0, beta=beta,
                     reward_function=reward_func, total_stake=total_stake, margin=0) for i in range(1, 11)}

    # pool does not belong in the top k, so stake_nm = pledge
    pool_11 = Pool(pool_id=11, cost=0.0001, pledge=0.001, owner=11, a0=a0, beta=beta,
                   reward_function=reward_func, total_stake=total_stake, margin=0.2)
    pools[11] = pool_11
    ranks = list(pools.values())
    ranks.sort(key=hlp.pool_comparison_key)
    pool_stake_nm = hlp.calculate_pool_stake_NM(pool=pool_11, pool_rankings=ranks, beta=beta, k=k)
    assert pool_stake_nm == 0.001

    # pool belongs in the top k and pool_stake < beta, so stake_nm = beta
    pool_11 = Pool(pool_id=11, cost=0.0001, pledge=0.002, owner=11, a0=a0, beta=beta,
                   reward_function=reward_func, total_stake=total_stake, margin=0)
    pools[11] = pool_11
    ranks = list(pools.values())
    ranks.sort(key=hlp.pool_comparison_key)
    pool_stake_nm = hlp.calculate_pool_stake_NM(pool=pool_11, pool_rankings=ranks, beta=beta, k=k)
    assert pool_stake_nm == 0.1

    # pool belongs in the top k and pool_stake > beta, so stake_nm = pool_stake
    pool_11 = Pool(pool_id=11, cost=0.0001, pledge=0.2, owner=11, a0=a0, beta=beta,
                   reward_function=reward_func, total_stake=total_stake, margin=0)
    pools[11] = pool_11
    ranks = list(pools.values())
    ranks.sort(key=hlp.pool_comparison_key)
    pool_stake_nm = hlp.calculate_pool_stake_NM(pool=pool_11, pool_rankings=ranks, beta=beta, k=k)
    assert pool_stake_nm == 0.2

    # pool doesn't belong in the top k because of (id) tie breaking, so stake_nm = pool_pledge
    pool_11 = Pool(pool_id=11, cost=0.0001, pledge=0.001, owner=11, a0=a0, beta=beta,
                   reward_function=reward_func, total_stake=total_stake, margin=0)
    pools[11] = pool_11
    ranks = list(pools.values())
    ranks.sort(key=hlp.pool_comparison_key)
    pool_stake_nm = hlp.calculate_pool_stake_NM(pool=pool_11, pool_rankings=ranks, beta=beta, k=k)
    assert pool_stake_nm == 0.001

    # there are less than k pools, so pool necessarily in the top k
    k = 100 # increase k
    beta = 0.01
    pool_11 = Pool(pool_id=11, cost=0.001, pledge=0.00001, owner=11, a0=a0, beta=beta,
                   reward_function=reward_func, total_stake=total_stake, margin=0)
    pools[11] = pool_11
    ranks = list(pools.values())
    ranks.sort(key=hlp.pool_comparison_key)
    pool_stake_nm = hlp.calculate_pool_stake_NM(pool=pool_11, pool_rankings=ranks, beta=beta, k=k)
    assert pool_stake_nm == 0.01


def test_read_stake_distr_from_file():
    # case 1: file exists and n == rows
    assert True

    # case 2: file exists and n < rows
    assert True

    # case 3: file exists and n > rows
    assert True

    # case 4: file does not exist
    filename = 'fake-filename'
    with pytest.raises(FileNotFoundError) as e_info:
        hlp.read_stake_distr_from_file(filename=filename, num_agents=1000)
    assert e_info.type == FileNotFoundError


def test_write_read_seq_id():
    hlp.write_seq_id(seq=555, filename='test-sequence.dat')

    seq_id = hlp.read_seq_id(filename='test-sequence.dat')
    assert seq_id == 555


def test_calculate_pool_reward_cip_50():
    theta = 100
    relative_stake = 0.01
    relative_pledge = 0.001
    k = 100
    beta = 1/k

    r = hlp.calculate_pool_reward_CIP_50(relative_stake, relative_pledge, beta, theta)

    assert r == 0.01

