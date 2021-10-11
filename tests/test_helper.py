import helper as hlp


# todo add more tests

def test_generate_stake_distr():
    assert True


def test_generate_cost_distr():
    assert True


def test_normalize_distr():
    assert True


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
        beta=saturation_point
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
        beta=saturation_point
    ) for i in range(len(stakes))]

    assert results[0] < results[1] < results[2]


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


