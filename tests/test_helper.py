import helper as hlp
import numpy as np


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


def test_is_list_flat():
    assert hlp.isListFlat([1, 2, 3]) is True
    assert hlp.isListFlat([[1, 1], [2, 2]]) is False


def test_flatten_list():
    assert hlp.flatten_list([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
    assert hlp.flatten_list([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    # note: only works with homogeneous nested lists of one level, so lists such as [[1, [2, 3]], [4, 5]] would not be properly flattened


'''
# examples taken from wikipedia: https://en.wikipedia.org/wiki/Softmax_function#Example
def test_softmax():
    assert (np.round(hlp.softmax([1, 2, 3, 4, 1, 2, 3]), 3) == np.array(
        [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175])).all()
    assert (np.round(hlp.softmax([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]), 3) == np.array(
        [0.125, 0.138, 0.153, 0.169, 0.125, 0.138, 0.153])).all()
'''


def test_calculate_rank():
    desirabilities = [0.2, 0.3, 0.1, 0.9, 0.8]
    ranks = [3, 2, 4, 0, 1]
    for i, rank in enumerate(ranks):
        assert hlp.calculate_rank(desirabilities, i) == ranks[i]
