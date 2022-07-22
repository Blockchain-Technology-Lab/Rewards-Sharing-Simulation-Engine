# -*- coding: utf-8 -*-
from numpy.random import default_rng
import numpy as np
from scipy import stats
import csv
import pathlib
from math import sqrt, floor, log10
from functools import lru_cache
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

sns.set_theme()

TOTAL_EPOCH_REWARDS_R = 1
MAX_NUM_POOLS = 1000
MIN_STAKE_UNIT = 2.2e-17 #todo change to reflect how much 1 lovelace is depending on total stake?
MIN_COST_PER_POOL = 1e-6

#todo make this read from file only?
def read_stake_distr_from_file(num_agents=10000, seed=42):
    default_filename = 'synthetic-stake-distribution-10000-agents.csv'
    filename = 'synthetic-stake-distribution-' + str(num_agents) + '-agents.csv'
    stk_dstr = []
    try:
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                stk_dstr.append(float(row[0]))
    except FileNotFoundError:
        try:
            with open(default_filename) as file:
                reader = csv.reader(file)
                for row in reader:
                    stk_dstr.append(float(row[0]))
        except FileNotFoundError:
            print("Couldn't find file to read stake distribution from. Please make sure that the file exists or "
                  "use another option for the stake distribution source.")
            raise
    if num_agents == len(stk_dstr):
        return stk_dstr
    rng = default_rng(seed=int(seed))
    if num_agents < len(stk_dstr):
        return rng.choice(stk_dstr, num_agents, replace=False)
    return rng.choice(stk_dstr, num_agents, replace=True)

def generate_sake_distr_disparity(n, x=0.3, c=3):
    stk_dstr = []
    high_end_stake = x / c
    low_end_stake = (1 - x) / (n - c)
    stk_dstr.extend([high_end_stake for _ in range(c)])
    stk_dstr.extend([low_end_stake for _ in range(n-c)])

    print(sum(stk_dstr))
    print(max(stk_dstr))
    print(min(stk_dstr))
    return stk_dstr

def generate_cost_distr_disparity(n, low, high, c=10):
    costs = []
    costs.extend([high for _ in range(c)])
    costs.extend([low for _ in range(n-c)])
    return costs


def generate_stake_distr_pareto(num_agents, pareto_param=2, seed=156, truncation_factor=-1, total_stake=-1):
    """
    Generate a distribution for the agents' initial stake (wealth),
    sampling from a Pareto distribution
    sampling from a Pareto distribution
    :param pareto_param: the shape parameter to be used for the Pareto distribution
    :param num_agents: the number of samples to draw
    :param total_stake: the sum to normalize all stakes to, if positive
    :return:
    """
    rng = default_rng(seed=int(seed))
    # Sample from a Pareto distribution with the specified shape
    a, m = pareto_param, 1.  # shape and mode
    stake_sample = list((rng.pareto(a, num_agents) + 1) * m)
    if truncation_factor > 0:
        stake_sample = truncate_pareto(rng, (a, m), stake_sample, truncation_factor)
    if total_stake > 0:
        stake_sample = normalize_distr(stake_sample, normal_sum=total_stake)
    return stake_sample


def truncate_pareto(rng, pareto_params, sample, truncation_factor):
    a, m = pareto_params
    while 1:
        # rejection sampling to ensure that the distribution is truncated
        max_value = max(sample)
        if max_value > sum(sample) / truncation_factor:
            sample.remove(max_value)
            sample.append((rng.pareto(a) + 1) * m)
        else:
            return sample


def generate_stake_distr_flat(num_agents, total_stake=1):
    stake_per_agent = total_stake / num_agents if num_agents > 0 else 0
    return [stake_per_agent for _ in range(num_agents)]


def generate_cost_distr_unfrm(num_agents, low, high, seed=156):
    """
    Generate a distribution for the agents' costs of operating pools,
    sampling from a uniform distribution
    :param num_agents:
    :param low:
    :param high:
    :return:
    """
    rng = default_rng(seed=int(seed))
    costs = rng.uniform(low=low, high=high, size=num_agents)
    return costs

def generate_cost_distr_bands(num_agents, low, high, num_bands, seed=156):
    rng = default_rng(seed=seed)
    bands = rng.uniform(low=low, high=high, size=num_bands)
    costs = rng.choice(bands, num_agents)
    return costs

def generate_cost_distr_bands_manual(num_agents, low, high, num_bands, seed=156):
    low_cost_agents = 5
    rng = default_rng(seed=seed)
    bands = rng.uniform(low=low, high=high, size=num_bands)
    costs = rng.choice(bands, num_agents-3)
    costs = np.append(costs, [low for _ in range(3)])

    low_cost_agents = 5
    common_cost = high#(low + high) / 2
    costs = [low if i < low_cost_agents else common_cost for i in range(num_agents)]
    return costs


def generate_cost_distr_nrm(num_agents, low, high, mean, stddev):
    """
    Generate a distribution for the agents' costs of operating pools,
    sampling from a truncated normal distribution
    """
    costs = stats.truncnorm.rvs(low, high,
                                loc=mean, scale=stddev,
                                size=num_agents)
    return costs


def normalize_distr(dstr, normal_sum=1):
    """
    returns an equivalent distribution where the sum equals 1 (or another value defined by normal_sum)
    :param dstr:
    :param normal_sum:
    :return:
    """
    s = sum(dstr)
    if s == 0:
        return dstr
    nrm_dstr = [normal_sum * i / s for i in dstr]
    flt_error = normal_sum - sum(nrm_dstr)
    nrm_dstr[-1] += flt_error
    return nrm_dstr

@lru_cache(maxsize=1024)
def calculate_potential_profit(pledge, cost, alpha, beta, reward_function_option, total_stake):
    """
    Calculate a pool's potential profit, which can be defined as the profit it would get at saturation level
    :param pledge:
    :param cost:
    :param alpha:
    :param beta:
    :return: float, the maximum possible profit that this pool can yield
    """
    relative_stake = beta / total_stake
    relative_pledge = pledge / total_stake
    potential_reward = calculate_pool_reward(relative_stake, relative_pledge, alpha, beta, reward_function_option, total_stake)
    return potential_reward - cost

@lru_cache(maxsize=1024)
def calculate_current_profit(stake, pledge, cost, alpha, beta, reward_function_option, total_stake):
    relative_pledge = pledge / total_stake
    relative_stake = stake / total_stake
    reward = calculate_pool_reward(relative_stake, relative_pledge, alpha, beta, reward_function_option, total_stake)
    return reward - cost

@lru_cache(maxsize=1024)
def calculate_pool_reward(relative_stake, relative_pledge, alpha, beta, reward_function_option, total_stake, curve_root=3, crossover_factor=8):
    beta = beta / total_stake
    if reward_function_option == 0:
        return calculate_pool_reward_old(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 1:
        return calculate_pool_reward_new(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 2:
        return calculate_pool_reward_flat_pledge_benefit(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 3:
        return calculate_pool_reward_new_sqrt(relative_stake, relative_pledge, alpha, beta)
    elif reward_function_option == 4:
        return calculate_pool_reward_curve_pledge_benefit(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor, total_stake)
    elif reward_function_option == 5:
        return calculate_pool_reward_curve_pledge_benefit_min_first(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor)
    elif reward_function_option == 6:
        return calculate_pool_reward_curve_pledge_benefit_no_min(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor)
    elif reward_function_option == 7:
        return calculate_pool_reward_CIP_50(relative_stake, relative_pledge, beta, L=alpha)
    else:
        raise ValueError("Invalid option for reward function.")

#todo change names
def calculate_pool_reward_old(relative_stake, relative_pledge, alpha, relative_beta):
    pledge_ = min(relative_pledge, relative_beta)
    stake_ = min(relative_stake, relative_beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (stake_ + (pledge_ * alpha * ((stake_ - pledge_ * (1 - stake_ / relative_beta)) / relative_beta)))
    return r

def calculate_pool_reward_new(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * stake_ * (1 + (alpha * pledge_ / beta))
    return r

def calculate_pool_reward_flat_pledge_benefit(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (stake_ + alpha * pledge_)
    return r

def calculate_pool_reward_new_sqrt(relative_stake, relative_pledge, alpha, beta):
    pledge_ = min(relative_pledge, beta)
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * stake_ * (1 + (alpha * sqrt(pledge_) / beta))
    return r

def calculate_pool_reward_curve_pledge_benefit(relative_stake, relative_pledge, alpha, relative_beta, curve_root,
                                               crossover_factor, total_stake):
    crossover = relative_beta * total_stake / crossover_factor
    pledge = relative_pledge * total_stake
    pledge_factor = (pledge ** (1 / curve_root)) * (crossover ** ((curve_root - 1) / curve_root)) / total_stake
    return calculate_pool_reward_old(relative_stake, pledge_factor, alpha, relative_beta)

def calculate_pool_reward_curve_pledge_benefit_min_first(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor):
    crossover = beta / crossover_factor
    pledge = min(relative_pledge, beta)
    pledge_ = (pledge ** (1 / curve_root)) * (crossover ** ((curve_root - 1) / curve_root))
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (
                stake_ + (pledge_ * alpha * ((stake_ - pledge_ * (1 - stake_ / beta)) / beta)))
    return r

def calculate_pool_reward_curve_pledge_benefit_no_min(relative_stake, relative_pledge, alpha, beta, curve_root, crossover_factor):
    crossover = beta / crossover_factor
    pledge_ = (relative_pledge ** (1 / curve_root)) * (crossover ** ((curve_root - 1) / curve_root))
    stake_ = min(relative_stake, beta)
    r = (TOTAL_EPOCH_REWARDS_R / (1 + alpha)) * (
                stake_ + (pledge_ * alpha * ((stake_ - pledge_ * (1 - stake_ / beta)) / beta)))
    return r

def calculate_pool_reward_CIP_50(relative_stake, relative_pledge, beta, L=100):
    pledge_factor = L * relative_pledge
    r = TOTAL_EPOCH_REWARDS_R * min(relative_stake, pledge_factor, beta)
    return r

@lru_cache(maxsize=1024)
def calculate_delegator_reward_from_pool(pool_margin, pool_cost, pool_reward, delegator_stake_fraction):
    margin_factor = (1 - pool_margin) * delegator_stake_fraction
    pool_profit = pool_reward - pool_cost
    r_d = max(margin_factor * pool_profit, 0)
    return r_d

@lru_cache(maxsize=1024)
def calculate_operator_reward_from_pool(pool_margin, pool_cost, pool_reward, operator_stake_fraction):
    margin_factor = pool_margin + ((1 - pool_margin) * operator_stake_fraction)
    pool_profit = pool_reward - pool_cost
    return pool_profit if pool_profit <= 0 else pool_profit * margin_factor

def calculate_pool_stake_NM(pool, pool_rankings, beta, k):
    """
    Calculate the non-myopic stake of a pool, given the pool and the state of the system (other active pools)
    :param pool:
    :param pool_rankings:
    :param beta: the saturation point of the system
    :param k: the desired number of pools of the system
    :return: the value of the non-myopic stake of the pool with id pool_id
    """
    rank_in_top_k = pool_rankings.index(pool) < k #todo would it be faster to make list of ids and check for id?
    return calculate_pool_stake_NM_from_rank(pool_pledge=pool.pledge, pool_stake=pool.stake, beta=beta, rank_in_top_k=rank_in_top_k)

#todo do I still need this?
def calculate_ranks(ranking_dict, *tie_breaking_dicts, rank_ids=True):
    """
    Rank the values of a dictionary from highest to lowest (highest value gets rank 1, second highest rank 2 and so on)
    @param ranking_dict:
    @param tie_breaking_dicts:
    @param rank_ids: if True, then the lowest id (e.g. the one corresponding to a pool created earlier) takes precedence
                    during ties that persist even after the other tie breaking rules have been applied.
                    If False and ties still exist, then the tie breaking is arbitrary.
    @return: dictionary with the item id as the key and the calculated rank as the value
    """
    if rank_ids:
        tie_breaking_dicts = list(tie_breaking_dicts)
        tie_breaking_dicts.append({key: -key for key in ranking_dict.keys()})
    final_ranking_dict = {
        key:
            (ranking_dict[key],) + tuple(tie_breaker_dict[key] for tie_breaker_dict in tie_breaking_dicts)
        for key in ranking_dict
    }
    ranks = {
        sorted_item[0]: i + 1 for i, sorted_item in
        enumerate(sorted(final_ranking_dict.items(), key=lambda item: item[1], reverse=True))
    }
    return ranks

def save_as_latex_table(df, sim_id, output_dir):
    path = pathlib.Path.cwd() / output_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with open(output_dir + sim_id + "-output.tex", 'w', newline='') as file:
        df.to_latex(file, index=False)

def generate_execution_id(args_dict):
    num_args_to_use = 5
    max_characters = 100
    return "-".join([str(key) + '-' + str(value)
                     if not isinstance(value, list)
                     else str(key) + '-' + "-".join([str(v) for v in value])
                     for key, value in list(args_dict.items())[:num_args_to_use]])[:max_characters]

@lru_cache(maxsize=1024)
def calculate_cost_per_pool(num_pools, initial_cost, cost_factor):
    """
    Calculate the average cost of an agent's pools, assuming that any additional pool costs less than the first one
    Specifically, if the first pool costs c1 and we use a factor of 0.6 then any subsequent pool would cost c2 = 0.6 * c1
    @param num_pools:
    @param initial_cost:
    @param cost_factor:
    @return:
    """
    return (initial_cost + (num_pools - 1) * cost_factor * initial_cost) / num_pools

@lru_cache(maxsize=1024)
def calculate_suitable_margin(potential_profit, target_desirability):
    m = 1 - target_desirability / potential_profit if potential_profit > 0 else 0
    return max(m, 0)

@lru_cache(maxsize=1024)
def calculate_pool_desirability(margin, potential_profit):
    return max((1 - margin) * potential_profit, 0)

@lru_cache(maxsize=1024)
def calculate_myopic_pool_desirability(margin, current_profit):
    return max((1 - margin) * current_profit, 0)

@lru_cache(maxsize=1024)
def calculate_operator_utility_from_pool(pool_stake, pledge, margin, cost, alpha, beta, reward_function_option, total_stake):
    relative_pool_stake = pool_stake / total_stake
    relative_pledge = pledge / total_stake
    r = calculate_pool_reward(relative_pool_stake, relative_pledge, alpha, beta, reward_function_option, total_stake)
    stake_fraction = pledge / pool_stake
    return calculate_operator_reward_from_pool(pool_margin=margin, pool_cost=cost, pool_reward=r, operator_stake_fraction=stake_fraction)


@lru_cache(maxsize=1024)
def calculate_pool_stake_NM_from_rank(pool_pledge, pool_stake, beta, rank_in_top_k):
    return max(beta, pool_stake) if rank_in_top_k else pool_pledge

@lru_cache(maxsize=1024)
def determine_pledge_per_pool(agent_stake, beta, num_pools):
    """
    The agents choose to allocate their entire stake as the pledge of their pools,
    so they divide it equally among them
    However, if they saturate all their pools with pledge and still have remaining stake,
    then they don't allocate all of it to their pools, as a pool with such a pledge above saturation
     would yield suboptimal rewards
    :return: list of pledge values
    """
    if num_pools <= 0:
        raise ValueError("Agent tried to calculate pledge for zero or less pools.")
    return min(agent_stake / num_pools, beta)


def export_csv_file(rows, filepath):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def export_json_file(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=lambda x: str(x))

def read_args_from_file(filepath):
    try:
        with open(filepath) as f:
            args = json.load(f)
        return args
    except FileNotFoundError:
        print("Couldn't find file to read parameter values from. Please make sure that the file exists under the name 'args.json'.")
        raise
    except ValueError:
        print("Invalid format for file 'args.json'.")
        raise

def read_seq_id(filename='sequence.dat'):
    try:
        with open(filename, 'r') as f:
            seq = int(f.read())
    except FileNotFoundError:
        seq = 0
    return seq

def write_seq_id(seq, filename='sequence.dat'):
    with open(filename, 'w') as f:
        f.write(str(seq))

def write_to_csv(filepath, header, row):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell()==0:
            writer.writerow(header)
        writer.writerow(row)

def plot_line(data, execution_id, color, x_label, y_label, filename, equilibrium_steps, pivot_steps,
              path, title='', show_equilibrium=False):
    equilibrium_colour = 'mediumseagreen'
    pivot_colour = 'gold'

    fig = plt.figure(figsize=(10,5))
    data.plot(color=color)
    if show_equilibrium:
        for i, step in enumerate(equilibrium_steps):
            label = "Equilibrium reached" if i == 0 else ""
            plt.axvline(x=step, label=label, c=equilibrium_colour)
    for i, step in enumerate(pivot_steps):
        label = "Parameter change" if i == 0 else ""
        plt.plot(step, data[step], 'x', label=label, c=pivot_colour)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    filename = execution_id + "-" + filename + ".png"
    plt.savefig(path / filename , bbox_inches='tight')
    plt.close(fig)

def plot_stack_area_chart(pool_sizes_by_step, execution_id, path):
    pool_sizes_by_agent = np.array(list(pool_sizes_by_step)).T
    fig = plt.figure(figsize=(10, 5))
    col = sns.color_palette("hls", 77)
    plt.stackplot(range(1, len(pool_sizes_by_step)), pool_sizes_by_agent[:, 1:], colors=col, edgecolor='black', lw=0.1)
    plt.xlim(xmin=0.0)
    plt.xlabel("Round")
    plt.ylabel("Stake per Operator")
    filename = "poolDynamics-" + execution_id + ".png"
    plt.savefig(path / filename, bbox_inches='tight')
    plt.close(fig)

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10 ** exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent,
                                                  precision) if coeff > 1 else r"$10^{{{0:d}}}$".format(exponent)

def plot_aggregate_data(df, variable_param, model_reporter, color, exec_id, output_dir, positive_only=True,
                        log_axis=False):
    path = output_dir / "figures"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    if positive_only:
        df = df[df[model_reporter] >= 0]
    x = df[variable_param]
    y = df[model_reporter]
    plt.scatter(x, y, color=color)
    plt.xlabel(variable_param)
    plt.ylabel(model_reporter)
    if log_axis:
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda t, _: t if t >= 0.1 else sci_notation(t, 0)))
        plt.minorticks_off()
    # plt.legend()
    filename = exec_id + "-" + model_reporter + "-per-" + variable_param + ".png"
    plt.savefig(path / filename, bbox_inches='tight')
    plt.close(fig)



def plot_aggregate_data_2(df, variable_param, model_reporter, output_dir, colour_values='blue', labels = None, legend_param=None,
                        positive_only=True, log_axis=False):
    import matplotlib.colors as colors

    path = output_dir / "figures"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    plt.figure()
    if positive_only:
        df = df[df[model_reporter] >= 0]
    x = df[variable_param]
    y = df[model_reporter]
    scatter = plt.scatter(x, y, c=colour_values, norm=colors.Normalize(vmin=colour_values.min(), vmax=colour_values.max()), cmap='viridis') #if log_axis else colors.LogNorm(vmin=colour_values.min(), vmax=colour_values.max()))
    # note: lognorm doesn't work when 0 is one of the values
    plt.xlabel(variable_param)
    if variable_param == 'alpha':
        plt.xlabel('α')
    plt.ylabel(model_reporter)
    if log_axis:
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda t, _: t if t >= 0.1 else sci_notation(t, 0)))
        plt.minorticks_off()
    labels = [round(value,3) if value > 1e-3 else value for value in labels]
    #plt.legend(handles=scatter.legend_elements()[0], labels=labels, title=legend_param)
    if legend_param == 'pareto_param':
        legend_param = 'Pareto shape parameter'
    elif legend_param == 'cost_factor':
        legend_param = 'φ'
    plt.colorbar(label=legend_param)

    filename = model_reporter + "-per-" + variable_param + ".png"
    plt.savefig(path / filename, bbox_inches='tight')


def plot_aggregate_data_heatmap(df, variables, model_reporters, output_dir):
    path = output_dir / "figures"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for reporter in model_reporters:
        fig = plt.figure()
        cols_to_keep = variables + [reporter]
        df_ = df[cols_to_keep]
        df_.set_index(variables, inplace=True)
        unstacked_df = df_[reporter].unstack(level=0)

        cbar_kws = {"orientation": "vertical",
                    # "shrink":1,
                    'extend': 'max',
                    # 'extendfrac':0.1,
                    # "ticks":np.arange(min_NC+1, initial_NC+1, step),
                    # "drawedges":True,
                    "label": reporter
                    }

        ax = sns.heatmap(unstacked_df, annot=True, annot_kws={"size": 12}, fmt='g', cbar_kws=cbar_kws, cmap='flare')

        ax.invert_yaxis()
        labels = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticklabels([str(round(float(label), 4)) for label in labels])

        filename = 'heatmap-' + reporter + "-" + '-'.join(variables) + ".png"
        plt.savefig(path / filename, bbox_inches='tight')
        plt.close(fig)


    def convert_currency(dollar_value, ada_price):
        return dollar_value / ada_price

def utility_from_profitable_pool(r, c, l, b, m):
    return l / b * (r - c) * (1 - m) + m * (r - c)

def util_by_margin_and_pools(agent, margin, num_pools):
    total_stake = agent.model.total_stake
    stake = agent.stake / total_stake
    alpha = agent.model.alpha
    k = agent.model.k
    beta = agent.model.beta / total_stake
    R = TOTAL_EPOCH_REWARDS_R
    phi = agent.model.cost_factor
    initial_cost = agent.cost

    top_k_des = [pool.desirability if pool is not None else 0 for pool in agent.model.pool_rankings][:k]
    top_k_des.reverse()

    pledge_per_pool = np.where(stake / num_pools < beta, stake / num_pools, beta)
    cost_per_pool = (1 + phi * num_pools - phi) * initial_cost / num_pools

    reward_per_pool = R / (1 + alpha) * (beta + pledge_per_pool * alpha)
    utility_per_pool = np.where(reward_per_pool - cost_per_pool > 0,
                                utility_from_profitable_pool(reward_per_pool, cost_per_pool, pledge_per_pool, beta,
                                                             margin), reward_per_pool - cost_per_pool)
    desirability = (1 - margin) * (reward_per_pool - cost_per_pool)

    margin_len = int(len(margin) / k)
    d_cutoff = np.array(top_k_des*margin_len)
    utility = np.where(desirability >= d_cutoff + num_pools * 0.00001, num_pools * utility_per_pool, 0)
    return utility


def plot_margin_pools_heatmap(agent):
    from matplotlib import cm

    k = agent.model.k

    x = np.linspace(1, k, k)
    # x = np.linspace(1, 10, 10)
    y = np.linspace(0, 0.25, 1000)
    X, Y = np.meshgrid(x, y)
    zs = np.array(util_by_margin_and_pools(agent, np.ravel(Y), np.ravel(X)))
    Z = zs.reshape(X.shape)

    mappable = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    mappable.set_array(Z)

    fig = plt.figure(figsize=(6, 5))
    ax2 = fig.add_subplot(111)
    ax2.set_ylabel('margin')
    ax2.set_xlabel('number of owned top-k pools')

    Z = np.ma.array(Z, mask=(Z == 0))
    masked_cmap = mappable.cmap.copy()
    masked_cmap.set_bad(color='black')

    ax2.imshow(Z, cmap=masked_cmap, norm=mappable.norm, interpolation='none', aspect='auto',
               origin='lower', extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)))
    plt.grid()

    plt.colorbar(mappable, label='utility')
    filename = 'heatmap-round-' + str(agent.model.schedule.steps) + '-agent-' + str(agent.unique_id) + '.png'
    plt.savefig(agent.model.directory / filename, bbox_inches='tight')
    plt.close(fig)

def calculate_pool_splitting_profit(alpha, phi, cost, relative_stake):
    return (1 + alpha) * (1 - phi) * cost - TOTAL_EPOCH_REWARDS_R * relative_stake * alpha

def sort_pools(pool):
    if pool is None:
        return 0, 0, 0
    # sort pools based on their desirability
    # break ties with potential profit and further ties with pool id
    return -pool.desirability, -pool.potential_profit, pool.id

def sort_pools_myopic(pool):
    if pool is None:
        return 0, 0, 0
    # sort pools based on their myopic desirability
    # break ties with pool id
    return -pool.myopic_desirability, pool.id