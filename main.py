import logic.sim as simulation
from logic.helper import *

import pathlib
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import time
import numpy as np
import seaborn as sns

sns.set_theme()

def main():
    print("Let the Pooling Games begin!")

    parser = argparse.ArgumentParser(description='Pooling Games')
    parser.add_argument('--n', type=int, default=1000,
                        help='The number of agents (natural number). Default is 1000.')
    parser.add_argument('--k', nargs="+", type=int, default=100,
                        help='The k value of the system (natural number). Default is 100.')
    parser.add_argument('--alpha', nargs="+", type=float, default=0.3,
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--cost_min', type=float, default=1e-5,
                        help='The minimum possible cost for operating a stake pool. Default is 1e-4.')
    parser.add_argument('--cost_max', type=float, default=1e-4,
                        help='The maximum possible cost for operating a stake pool. Default is 1e-3.')
    parser.add_argument('--cost_factor', nargs="+", type=float, default=0.4,
                        help='The factor that determines how much an additional pool costs. '
                             'Default is 40%%.')
    parser.add_argument('--pareto_param', type=float, default=2.0,
                        help='The parameter that determines the shape of the distribution that the stake will be '
                             'sampled from. Default is 2.')
    parser.add_argument('--relative_utility_threshold', nargs="+", type=float, default=0,
                        help='The utility increase ratio under which moves are disregarded. Default is 0%%.')
    parser.add_argument('--absolute_utility_threshold', nargs="+", type=float, default=1e-9,
                        help='The utility threshold under which moves are disregarded. Default is 1e-9.')
    parser.add_argument('--agent_activation_order', type=str, default='Random',
                        help='agent activation order. Default is random.')
    parser.add_argument('--seed', default=None,
                        help='Seed for reproducibility. Default is None, which means that no seed is given.')
    parser.add_argument("--min_steps_to_keep_pool", type=int, default=5,
                        help='The number of steps for which a agent remains idle after opening a pool. Default is 5.')
    parser.add_argument('--myopic_fraction', nargs="+", type=float, default=0,
                        help='The fraction of myopic agents in the simulation. Default is 0%%.')
    parser.add_argument('--abstention_rate', type=float, default=0,
                        help='The fraction of the total stake that remains inactive. Default is 0.')
    parser.add_argument('--pool_splitting', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='Are individual agents allowed to create multiple pools? Default is yes.')
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='The maximum number of iterations of the system. Default is 2000.')
    parser.add_argument('--ms', type=int, default=10,
                        help='The minimum consecutive idle steps that are required to declare convergence. '
                             'Default is 10. But if min_steps_to_keep_pool > ms then ms = min_steps_to_keep_pool + 1.')
    parser.add_argument('--stake_distr_source', type=str, default='pareto',
                        help='The distribution type to use for the initial allocation of stake to the agents.')
    parser.add_argument('--extra_cost_type', type=str, default='fixed_fraction',
                        help='The method used to calculate the cost of any additional pool.')
    parser.add_argument('--execution_id', type=str, default='',
                        help='An optional identifier for the specific simulation run, '
                             'which will be included in the output.')
    parser.add_argument('--reward_function_option', type=int, default=0,
                        help='The reward function to use in the simulation. 0 for the old function, 1 for the new one, '
                             '2 for alternative-1 and 3 for alternative-2.')
    parser.add_argument('--input_from_file', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='If True then the input is read from a file (args.json) and any other command line '
                             'arguments are discarded. Default is False.')
    args = parser.parse_args()

    # todo deal with invalid inputs, e.g. negative n
    # todo make it possible to run more simulations w/o having to rerun the program (e.g. press any key to continue)

    sim = simulation.Simulation(
        n=args.n,
        k=args.k,
        alpha=args.alpha,
        stake_distr_source=args.stake_distr_source,
        myopic_fraction=args.myopic_fraction,
        abstention_rate=args.abstention_rate,
        #abtsention known
        relative_utility_threshold=args.relative_utility_threshold,
        absolute_utility_threshold=args.absolute_utility_threshold,
        min_steps_to_keep_pool=args.min_steps_to_keep_pool,
        pool_splitting=args.pool_splitting,
        seed=args.seed,
        pareto_param=args.pareto_param,
        max_iterations=args.max_iterations,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        cost_factor=args.cost_factor,
        agent_activation_order=args.agent_activation_order.capitalize(),
        #total stake
        ms=args.ms,
        extra_cost_type=args.extra_cost_type,
        reward_function_option = args.reward_function_option,
        execution_id=args.execution_id,
        #seq_id
        #parent_dir
        input_from_file=args.input_from_file
    )

    sim.run_model()

    sim_df = sim.datacollector.get_model_vars_dataframe()
    execution_id = sim.execution_id

    execution_dir = sim.directory
    figures_dir = execution_dir / "figures"
    pathlib.Path(figures_dir).mkdir(parents=True, exist_ok=True)

    equilibrium_steps = sim.equilibrium_steps
    pivot_steps = sim.pivot_steps
    print('equilibrium steps: ', equilibrium_steps)
    print('pivot steps: ', pivot_steps)

    plot_line(execution_id, sim_df["Max pools per operator"], 'orange', "MaxPoolsPerAgent", "Round",
              "Max pools per operator", "Max pools per operator", equilibrium_steps, pivot_steps, figures_dir, True)

    plot_line(execution_id, sim_df["Pool count"], 'C0', "Number of pools over time", "Round",
              "Pool count", "poolCount", equilibrium_steps, pivot_steps, figures_dir, True)

    plot_line(execution_id, sim_df["Average pledge"], 'red', "Average pledge over time", "Round",
              "Average pledge", "Average pledge", equilibrium_steps, pivot_steps, figures_dir, True)

    plot_line(execution_id, sim_df["Total pledge"], 'purple', "Total pledge over time", "Round",
              "Total pledge", "Total pledge", equilibrium_steps, pivot_steps, figures_dir, True)

    plot_line(execution_id, sim_df["Mean stake rank"], 'green', "Avg stake rank of operators over time", "Round",
              "Mean stake rank", "Mean stake rank", equilibrium_steps, pivot_steps, figures_dir, True)

    plot_line(execution_id, sim_df["Mean cost rank"], 'brown', "Avg cost rank of operators over time", "Round",
              "Avg cost rank", "Mean cost rank", equilibrium_steps, pivot_steps, figures_dir, True)

    pool_sizes_by_step = sim_df['Stake per agent']
    pool_sizes_by_pool = np.array(list(pool_sizes_by_step)).T
    plt.figure(figsize=(10,5))
    col = sns.color_palette("hls", 77)
    plt.stackplot(range(1, len(pool_sizes_by_step)), pool_sizes_by_pool[:,1:], colors=col, edgecolor='black', lw=0.1)
    plt.xlim(xmin=0.0)
    plt.xlabel("Round")
    plt.ylabel("Stake per Operator")
    filename = "poolDynamics-" + execution_id + ".png"
    plt.savefig(figures_dir / filename, bbox_inches='tight')


def plot_line(execution_id, data, color, title, x_label, y_label, filename, equilibrium_steps, pivot_steps,
              path, show_equilibrium=False):

    equilibrium_colour = 'mediumseagreen'
    pivot_colour = 'gold'

    plt.figure(figsize=(10,5))
    data.plot(color=color)
    if show_equilibrium:
        for i, step in enumerate(equilibrium_steps):
            label = "Equilibrium reached" if i == 0 else ""
            plt.axvline(x=step, label=label, c=equilibrium_colour)  # todo if it exceeds max iterations??
    for i, step in enumerate(pivot_steps):
        label = "Parameter change" if i == 0 else ""
        plt.plot(step, data[step], 'x', label=label, c=pivot_colour)
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    filename = execution_id + "-" + filename + ".png"
    plt.savefig(path / filename , bbox_inches='tight')


def main_with_profiling():
    import cProfile
    from pstats import Stats

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(10)


if __name__ == "__main__":
    main()  # for profiling the code, comment this line and uncomment the one below
    '''main_with_profiling()
    cache_funcs = [calculate_potential_profit, calculate_pool_reward, #calculate_cost_per_pool, calculate_myopic_pool_desirability,
                   calculate_delegator_reward_from_pool, calculate_operator_reward_from_pool, calculate_cost_per_pool_fixed_fraction,
                   calculate_pool_desirability, calculate_pool_stake_NM_from_rank, determine_pledge_per_pool]
    for func in cache_funcs:
        print(func.__name__,': ', func.cache_info())'''
