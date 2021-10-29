import logic.sim as simulation

import pathlib
import matplotlib.pyplot as plt
import argparse
import pickle as pkl


def main():
    print("Let the Pooling Games begin!")

    parser = argparse.ArgumentParser(description='Pooling Games')
    parser.add_argument('--n', type=int, default=100,
                        help='The number of players (natural number). Default is 100.')
    parser.add_argument('--k', nargs="+", type=int, default=10,
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--alpha', nargs="+", type=float, default=0.3,
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--cost_min', type=float, default=0.001,
                        help='The minimum possible cost for operating a stake pool. Default is 0.001.')
    parser.add_argument('--cost_max', type=float, default=0.002,
                        help='The maximum possible cost for operating a stake pool. Default is 0.002.')
    parser.add_argument('--common_cost', nargs="+", type=float, default=0.0001,
                        help='The additional cost that applies to all players for each pool they operate. '
                             'Default is 0.0001.')
    parser.add_argument('--pareto_param', type=float, default=2.0,
                        help='The parameter that determines the shape of the distribution that the stake will be '
                             'sampled from. Default is 2.')
    parser.add_argument('--relative_utility_threshold', nargs="+", type=float, default=0.1,
                        help='The utility increase ratio under which moves are disregarded. Default is 10%%.')
    parser.add_argument('--absolute_utility_threshold', nargs="+", type=float, default=1e-9,
                        help='The utility threshold under which moves are disregarded. Default is 1e-9.')
    parser.add_argument('--player_activation_order', type=str, default='Random',
                        help='Player activation order. Default is random.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility. Default is 42.')
    parser.add_argument("--min_steps_to_keep_pool", type=int, default=5,
                        help='The number of steps for which a player remains idle after opening a pool. Default is 5.')
    parser.add_argument('--myopic_fraction', nargs="+", type=float, default=0.1,
                        help='The fraction of myopic players in the simulation. Default is 10%%.')
    parser.add_argument('--abstention_rate', nargs="+", type=float, default=0.1,
                        help='The percentage of players that will abstain from the game in this run. Default is 10%%.')
    parser.add_argument('--pool_splitting', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='Are individual players allowed to create multiple pools? Default is yes.')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='The maximum number of iterations of the system. Default is 1000.')
    parser.add_argument('--ms', type=int, default=10,
                        help='The minimum consecutive idle steps that are required to declare convergence. '
                             'Default is 10. But if min_steps_to_keep_pool > ms then ms = min_steps_to_keep_pool + 1. ')
    parser.add_argument('--simulation_id', type=str, default='',
                        help='An optional identifier for the specific simulation run, '
                             'which will be included in the output.')

    args = parser.parse_args()

    # todo deal with invalid inputs, e.g. negative n
    # todo make it possible to run more simulations w/o having to rerun the program (e.g. press any key to continue)

    sim = simulation.Simulation(
        n=args.n,
        k=args.k,
        alpha=args.alpha,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        common_cost=args.common_cost,
        pareto_param=args.pareto_param,
        relative_utility_threshold=args.relative_utility_threshold,
        absolute_utility_threshold=args.absolute_utility_threshold,
        player_activation_order=args.player_activation_order.capitalize(),
        seed=args.seed,
        min_steps_to_keep_pool=args.min_steps_to_keep_pool,
        myopic_fraction=args.myopic_fraction,
        abstention_rate=args.abstention_rate,
        pool_splitting=args.pool_splitting,
        max_iterations=args.max_iterations,
        ms=args.ms,
        simulation_id=args.simulation_id
    )

    sim.run_model()

    sim_df = sim.datacollector.get_model_vars_dataframe()

    simulation_id = args.simulation_id
    if simulation_id == '':
        # No identifier was provided by the user, so we construct one based on the simulation's parameter values
        simulation_id = "".join(['-' + str(key) + '=' + str(value) for key, value in sim.arguments.items()
                                 if type(value) == bool or type(value) == int or type(value) == float])[:180]

    '''pickled_simulation_filename = "simulation-object-" + simulation_id + ".pkl"
    with open(pickled_simulation_filename, "wb") as pkl_file:
        pkl.dump(sim, pkl_file)'''

    output_dir = "output/"
    figures_dir = "output/figures/"
    path = pathlib.Path.cwd() / figures_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    pool_nums = sim_df["#Pools"]
    if sim.schedule.steps >= sim.max_iterations:
        # If the max number of iterations was reached, then we save the data about the pool numbers
        # in order to later analyse the statistic properties of the execution
        filename = output_dir + simulation_id + "-poolCount" + ".pkl"
        with open(filename, "wb") as pkl_file:
            pkl.dump(pool_nums, pkl_file)



        # todo how about multiple equilibria? show them all or only last one?
    equilibrium_step = len(pool_nums) - sim.min_consecutive_idle_steps_for_convergence if sim.schedule.steps < sim.max_iterations else -1

    plot_line(simulation_id, sim_df["#Pools"], 'cyan', "Number of pools over time", "Round",
              "#Pools", "poolCount", equilibrium_step, True)

    '''pool_sizes_by_step = sim_df["PoolSizes"]  # todo fix
    # print(pool_sizes_by_step)
    pool_sizes_by_pool = np.array(list(pool_sizes_by_step)).T
    print(pool_sizes_by_pool)
    plt.figure()
    plt.stackplot(range(len(pool_sizes_by_step)), pool_sizes_by_pool)
    plt.title("Pool dynamics")
    plt.xlabel("Iteration")
    plt.ylabel("Stake")
    plt.savefig(figures_dir + "poolDynamics.png", bbox_inches='tight')'''

    '''last_stakes = sim_df["StakePairs"].iloc[-1]
    x = last_stakes['x']
    y = last_stakes['y']
    plt.figure()
    plt.scatter(x, y)
    plt.title("Owner stake vs pool stake")
    plt.xlabel("Pool owner stake")
    plt.ylabel("Pool stake")
    plt.savefig(figures_dir + "stakePairs" + current_run_descriptor + ".png", bbox_inches='tight')'''


    plot_line(simulation_id, sim_df["AvgPledge"], 'red', "Average pledge over time", "Round",
              "Average pledge", "avgPledge", equilibrium_step, True)

    plot_line(simulation_id, sim_df["TotalPledge"], 'purple', "Total pledge over time", "Round",
              "Total pledge", "totalPledge", equilibrium_step, True)

    plot_line(simulation_id, sim_df["MeanAbsDiff"], 'green', "Mean Absolute Difference of Controlled Stake", "Round",
              "Mean abs diff", "meanAbsDiff", equilibrium_step)

def plot_line(simulation_id, data, color, title, x_label, y_label, filename, equilibrium_step, show_equilibrium=False):
    figures_dir = "output/figures/"
    path = pathlib.Path.cwd() / figures_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    plt.figure()
    data.plot(color=color)
    if show_equilibrium:
        #if sim.schedule.steps < sim.max_iterations:
        plt.axvline(x=equilibrium_step, label="Equilibrium reached", c='indigo') #todo if it exceeds max iterations??
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(figures_dir + simulation_id[:10] + "-" + filename + ".png", bbox_inches='tight')

    #plt.show()


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
    # main_with_profiling()
