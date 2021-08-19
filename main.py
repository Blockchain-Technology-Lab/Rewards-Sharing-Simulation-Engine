from logic.sim import Simulation
from pathlib import Path

import matplotlib.pyplot as plt
import argparse
import pickle as pkl


def main():
    print("Let the Pooling Games begin!")

    parser = argparse.ArgumentParser(description='Pooling Games')
    parser.add_argument('--n', type=int, default=100,
                        help='The number of players (natural number). Default is 100.')
    parser.add_argument('--k', type=int, default=10,
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--cost_min', type=float, default=0.001,
                        help='The minimum possible cost for operating a stake pool. Default is 0.001.')
    parser.add_argument('--cost_max', type=float, default=0.002,
                        help='The maximum possible cost for operating a stake pool. Default is 0.002.')
    parser.add_argument('--common_cost', type=float, default=0.0001,
                        help='The additional cost that applies to all players for each pool they operate. '
                             'Default is 0.0001.')
    parser.add_argument('--pareto_param', type=float, default=2.0,
                        help='The parameter that determines the shape of the distribution that the stake will be '
                             'sampled from. Default is 2.')
    parser.add_argument('--inertia_ratio', type=float, default=0.1,
                        help='The utility increase ratio under which moves are disregarded. Default is 10%%.')
    parser.add_argument('--player_activation_order', type=str, default='Random',
                        help='Player activation order. Default is random.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility. Default is 42.')
    parser.add_argument("--min_steps_to_keep_pool", type=int, default=5,
                        help='The number of steps for which a player remains idle after opening a pool. Default is 5.')
    parser.add_argument('--myopic_fraction', type=float, default=0.1,
                        help='The fraction of myopic players in the simulation. Default is 0.')
    parser.add_argument('--abstaining_fraction', type=float, default=0.1,
                        help='The percentage of players that will abstain from the game in this run.')
    parser.add_argument('--pool_splitting', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='Are individual players allowed to create multiple pools? Default is yes.')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='The maximum number of iterations of the system. Default is 1000.')
    parser.add_argument('--ms', type=int, default=10,
                        help='The minimum consecutive idle steps that are required to declare convergence. '
                             'Default is 10. But if min_steps_to_keep_pool > ms then ms = min_steps_to_keep_pool + 1. ')

    args = parser.parse_args()

    # todo deal with invalid inputs, e.g. negative n
    # todo make it possible to run more simulations w/o having to rerun the program (e.g. press any key to continue)
    sim = Simulation(n=args.n, k=args.k, alpha=args.alpha,
                     cost_min=args.cost_min, cost_max=args.cost_max, common_cost=args.common_cost,
                     pareto_param=args.pareto_param, inertia_ratio=args.inertia_ratio,
                     player_activation_order=args.player_activation_order,
                     seed=args.seed, min_steps_to_keep_pool=args.min_steps_to_keep_pool,
                     myopic_fraction=args.myopic_fraction, abstaining_fraction=args.abstaining_fraction,
                     pool_splitting=args.pool_splitting, max_iterations=args.max_iterations
                     )

    sim.run_model()

    sim_df = sim.datacollector.get_model_vars_dataframe()

    sim_params = sim.arguments
    current_run_descriptor = "".join(['-' + str(key) + '=' + str(value) for key, value in sim_params.items()
                                      if type(value) == bool or type(value) == int or type(value) == float])[:180]
    figures_dir = "figures/"
    path = Path.cwd() / figures_dir
    Path(path).mkdir(parents=True, exist_ok=True)

    pool_nums = sim_df["#Pools"]
    if sim.schedule.steps >= sim.max_iterations:
        # If the max number of iterations was reached, then we want to analyse the statistic properties of the execution
        filename = figures_dir + "poolCount" + current_run_descriptor + ".pkl"
        with open(filename, "wb") as pkl_file:
            pkl.dump(pool_nums, pkl_file)
    plt.figure()
    pool_nums.plot()
    if sim.schedule.steps < sim.max_iterations:
        equilibrium_step = len(pool_nums) - sim.min_consecutive_idle_steps_for_convergence
        plt.axvline(x=equilibrium_step, label="Equilibrium at step {}".format(equilibrium_step), c='r')
    plt.title("Number of pools over time")
    plt.ylabel("#Pools")
    plt.xlabel("Round")
    plt.legend()
    plt.savefig(figures_dir + "poolCount" + current_run_descriptor + ".png", bbox_inches='tight')

    pool_sizes_by_step = sim_df["PoolSizes"]  # todo fix
    # print(pool_sizes_by_step)
    '''pool_sizes_by_pool = np.array(list(pool_sizes_by_step)).T
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

    avg_pledge = sim_df["AvgPledge"]
    plt.figure()
    avg_pledge.plot(color='r')
    if sim.schedule.steps < sim.max_iterations:
        plt.axvline(x=equilibrium_step, label="Equilibrium at step {}".format(equilibrium_step))
    plt.title("Average pledge over time")
    plt.ylabel("Average pledge")
    plt.xlabel("Round")
    plt.legend()
    plt.savefig(figures_dir + "avgPledge" + current_run_descriptor + ".png", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()  # for profiling the code, comment this line and uncomment the ones below

    '''import cProfile
    from pstats import Stats

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(10)'''
