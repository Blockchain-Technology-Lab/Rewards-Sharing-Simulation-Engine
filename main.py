from logic.sim import Simulation

import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    print("Hello Blockchain World!")
    parser = argparse.ArgumentParser(description='Pooling Games')
    parser.add_argument('--n', type=int, default=100,
                        help='The number of players.')
    parser.add_argument('--k', type=int, default=10,
                        help='The k value of the system.')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='The alpha value of the system.')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='The maximum number of iterations of the system.')
    parser.add_argument('--cost_min', type=float, default=0.001,
                        help='The minimum possible cost for operating a stake pool.')
    parser.add_argument('--cost_max', type=float, default=0.002,
                        help='The maximum possible cost for operating a stake pool.')
    parser.add_argument('--pareto_param', type=float, default=2.0,
                        help='The parameter that determines the shape of the distribution that the stake will be sampled from.')
    parser.add_argument('--player_activation_order', type=str, default='Random',
                        help='Player activation order.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')
    parser.add_argument('--myopic_fraction', type=float, default=0.0,
                        help='The fraction of myopic players in the simulation.')

    args = parser.parse_args()

    sim = Simulation(n=args.n, k=args.k, alpha=args.alpha, max_iterations=args.max_iterations,
                     cost_min=args.cost_min, cost_max=args.cost_max, pareto_param=args.pareto_param,
                     player_activation_order=args.player_activation_order, seed=args.seed, myopic_fraction=args.myopic_fraction)
    sim.run_model(156)

    sim_df = sim.datacollector.get_model_vars_dataframe()

    figures_dir = "figures/"

    pool_nums = sim_df["#Pools"]
    plt.figure()
    pool_nums.plot()
    plt.title("#Pools")
    plt.savefig(figures_dir + "poolCount.png", bbox_inches='tight')

    '''agent_utility = sim.datacollector.get_agent_vars_dataframe()
    #print(agent_utility)
    end_util = agent_utility.xs(2, level="Step")["Utility"]
    plt.figure()
    plt.hist(end_util)
    plt.title("Step 2 utility")
    plt.savefig(figures_dir + "utilityStep2.png", bbox_inches='tight')

    one_agent_util = agent_utility.xs(3, level="AgentID")
    plt.figure()
    plt.plot(one_agent_util)
    plt.title("Agent 3 utility")
    plt.xlabel("Iteration")
    plt.ylabel("Utility")
    plt.savefig(figures_dir + "utilityAgent3.png", bbox_inches='tight')'''

    pool_sizes_by_step = sim_df["PoolSizes"] #todo fix
    print(pool_sizes_by_step)
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
    plt.savefig(figures_dir + "stakePairs.png", bbox_inches='tight')'''

    plt.show()


if __name__ == "__main__":
    main()