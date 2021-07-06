from logic.sim import Simulation

import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Hello Blockchain World!")
    sim = Simulation(n=100, k=10)
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

    pool_sizes_by_step = sim_df["Pool"]
    pool_sizes_by_pool = np.array(list(pool_sizes_by_step)).T
    plt.figure()
    plt.stackplot(range(len(pool_sizes_by_step)), pool_sizes_by_pool)
    plt.title("Pool dynamics")
    plt.xlabel("Iteration")
    plt.ylabel("Stake")
    plt.savefig(figures_dir + "poolDynamics.png", bbox_inches='tight')

    last_stakes = sim_df["StakePairs"].iloc[-1]
    x = last_stakes['x']
    y = last_stakes['y']
    plt.figure()
    plt.scatter(x, y)
    plt.title("Owner stake vs pool stake")
    plt.xlabel("Pool owner stake")
    plt.ylabel("Pool stake")
    plt.savefig(figures_dir + "stakePairs.png", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()