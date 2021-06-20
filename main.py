from sim import Simulation

import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Hello Blockchain World!")
    sim = Simulation(n=10, k=3)
    sim.run_model(156)

    sim_df = sim.datacollector.get_model_vars_dataframe()

    plt.figure()
    pool_nums = sim_df["#Pools"]
    pool_nums.plot()
    plt.title("#Pools")
    plt.show()

    agent_utility = sim.datacollector.get_agent_vars_dataframe()
    print(agent_utility)
    end_util = agent_utility.xs(15, level="Step")["Utility"]
    plt.figure()
    plt.hist(end_util)
    plt.title("Step 15 utility")
    plt.show()

    plt.figure()
    one_agent_util = agent_utility.xs(3, level="AgentID")
    plt.plot(one_agent_util)
    plt.title("Agent 3 utility")
    plt.xlabel("Step")
    plt.ylabel("Utility")
    plt.show()

    plt.figure()
    pool_sizes_by_step = sim_df["Pool"]
    pool_sizes_by_pool = np.array(list(pool_sizes_by_step)).T
    plt.stackplot(range(len(pool_sizes_by_step)), pool_sizes_by_pool)
    plt.title("Pool sizes")
    plt.show()

    #todo save relevant charts as images in a dedicated directory


if __name__ == "__main__":
    main()