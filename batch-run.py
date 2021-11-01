from multiprocessing import freeze_support

from mesa.batchrunner import BatchRunner
from mesa.batchrunner import BatchRunnerMP
import time
import matplotlib.pyplot as plt
import pickle as pkl

import logic.sim as sim

if __name__ == '__main__':
    freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux / uncomment for windows)

    fixed_params = {"n": 100, "simulation_id": "temp", "seed": 42}

    # variable_params = {"abstention_rate": [fraction/100 for fraction in range(1, 87)]}
    variable_params = {"k": list(range(1, 30))}
    # variable_params = {"alpha": [a / 100 for a in range(52)]}
    # todo figure out how to run the model with only certain combinations of the variable params

    '''batch_run = BatchRunner(Simulation,
                            variable_params,
                            fixed_params,
                            iterations=1,
                            max_steps=50,
                            model_reporters={"#Pools": get_number_of_pools},
                            display_progress=True)
    start_time = time.time()
    batch_run.run_all()
    print("Batch run without multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    run_data = batch_run.get_model_vars_dataframe()
    print(run_data.head())
    plt.figure()
    plt.scatter(run_data['alpha'], run_data['#Pools'])
    plt.xlabel("α")
    plt.ylabel("#pools")
    plt.show()'''
    # only use the non-multiprocessing batch-run (uncomment the lines above and comment the lines below) if the multiprocessing one doesn't work for some reason

    batch_run_MP = BatchRunnerMP(
        sim.Simulation,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=1,
        model_reporters={
            "#Pools": sim.get_number_of_pools,
            # "avgPledge": sim.get_avg_pledge,
            # "avg_pools_per_operator": sim.get_avg_pools_per_operator,
            # "max_pools_per_operator": sim.get_max_pools_per_operator,
            # "median_pools_per_operator": sim.get_median_pools_per_operator,
            # "avgSatRate": sim.get_avg_sat_rate,
            # "nakamotoCoeff": sim.get_nakamoto_coefficient,
            # "NCR": sim.get_NCR,
            # "MinAggregatePledge": sim.get_min_aggregate_pledge,
            # "pledge_rate": sim.get_pledge_rate,
            # "homogeneity_factor": sim.get_homogeneity_factor,
            "stake_mean_abs_diff": sim.get_controlled_stake_mean_abs_diff,
            "stat_diff": sim.get_controlled_stake_distr_stat_diff
        })
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    # Extract data from the batch runner
    run_data_MP = batch_run_MP.get_model_vars_dataframe()
    # print(run_data_MP.head())

    pickled_batch_run_data = "batch-run-data.pkl"
    with open(pickled_batch_run_data, "wb") as pkl_file:
        pkl.dump(run_data_MP, pkl_file) # todo add id to batch run and include in output file

    figures_dir = "output/figures/"

    '''plt.figure()
    plt.scatter(run_data_MP['abstention_rate'], run_data_MP['#Pools'])
    plt.xlabel("Abstention rate")
    plt.ylabel("Pool count")
    plt.savefig(figures_dir + "pool-count-per-abst-rate.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['abstention_rate'], run_data_MP['homogeneity_factor'], c='blue')
    plt.xlabel("Abstention rate")
    plt.ylabel("Pool homogeneity factor")
    plt.savefig(figures_dir + "homogeneity-per-abst-rate.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['abstention_rate'], run_data_MP['avgSatRate'])
    plt.xlabel("Abstention rate")
    plt.ylabel("Average saturation rate")
    plt.savefig(figures_dir + "abstention-vs-saturation.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['abstention_rate'], run_data_MP['nakamotoCoeff'], c='pink')
    plt.xlabel("Abstention rate")
    plt.ylabel("Nakamoto coefficient")
    plt.savefig(figures_dir + "nakamoto-coeff-per-abst-rate.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['abstention_rate'], run_data_MP['NCR'], c='orange')
    plt.xlabel("Abstention rate")
    plt.ylabel("Nakamoto coefficient rate")
    plt.savefig(figures_dir + "ncr-per-abst-rate.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['abstention_rate'], run_data_MP['MinAggregatePledge'], c='g')
    plt.xlabel("Abstention rate")
    plt.ylabel("Min Aggregate Pledge")
    plt.savefig(figures_dir + "min-aggregate-pledge-per-abst-rate.png", bbox_inches='tight')'''

    plt.figure()
    plt.scatter(run_data_MP['k'], run_data_MP['#Pools'])
    plt.xlabel("k")
    plt.ylabel("#pools")

    plt.figure()
    plt.scatter(run_data_MP['k'], run_data_MP['stake_mean_abs_diff'], c='magenta')
    plt.xlabel("k")
    plt.ylabel("Stake mean absolute difference")
    plt.savefig(figures_dir + "stake_mean_abs_diff-VS-k.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['k'], run_data_MP['stat_diff'], c='c')
    plt.xlabel("k")
    plt.ylabel("Stake distributions statistical difference")
    plt.savefig(figures_dir + "stake_stat_diff-VS-k.png", bbox_inches='tight')

    '''plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['avgPledge'], c='r')
    plt.xlabel("alpha")
    plt.ylabel("Average pledge")'''

    '''plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['nakamotoCoeff'], c='pink')
    plt.xlabel("alpha")
    plt.ylabel("Nakamoto coefficient")
    plt.savefig(figures_dir + "nakamoto-coeff-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['NCR'], c='orange')
    plt.xlabel("alpha")
    plt.ylabel("Nakamoto coefficient rate")
    plt.savefig(figures_dir + "ncr-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['MinAggregatePledge'], c='g')
    plt.xlabel("alpha")
    plt.ylabel("Min Aggregate Pledge")
    plt.savefig(figures_dir + "min-aggregate-pledge-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['pledge_rate'], c='yellow')
    plt.xlabel("alpha")
    plt.ylabel("Pledge rate")
    plt.savefig(figures_dir + "pledge-rate-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['avg_pools_per_operator'], c='g')
    plt.xlabel("alpha")
    plt.ylabel("Average #pools per operator")
    plt.savefig(figures_dir + "avg-pools-per-operator-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['max_pools_per_operator'], c='r')
    plt.xlabel("alpha")
    plt.ylabel("Max #pools per operator")
    plt.savefig(figures_dir + "max-pools-per-operator-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['median_pools_per_operator'], c='purple')
    plt.xlabel("alpha")
    plt.ylabel("Median #pools per operator")
    plt.savefig(figures_dir + "median-pools-per-operator-per-alpha.png", bbox_inches='tight')'''

    plt.show()

    # ordered dicts with data from each step of each run (the combinations of variable params act as the keys)
    # for example data_collector_model[(0.1, 0.02, 1)] shows the values of the parameters collected at model level
    # when α=0.1 and cost_max=0.02 (which happened to be the run with index 1)
    data_collector_agents = batch_run_MP.get_collector_agents()
    data_collector_model = batch_run_MP.get_collector_model()
