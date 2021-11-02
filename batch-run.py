import argparse
import multiprocessing
from mybatchrunner import MyBatchRunner
import time
import matplotlib.pyplot as plt
import pickle as pkl

import logic.sim as sim


def main():
    # multiprocessing.freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux / uncomment for windows)

    # single value for fixed params and [lower_bound, upper_bound, step] for variable params
    # note: there needs to be at least one variable parameter
    parser = argparse.ArgumentParser(description='Pooling Games Batch Run')
    parser.add_argument('--n', nargs="+", type=int, default=100,
                        help='The number of players (natural number). Default is 100.')
    parser.add_argument('--k', nargs="+", type=int, default=10,
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--alpha', nargs="+", type=float, default=[0, 5, 0.05],
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--abstention_rate', nargs="+", type=float, default=0.1,
                        help='The percentage of players that will abstain from the game in this run. Default is 10%%.')

    fixed_params = {"simulation_id": "temp", "seed": 42}
    variable_params = {}
    args_dict = vars(parser.parse_args())
    for arg_name, arg_values in args_dict.items():
        if type(arg_values) is list:
            if len(arg_values) > 2:
                int_range = [int(v*100) for v in arg_values]
                variable_params[arg_name] = [v/100 for v in range(int_range[0], int_range[1], int_range[2])]
            else:
                fixed_params[arg_name] = arg_values[0]
        else:
            fixed_params[arg_name] = arg_values

    # variable_params = {"abstention_rate": [fraction/100 for fraction in range(1, 87)]}
    # variable_params = {"alpha": [a / 100 for a in range(49)]}
    # todo figure out how to run the model with only certain combinations of the variable params

    print(fixed_params)
    print('-------------')
    print(variable_params)

    batch_run_MP = MyBatchRunner(sim.Simulation,
                                 variable_parameters=variable_params,
                                 fixed_parameters=fixed_params,
                                 iterations=1,
                                 model_reporters={
                                     #"#Pools": sim.get_final_number_of_pools
                                     "avgPledge": sim.get_avg_pledge,
                                     #"avg_pools_per_operator": sim.get_avg_pools_per_operator,
                                     #"max_pools_per_operator": sim.get_max_pools_per_operator,
                                     #"median_pools_per_operator": sim.get_median_pools_per_operator,
                                     # "avgSatRate": sim.get_avg_sat_rate,
                                     "nakamotoCoeff": sim.get_nakamoto_coefficient,
                                     #"NCR": sim.get_NCR,
                                     "MinAggregatePledge": sim.get_min_aggregate_pledge
                                     #"pledge_rate": sim.get_pledge_rate,
                                     # "homogeneity_factor": sim.get_homogeneity_factor
                                 })
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    '''pickled_batch_run_object = "batch-run-object.pkl"
    with open(pickled_batch_run_object, "wb") as pkl_file:
        pkl.dump(batch_run_MP, pkl_file)'''

    # Extract data from the batch runner
    run_data_MP = batch_run_MP.get_model_vars_dataframe()
    # print(run_data_MP.head())

    pickled_batch_run_data = "batch-run-data.pkl"
    with open(pickled_batch_run_data, "wb") as pkl_file:
        pkl.dump(run_data_MP, pkl_file)

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

    '''plt.figure()
    plt.scatter(run_data_MP['k'], run_data_MP['#Pools'])
    plt.xlabel("k")
    plt.ylabel("#pools")
    plt.savefig(figures_dir + "pool-count-per-k.png", bbox_inches='tight')'''

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['avgPledge'], c='r')
    plt.xlabel("alpha")
    plt.ylabel("Average pledge")
    plt.savefig(figures_dir + "avg-pledge-per-alpha.png", bbox_inches='tight')

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['nakamotoCoeff'], c='pink')
    plt.xlabel("alpha")
    plt.ylabel("Nakamoto coefficient")
    plt.savefig(figures_dir + "nakamoto-coeff-per-alpha.png", bbox_inches='tight')

    '''plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['NCR'], c='orange')
    plt.xlabel("alpha")
    plt.ylabel("Nakamoto coefficient rate")
    plt.savefig(figures_dir + "ncr-per-alpha.png", bbox_inches='tight')'''

    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['MinAggregatePledge'], c='g')
    plt.xlabel("alpha")
    plt.ylabel("Min Aggregate Pledge")
    plt.savefig(figures_dir + "min-aggregate-pledge-per-alpha.png", bbox_inches='tight')

    '''plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['pledge_rate'], c='yellow')
    plt.xlabel("alpha")
    plt.ylabel("Pledge rate")
    plt.savefig(figures_dir + "pledge-rate-per-alpha.png", bbox_inches='tight')'''

    '''plt.figure()
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


if __name__ == '__main__':
    main()

