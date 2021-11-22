import argparse
import multiprocessing
from mybatchrunner import MyBatchRunner
import time
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

import logic.sim as sim


def main():
    # multiprocessing.freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux / uncomment for windows)

    # single value for fixed params and [lower_bound, upper_bound, step] for variable params
    # note: there needs to be at least one variable parameter
    parser = argparse.ArgumentParser(description='Pooling Games Batch Run')
    parser.add_argument('--execution_id', type=str, default='console-batch-run',
                        help='The identifier of this execution, to be used for naming the output files.')
    parser.add_argument('--seed', default=42,
                        help='Seed for reproducibility (set seed=None if reproducibility is not required). Default is 42.')
    parser.add_argument('--max_iterations', type=int, default=5000,
                        help='The maximum number of iterations of the system. Default is 1000.')
    parser.add_argument('--n', nargs="+", type=int, default=1000,
                        help='The number of players (natural number). Default is 100.')
    parser.add_argument('--k', nargs="+", type=int, default=[100, 200, 2],
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--alpha', nargs="+", type=float, default=0.3,
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--abstention_rate', nargs="+", type=float, default=0,
                        help='The percentage of players that will abstain from the game in this run. Default is 10%%.')

    fixed_params = {
        "execution_id": "temp",
        "player_activation_order": "Random",
        "relative_utility_threshold": 0,
        "myopic_fraction": 0,
        "pool_splitting": True,
        "min_steps_to_keep_pool": 5,
    }
    variable_params = {}
    args_dict = vars(parser.parse_args())
    for arg_name, arg_values in args_dict.items():
        if isinstance(arg_values, list):
            if len(arg_values) > 2:
                int_range = [int(v * 100) for v in arg_values]
                variable_params[arg_name] = [v / 100 for v in range(int_range[0], int_range[1], int_range[2])]
            else:
                fixed_params[arg_name] = arg_values[0]
        else:
            fixed_params[arg_name] = arg_values
    fixed_params.pop("execution_id")
    # specific ranges of alpha that may be useful to check
    # variable_params['alpha'] = [0] + list(np.logspace(-1, 1.5, num=19))
    # variable_params['alpha'] = [v / 100 for v in range(51)] + [v/10 for v in range(10, 55, 5)] + [v for v in range(6,11)]

    print(fixed_params)
    print('-------------')
    print(variable_params)

    model_reporters = {
        "#Pools": sim.get_final_number_of_pools,
        "avgPledge": sim.get_avg_pledge,
        "avg_pools_per_operator": sim.get_avg_pools_per_operator,
        "max_pools_per_operator": sim.get_max_pools_per_operator,
        "median_pools_per_operator": sim.get_median_pools_per_operator,
        "avgSatRate": sim.get_avg_sat_rate,
        "nakamotoCoeff": sim.get_nakamoto_coefficient,
        "StatisticalDistance": sim.get_controlled_stake_distr_stat_dist,
        # "NCR": sim.get_NCR,
        #"MinAggregatePledge": sim.get_min_aggregate_pledge,
        # "pledge_rate": sim.get_pledge_rate,
        "homogeneity_factor": sim.get_homogeneity_factor
    }

    batch_run_MP = MyBatchRunner(
        sim.Simulation,
        # nr_processes=1,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        max_steps=args_dict['max_iterations'],
        iterations=1,
        model_reporters=model_reporters
    )
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    '''pickled_batch_run_object = "batch-run-object.pkl"
    with open(pickled_batch_run_object, "wb") as pkl_file:
        pkl.dump(batch_run_MP, pkl_file)'''

    # Extract data from the batch runner
    run_data_MP = batch_run_MP.get_model_vars_dataframe()
    # print(run_data_MP.head())

    output_dir = "output/22-11-21"
    pickled_batch_run_data = output_dir + "/batch-run-data.pkl"
    with open(pickled_batch_run_data, "wb") as pkl_file:
        pkl.dump(run_data_MP, pkl_file)

    variable_param = list(variable_params.keys())[0]
    colours = [np.random.rand(3, ) for i in range(len(model_reporters))]
    for i, model_reporter in enumerate(model_reporters):
        plot_aggregate_data(run_data_MP, variable_param, model_reporter, colours[i], args_dict["execution_id"], output_dir)

    # ordered dicts with data from each step of each run (the combinations of variable params act as the keys)
    # for example data_collector_model[(0.1, 0.02, 1)] shows the values of the parameters collected at model level
    # when α=0.1 and cost_max=0.02 (which happened to be the run with index 1)
    #data_collector_agents = batch_run_MP.get_collector_agents()
    #data_collector_model = batch_run_MP.get_collector_model()

    plt.show()


def plot_aggregate_data(df, variable_param, model_reporter, color, exec_id, output_dir):
    figures_dir = output_dir + "/figures/"
    plt.figure()
    plt.scatter(df[variable_param], df[model_reporter], color=color)
    plt.xlabel(variable_param)
    plt.ylabel(model_reporter)
    plt.savefig(figures_dir + exec_id + "-" + model_reporter + "-per-" + variable_param + ".png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

