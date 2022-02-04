import argparse
import multiprocessing
import pathlib
import random

from mybatchrunner import MyBatchRunner
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from math import floor, log10
from collections import defaultdict

import logic.sim as sim


def main():
    # multiprocessing.freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux / uncomment for windows)

    # single value for fixed params and [lower_bound, upper_bound, step] for variable params
    # note: there needs to be at least one variable parameter
    parser = argparse.ArgumentParser(description='Pooling Games Batch Run')
    parser.add_argument('--execution_id', type=str, default='unnamed-batch-run',
                        help='The identifier of this execution, to be used for naming the output files.')
    parser.add_argument('--seed', default='None',
                        help='Seed for reproducibility. Default is None, which means that a seed will be generated '
                             'randomly and then used for al executions of the batch run.')
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='The maximum number of iterations of the system. Default is 1000.')
    parser.add_argument('--n', nargs="+", type=int, default=1000,
                        help='The number of players (natural number). Default is 100.')
    parser.add_argument('--k', nargs="+", type=int, default=[100, 200, 2],
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--alpha', nargs="+", type=float, default=0.3,
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--abstention_rate', nargs="+", type=float, default=0,
                        help='The percentage of players that will abstain from the game in this run. Default is 10%%.')
    parser.add_argument('--cost_min', type=float, default=1e-4,
                        help='The minimum possible cost for operating a stake pool. Default is 1e-4.')
    parser.add_argument('--cost_max', type=float, default=1e-3,
                        help='The maximum possible cost for operating a stake pool. Default is 2e-3.')
    parser.add_argument('--cost_factor', nargs="+", type=float, default=0.6,
                        help='The factor that determines how much an additional pool costs. '
                             'Default is 60%.')
    parser.add_argument('--stake_distr_type', type=str, default='Pareto',
                        help='The distribution type to use for the initial allocation of stake to the players.')
    parser.add_argument('--extra_cost_type', type=str, default='fixed_fraction',
                        help='The method used to calculate the cost of any additional pool.')
    args_dict = vars(parser.parse_args())

    batch_run_id = args_dict["execution_id"]
    args_dict.pop("execution_id")

    seed = args_dict["seed"]
    args_dict.pop("seed")
    if seed == "None":
        seed = random.randint(0, 9999999)
    batch_run_id += '-seed-' + str(seed)

    fixed_params = {
        "execution_id": "temp",
        "player_activation_order": "Random",
        "relative_utility_threshold": 0,
        "myopic_fraction": 0,
        "pool_splitting": True,
        "min_steps_to_keep_pool": 5,
        "seed": seed
    }
    variable_params = {}

    for arg_name, arg_values in args_dict.items():
        if isinstance(arg_values, list):
            if len(arg_values) > 2:
                if arg_name == 'alpha':
                    variable_params[arg_name] = list(np.logspace(arg_values[0], arg_values[1], num=int(arg_values[2])))
                else:
                    scale_factor = 1e6
                    int_range = [int(v * scale_factor) for v in arg_values]
                    variable_params[arg_name] = [v / scale_factor for v in range(int_range[0], int_range[1], int_range[2])]
            else:
                fixed_params[arg_name] = arg_values[0]
        else:
            fixed_params[arg_name] = arg_values

    print("Fixed params: ", fixed_params)
    print('-------------')
    print("Variable params: ", variable_params)
    
    variable_param = list(variable_params.keys())[0]

    # note that any new model reporters should be added to the end, to maintain the colour allocation
    all_model_reporters = {
        "Pool count": sim.get_final_number_of_pools,
        "Average pledge": sim.get_avg_pledge,
        "Total pledge": sim.get_total_pledge,
        "Average pools per operator": sim.get_avg_pools_per_operator,
        "Max pools per operator": sim.get_max_pools_per_operator,
        "Median pools per operator": sim.get_median_pools_per_operator,
        "Average saturation rate": sim.get_avg_sat_rate,
        "Nakamoto coefficient": sim.get_nakamoto_coefficient,
        "Statistical distance": sim.get_controlled_stake_distr_stat_dist,
        "Nakamoto coefficient rate": sim.get_NCR,
        "Min aggregate pledge": sim.get_min_aggregate_pledge,
        "Pledge rate": sim.get_pledge_rate,
        "Homogeneity factor": sim.get_homogeneity_factor,
        "Iterations": sim.get_iterations,
        "Average stake rank": sim.get_avg_stk_rnk,
        "Average cost rank": sim.get_avg_cost_rnk
    }

    rng = np.random.default_rng(seed=156)
    random_colours = rng.random((len(all_model_reporters), 3))
    all_reporter_colours = dict(zip(all_model_reporters.keys(), random_colours))

    default_model_reporters = ["Pool count", "Nakamoto coefficient", "Iterations"] #, "MinAggregatePledge"]
    additional_model_reporters = defaultdict(lambda: [])
    additional_model_reporters['alpha'] = [
            "Average pledge", "Total pledge", "Max pools per operator", "Median pools per operator",
            "Average stake rank", "Average cost rank"
        ]
    additional_model_reporters['k'] = ["Statistical distance", "Homogeneity factor"]
    additional_model_reporters['abstention_rate'] = ["Statistical distance", "Homogeneity factor"]

    
    model_reporters = {
        k: all_model_reporters[k] for k in set(default_model_reporters + additional_model_reporters[variable_param])
    }

    batch_run_MP = MyBatchRunner(
        sim.Simulation,
        # nr_processes=1, # set number of processes to 1 only for debugging purposes
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        max_steps=args_dict['max_iterations'],
        iterations=1,
        model_reporters=model_reporters,
        execution_id=batch_run_id
    )
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    # Extract data from the batch runner
    run_data_MP = batch_run_MP.get_model_vars_dataframe()

    output_dir = "output/"
    day = time.strftime("%d-%m-%Y")
    day_output_dir = output_dir + day
    path = pathlib.Path.cwd() / day_output_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    batch_run_data = day_output_dir + "/batch-run-data-" + batch_run_id + ".csv"
    run_data_MP.to_csv(batch_run_data, index=False)

    if variable_param == 'alpha':
        # determine which alpha values were good, from the ones that were tried
        # we define a suitable value for alpha by taking the minimum value that gives NCR >= 3%
        min_ncr = 0.03
        min_nc = min_ncr * fixed_params['n']
        if min_nc > fixed_params['k'] :
            min_nc = int(fixed_params['k']/3)
        suitable_rows = run_data_MP[run_data_MP['Nakamoto coefficient'] >= min_nc]
        min_alpha_suitable_row = suitable_rows[suitable_rows.alpha == suitable_rows.alpha.min()][['n','k', 'cost_min', 'alpha']]
        cost_alpha_csv_file = output_dir + "/cost-alpha.csv"
        with open(cost_alpha_csv_file, "a") as f:
            min_alpha_suitable_row.to_csv(f, mode='a', index=False, header=f.tell()==0)

    useLogAxis = True if variable_param == 'alpha' else False
    for i, model_reporter in enumerate(model_reporters):
        plot_aggregate_data(run_data_MP, variable_param, model_reporter, all_reporter_colours[model_reporter], batch_run_id, day_output_dir, log_axis=useLogAxis)

    # ordered dicts with data from each step of each run (the combinations of variable params act as the keys)
    # for example data_collector_model[(0.1, 0.02, 1)] shows the values of the parameters collected at model level
    # when α=0.1 and cost_max=0.02 (which happened to be the run with index 1)
    #data_collector_agents = batch_run_MP.get_collector_agents()
    #data_collector_model = batch_run_MP.get_collector_model()


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision) if coeff > 1 else r"$10^{{{0:d}}}$".format(exponent)

def plot_aggregate_data(df, variable_param, model_reporter, color, exec_id, output_dir, positive_only=True, log_axis=False):
    figures_dir = output_dir + "/figures/"
    path = pathlib.Path.cwd() / figures_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    plt.figure()

    if positive_only:
        df = df[df[model_reporter] >= 0]
    x = df[variable_param]
    y = df[model_reporter]
    plt.scatter(x, y, color=color)
    plt.xlabel(variable_param)
    plt.ylabel(model_reporter)
    if log_axis:
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda t, _: t if t >= 0.1 else sci_notation(t, 0)))
        plt.minorticks_off()

    plt.savefig(figures_dir + exec_id + "-" + model_reporter + "-per-" + variable_param + ".png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

