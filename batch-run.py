import argparse
import multiprocessing
import random

from mybatchrunner import MyBatchRunner
import time
import numpy as np
from collections import defaultdict

from logic.model_reporters import all_model_reporters
import logic.sim as sim
import logic.helper as hlp


def main():
    # single value for fixed params and [lower_bound, upper_bound, step] for variable params
    # note: there needs to be at least one variable parameter when using batch run
    parser = argparse.ArgumentParser(description='Pooling Games Batch Run')
    parser.add_argument('--execution_id', type=str, default='unnamed-batch-run',
                        help='The identifier of this execution, to be used for naming the output files.')
    parser.add_argument('--seed', default='None',
                        help='Seed for reproducibility. Default is None, which means that a seed will be generated '
                             'randomly and then used for all executions of the batch run.')
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='The maximum number of iterations of the system. Default is 1000.')
    parser.add_argument('--n', nargs="+", type=int, default=1000,
                        help='The number of agents (natural number). Default is 100.')
    parser.add_argument('--k', nargs="+", type=int, default=[100, 501, 100],
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--alpha', nargs="+", type=float, default=0.3,
                        help='The alpha value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--profile_distr', nargs="+", type=float, default=[1, 0, 0],
                        help='The probability distribution for assigning different profiles to the agents. Default is [1, 0, 0], i.e. 100%% non-myopic agents.')
    parser.add_argument('--inactive_stake_fraction', type=float, default=0,
                        help='The fraction of the total stake that remains inactive (does not belong to any of the agents). Default is 0.')
    parser.add_argument('--inactive_stake_fraction_known', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Is the inactive stake fraction of the system known beforehand? Default is no.')
    parser.add_argument('--pareto_param', nargs="+", type=float, default=2.0,
                        help='The shape value of the Pareto distribution for the initial stake allocation.')
    parser.add_argument('--cost_min', nargs="+", type=float, default=1e-4,
                        help='The minimum possible cost for operating a stake pool. Default is 1e-4.')
    parser.add_argument('--cost_max', type=float, default=1e-3,
                        help='The maximum possible cost for operating a stake pool. Default is 2e-3.')
    parser.add_argument('--extra_pool_cost_fraction', nargs="+", type=float, default=0.4,
                        help='The factor that determines how much an additional pool costs as a fraction of '
                             'the original cost value of the stakeholder. Default is 40%%.')
    parser.add_argument('--stake_distr_source', type=str, default='Pareto',
                        help='The distribution type to use for the initial allocation of stake to the agents.')
    parser.add_argument('--reward_function_option', type=int, default=0,
                        help='The reward function to use in the simulation. 0 for the old function, 1 for the new one, '
                             '2 for alternative-1 and 3 for alternative-2.')
    parser.add_argument('--relative_utility_threshold', nargs="+", type=float, default=0,
                        help='The utility increase ratio under which moves are disregarded. Default is 0%%.')
    parser.add_argument('--absolute_utility_threshold', nargs="+", type=float, default=0,
                        help='The utility threshold under which moves are disregarded. Default is 1e-9.')
    parser.add_argument('--pool_opening_process', type=str, default='local-search',
                        help='The heuristic to use for determining a pool strategy. Options: local-search (default), plus-one.')
    parser.add_argument('--iterations_after_convergence', type=int, default=10,
                        help='The minimum consecutive idle iterations that are required before terminations. '
                             'Default is 10. But if min_steps_to_keep_pool > ms then ms = min_steps_to_keep_pool + 1.')

    args_dict = vars(parser.parse_args())

    batch_run_id = args_dict["execution_id"]
    args_dict.pop("execution_id")

    seed = args_dict["seed"]
    args_dict.pop("seed")
    if seed == "None":
        seed = random.randint(0, 9999999)

    fixed_params = {
        "seed": seed
    }
    variable_params = {}

    fixed_params['profile_distr'] = args_dict['profile_distr']
    args_dict.pop('profile_distr')
    for arg_name, arg_values in args_dict.items():
        if isinstance(arg_values, list):
            if len(arg_values) > 2:
                if arg_name == 'alpha' or arg_name == 'cost_min':
                    variable_params[arg_name] = [float(x) for x in np.logspace(arg_values[0], arg_values[1], num=int(arg_values[2]))]
                else:
                    scale_factor = 1e6
                    int_range = [int(v * scale_factor) for v in arg_values]
                    #todo this is the reason why some ints turn to floats (e.g. k, n)
                    variable_params[arg_name] = [v / scale_factor for v in range(int_range[0], int_range[1], int_range[2])]
            else:
                fixed_params[arg_name] = arg_values[0]
        else:
            fixed_params[arg_name] = arg_values

    print("Fixed params: ", fixed_params)
    print('-------------')
    print("Variable params: ", variable_params)

    default_model_reporters = ["Nakamoto coefficient", "Number of pool splitters",
                               #"Cost efficient stakeholders",
                               # "Iterations",
                               "Total pledge"
                               ]
                               #"Gini-id", "Gini-id stake", "Gini-id stake (k)", "Gini-id (k)"]
                               #"Gini-id stake (fraction)", "Gini-id (fraction)"]  # , "Min-aggregate pledge"]
    additional_model_reporters = defaultdict(lambda: [])
    additional_model_reporters['alpha'] = [
        "Mean pledge", #"Total pledge",
        "Max pools per operator", "Median pools per operator",
        "Mean stake rank", "Mean cost rank", "Median stake rank", "Median cost rank"#,
        #"Min-aggregate pledge"
        ]
    additional_model_reporters['k'] = ["Pool count", "Pool homogeneity factor"] #"Statistical distance"
    additional_model_reporters['inactive_stake_fraction'] = ["Pool count", "Pool homogeneity factor"] #"Statistical distance"

    current_additional_model_reporters = []
    for variable_param in variable_params:
        current_additional_model_reporters.extend(additional_model_reporters[variable_param])
    model_reporters = {
        reporter: all_model_reporters[reporter] for reporter in set(default_model_reporters + current_additional_model_reporters)
    }
    print(model_reporters)

    batch_run_MP = MyBatchRunner(
        sim.Simulation,
        # nr_processes=1, # set number of processes to 1 only for debugging purposes
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        max_steps=args_dict['max_iterations'] + 1,
        iterations=1,
        model_reporters=model_reporters,
        execution_id=batch_run_id
    )
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    # Extract data from the batch runner
    batch_run_data = batch_run_MP.get_model_vars_dataframe()

    # ordered dicts with data from each step of each run (the combinations of variable params act as the keys)
    # for example data_collector_model[(0.1, 0.02, 1)] shows the values of the parameters collected at model level
    # when α=0.1 and cost_max=0.02 (which happened to be the run with index 1)
    #data_collector_agents = batch_run_MP.get_collector_agents()
    #data_collector_model = batch_run_MP.get_collector_model()

    rng = np.random.default_rng(seed=156)
    random_colours = rng.random((len(all_model_reporters), 3))
    all_reporter_colours = dict(zip(all_model_reporters.keys(), random_colours))
    all_reporter_colours['Mean pledge'] = 'red'
    all_reporter_colours["Pool count"] = 'C0'
    all_reporter_colours["Total pledge"] = 'purple'
    all_reporter_colours["Nakamoto coefficient"] = 'pink'

    for variable_param in variable_params:
        useLogAxis = True if variable_param == 'alpha' else False
        for model_reporter in model_reporters:
            hlp.plot_aggregate_data(batch_run_data, variable_param, model_reporter, all_reporter_colours[model_reporter],
                                batch_run_id, batch_run_MP.directory, log_axis=useLogAxis)

    vp = list(variable_params.keys())
    if len(vp) == 2:
        # plot heatmap when we have two variable parameters
        hlp.plot_aggregate_data_heatmap(batch_run_data, vp, model_reporters, batch_run_MP.directory)


if __name__ == "__main__":
    #multiprocessing.freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux or mac / uncomment for windows) #todo is that still needed? if yes, detect os and act accordingly
    main()