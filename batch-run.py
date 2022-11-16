import argparse
import random
import pandas as pd

from custom_batchrunner import custom_batch_run
import time
import numpy as np

import logic.sim as sim
import logic.helper as hlp
from logic.model_reporters import ALL_MODEL_REPORTEERS

if __name__ == "__main__":
    #multiprocessing.freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux or mac / uncomment for windows) #todo is that still needed? if yes, detect os and act accordingly

    # single value for fixed params and [lower_bound, upper_bound, step] for variable params
    parser = argparse.ArgumentParser(description='Pooling Games Batch Run')
    parser.add_argument('--execution_id', nargs="?", type=str, default='batch-run',
                        help='The identifier of this execution, to be used for naming the output files.')
    parser.add_argument('--seed', nargs="?", type=int, default=None,
                        help='Seed for reproducibility. Default is None, which means that a seed will be generated '
                             'randomly and then used for all executions of the batch run.')
    parser.add_argument('--max_iterations', nargs="?", type=int, default=2000,
                        help='The maximum number of iterations of the system. Default is 1000.')
    parser.add_argument('--n', nargs="*", type=int, default=1000,
                        help='The number of agents (natural number). Default is 100.')
    parser.add_argument('--k', nargs="*", type=int, default=[100, 200, 300],
                        help='The k value of the system (natural number). Default is 10.')
    parser.add_argument('--a0', nargs="*", type=float, default=0.3,
                        help='The a0 value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--inactive_stake_fraction', nargs="*", type=float, default=0,
                        help='The fraction of the total stake that remains inactive (does not belong to any of the agents). Default is 0.')
    parser.add_argument('--inactive_stake_fraction_known', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Is the inactive stake fraction of the system known beforehand? Default is no.')
    parser.add_argument('--pareto_param', nargs="*", type=float, default=2.0,
                        help='The shape value of the Pareto distribution for the initial stake allocation.')
    parser.add_argument('--cost_min', nargs="*", type=float, default=1e-5,
                        help='The minimum possible cost for operating a stake pool. Default is 1e-4.')
    parser.add_argument('--cost_max', nargs="*", type=float, default=1e-4,
                        help='The maximum possible cost for operating a stake pool. Default is 2e-3.')
    parser.add_argument('--extra_pool_cost_fraction', nargs="*", type=float, default=0.4,
                        help='The factor that determines how much an additional pool costs as a fraction of '
                             'the original cost value of the stakeholder. Default is 40%%.')
    parser.add_argument('--stake_distr_source', nargs="?", type=str, default='Pareto',
                        help='The distribution type to use for the initial allocation of stake to the agents.')
    parser.add_argument('--reward_scheme', nargs="?", type=int, default=0, choices=range(4),
                        help='The reward scheme to use in the simulation. 0 for the original reward scheme of Cardano, '
                             '1 for a simplified version of it, 2 for a reward scheme with flat pledge benefit, 3 for '
                             'a reward scheme with curved pledge benefit (CIP-7) and 4 for the reward scheme of CIP-50.')
    parser.add_argument('--relative_utility_threshold', nargs="+", type=float, default=0,
                        help='The utility increase ratio under which moves are disregarded. Default is 0%%.')
    parser.add_argument('--absolute_utility_threshold', nargs="+", type=float, default=0,
                        help='The utility threshold under which moves are disregarded. Default is 1e-9.')
    parser.add_argument('--iterations_after_convergence', type=int, default=10,
                        help='The minimum consecutive idle iterations that are required before terminations. '
                             'Default is 10.')
    parser.add_argument('--agent_profile_distr', nargs="*", type=float, default=[1, 0, 0],
                        help='The probability distribution for assigning different profiles to the agents. Default is [1, 0, 0], i.e. 100%% non-myopic agents.')

    # args missing from here: agent_profile_distr, generate graphs, metrics, input_from_file, agent_activation_order
    # make sure that all args are fine with batch run

    args_dict = vars(parser.parse_args())

    batch_run_id = args_dict["execution_id"]
    args_dict.pop("execution_id")

    seed = args_dict["seed"]
    args_dict.pop("seed")
    if seed is None:
        seed = random.randint(0, 9999999)

    args_dict.pop("agent_profile_distr") #todo deal with this instead of popping

    params = {}
    variable_params = {}

    for arg_name, arg_value in args_dict.items():
        params[arg_name] = arg_value
        if isinstance(arg_value, list) and len(arg_value) > 1:
            variable_params[arg_name] = arg_value

    print("Variable parameter(s): ", variable_params)

    start_time = time.time()
    results, batch_run_directory = custom_batch_run(
        sim.Simulation,
        parameters=params,
        iterations=1, #todo maybe add as command-line option
        max_steps=params['max_iterations'],
        number_processes=None,
        data_collection_period=-1,
        display_progress=True,
        batch_run_id=batch_run_id,
        initial_seed=seed
    )
    print("Batch run took  {:.2f} seconds to complete.".format(time.time() - start_time))

    results_df = pd.DataFrame(results)

    rng = np.random.default_rng(seed=156)
    random_colours = rng.random((len(ALL_MODEL_REPORTEERS), 3))
    all_reporter_colours = dict(zip(ALL_MODEL_REPORTEERS.keys(), random_colours))
    model_reporters = [key for key in results_df.keys() if key in ALL_MODEL_REPORTEERS.keys()] #todo figure out if I can exclude some model reporters from the sim e.g. stake by agent (or somehow deal with them in plots if not excluded)
    for variable_param in variable_params:
        useLogAxis = False  #True if variable_param == 'a0' else False #todo use logaxis for a0 or not?
        for model_reporter in model_reporters:
            hlp.plot_aggregate_data(results_df, variable_param, model_reporter,
                                    all_reporter_colours[model_reporter],
                                    batch_run_id, batch_run_directory, log_axis=useLogAxis)

    vp = list(variable_params.keys())
    if len(vp) == 2:
        # plot heatmap when we have two variable parameters
        hlp.plot_aggregate_data_heatmap(results_df, vp, model_reporters, batch_run_directory)