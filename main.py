import logic.sim as simulation

import pathlib
import argparse


def main():
    print("Let the Pooling Games begin!")

    parser = argparse.ArgumentParser(description='Pooling Games')
    parser.add_argument('--n', type=int, default=1000,
                        help='The number of agents (natural number). Default is 1000.')
    parser.add_argument('--k', nargs="+", type=int, default=100,
                        help='The k value of the system (natural number). Default is 100.')
    parser.add_argument('--a0', nargs="+", type=float, default=0.3,
                        help='The a0 value of the system (decimal number between 0 and 1). Default is 0.3')
    parser.add_argument('--cost_min', type=float, default=1e-5,
                        help='The minimum possible cost for operating a stake pool. Default is 1e-4.')
    parser.add_argument('--cost_max', type=float, default=1e-4,
                        help='The maximum possible cost for operating a stake pool. Default is 1e-3.')
    parser.add_argument('--extra_pool_cost_fraction', nargs="+", type=float, default=0.4,
                        help='The factor that determines how much an additional pool costs as a fraction of '
                             'the original cost value of the stakeholder. Default is 40%%.')
    parser.add_argument('--pareto_param', type=float, default=2.0,
                        help='The parameter that determines the shape of the distribution that the stake will be '
                             'sampled from. Default is 2.')
    parser.add_argument('--relative_utility_threshold', nargs="+", type=float, default=0,
                        help='The utility increase ratio under which moves are disregarded. Default is 0%%.')
    parser.add_argument('--absolute_utility_threshold', nargs="+", type=float, default=1e-9,
                        help='The utility threshold under which moves are disregarded. Default is 1e-9.')
    parser.add_argument('--agent_activation_order', type=str, default='Random',
                        help='agent activation order. Default is random.')
    parser.add_argument('--seed', default=None,
                        help='Seed for reproducibility. Default is None, which means that no seed is given.')
    parser.add_argument("--min_steps_to_keep_pool", type=int, default=0,
                        help='The number of steps for which an agent remains idle after opening a pool. Default is 0.')
    parser.add_argument('--profile_distr', nargs="+", type=float, default=[1, 0, 0],
                        help='The probability distribution for assigning different profiles to the agents. Default is [1, 0, 0], i.e. 100%% non-myopic agents.')
    parser.add_argument('--inactive_stake_fraction', type=float, default=0,
                        help='The fraction of the total stake that remains inactive (does not belong to any of the agents). Default is 0.')
    parser.add_argument('--inactive_stake_fraction_known', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Is the inactive stake fraction of the system known beforehand? Default is no.')
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='The maximum number of iterations of the system. Default is 2000.')
    parser.add_argument('--iterations_after_convergence', type=int, default=10,
                        help='The minimum consecutive idle iterations that are required before terminations. '
                             'Default is 10. But if min_steps_to_keep_pool > ms then ms = min_steps_to_keep_pool + 1.')
    parser.add_argument('--stake_distr_source', type=str, default='pareto',
                        help='The distribution type to use for the initial allocation of stake to the agents.')
    parser.add_argument('--execution_id', type=str, default='',
                        help='An optional identifier for the specific simulation run, '
                             'which will be included in the output.')
    parser.add_argument('--reward_function_option', type=int, default=0,
                        help='The reward function to use in the simulation. 0 for the old function, 1 for the new one, '
                             '2 for alternative-1 and 3 for alternative-2.')
    parser.add_argument('--input_from_file', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='If True then the input is read from a file (args.json) and any other command line '
                             'arguments are discarded. Default is False.')
    parser.add_argument('--metrics', nargs="+", type=int, default=None,
                        help='The list of ids that correspond to metrics that are tracked during the simulation. Default is [1,2,3]')
    parser.add_argument('--generate_graphs', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='If True then the graphs are generated upon completion. Default is True.')
    parser.add_argument('--pool_opening_process', type=str, default='local-search',
                        help='The heuristic to use for determining a pool strategy. Options: local-search (default), plus-one.')

    args = parser.parse_args()

    # todo deal with invalid inputs, e.g. negative n
    # todo make it possible to run more simulations w/o having to rerun the program (e.g. press any key to continue)

    sim = simulation.Simulation(
        n=args.n,
        k=args.k,
        a0=args.a0,
        stake_distr_source=args.stake_distr_source,
        profile_distr=args.profile_distr,
        inactive_stake_fraction=args.inactive_stake_fraction,
        inactive_stake_fraction_known= args.inactive_stake_fraction_known,
        relative_utility_threshold=args.relative_utility_threshold,
        absolute_utility_threshold=args.absolute_utility_threshold,
        min_steps_to_keep_pool=args.min_steps_to_keep_pool,
        seed=args.seed,
        pareto_param=args.pareto_param,
        max_iterations=args.max_iterations,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        extra_pool_cost_fraction=args.extra_pool_cost_fraction,
        agent_activation_order=args.agent_activation_order.capitalize(),
        #total stake
        iterations_after_convergence=args.iterations_after_convergence,
        reward_function_option = args.reward_function_option,
        execution_id=args.execution_id,
        #seq_id
        #parent_dir
        metrics=args.metrics,
        generate_graphs=args.generate_graphs,
        input_from_file=args.input_from_file,
        pool_opening_process=args.pool_opening_process
    )

    sim.run_model()

    sim_df = sim.datacollector.get_model_vars_dataframe()
    execution_id = sim.execution_id

    execution_dir = sim.directory
    figures_dir = execution_dir / "figures"
    pathlib.Path(figures_dir).mkdir(parents=True, exist_ok=True)

    if sim.has_converged():
        print('Reached equilibrium at step: ', sim.equilibrium_steps)
    else:
        print('Maximum iterations reached without convergence to an equilibrium.')


def main_with_profiling():
    import cProfile
    from pstats import Stats

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(10)


if __name__ == "__main__":
    main()  # for profiling the code, comment this line and uncomment the one below
    '''main_with_profiling()
    cache_funcs = [calculate_potential_profit, calculate_pool_reward, #calculate_cost_per_pool, calculate_myopic_pool_desirability,
                   calculate_delegator_reward_from_pool, calculate_operator_reward_from_pool, calculate_cost_per_pool_fixed_fraction,
                   calculate_pool_desirability, calculate_pool_stake_NM_from_rank, determine_pledge_per_pool]
    for func in cache_funcs:
        print(func.__name__,': ', func.cache_info())'''
