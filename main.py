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
    parser.add_argument('--L', nargs="+", type=int, default=100,
                        help='The L value of the system. Default is 100')
    parser.add_argument('--cost_min', type=float, default=1e-5,
                        help='The minimum possible cost for operating a stake pool. Default is 1e-4.')
    parser.add_argument('--cost_max', type=float, default=1e-4,
                        help='The maximum possible cost for operating a stake pool. Default is 1e-3.')
    parser.add_argument('--cost_factor', nargs="+", type=float, default=0.4,
                        help='The factor that determines how much an additional pool costs. '
                             'Default is 40%%.')
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
                        help='The number of steps for which a agent remains idle after opening a pool. Default is 0.')
    parser.add_argument('--myopic_fraction', nargs="+", type=float, default=0,
                        help='The fraction of myopic agents in the simulation. Default is 0%%.')
    parser.add_argument('--abstention_rate', type=float, default=0,
                        help='The fraction of the total stake that remains inactive. Default is 0.')
    parser.add_argument('--pool_splitting', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='Are individual agents allowed to create multiple pools? Default is yes.')
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='The maximum number of iterations of the system. Default is 2000.')
    parser.add_argument('--steps_for_convergence', type=int, default=10,
                        help='The minimum consecutive idle steps that are required to declare convergence. '
                             'Default is 10. But if min_steps_to_keep_pool > ms then ms = min_steps_to_keep_pool + 1.')
    parser.add_argument('--stake_distr_source', type=str, default='pareto',
                        help='The distribution type to use for the initial allocation of stake to the agents.')
    parser.add_argument('--extra_cost_type', type=str, default='fixed_fraction',
                        help='The method used to calculate the cost of any additional pool.')
    parser.add_argument('--execution_id', type=str, default='',
                        help='An optional identifier for the specific simulation run, '
                             'which will be included in the output.')
    parser.add_argument('--input_from_file', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='If True then the input is read from a file (args.json) and any other command line '
                             'arguments are discarded. Default is False.')
    parser.add_argument('--metrics', nargs="+", type=int, default=None,
                        help='The list of ids that correspond to metrics that are tracked during the simulation. Default is [1,2,3]')
    parser.add_argument('--generate_graphs', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='If True then the graphs are generated upon completion. Default is True.')
    parser.add_argument('--abstention_known', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Is the abstention rate of the system known beforehand? Default is no.')
    parser.add_argument('--pool_opening_process', type=str, default='plus-one',
                        help='The heuristic to use for determining a pool strategy. Options: local-search (default), plus-one.')

    args = parser.parse_args()

    # todo deal with invalid inputs, e.g. negative n
    # todo make it possible to run more simulations w/o having to rerun the program (e.g. press any key to continue)

    sim = simulation.Simulation(
        n=args.n,
        k=args.k,
        L=args.L,
        stake_distr_source=args.stake_distr_source,
        myopic_fraction=args.myopic_fraction,
        abstention_rate=args.abstention_rate,
        abstention_known = args.abstention_known,
        relative_utility_threshold=args.relative_utility_threshold,
        absolute_utility_threshold=args.absolute_utility_threshold,
        min_steps_to_keep_pool=args.min_steps_to_keep_pool,
        pool_splitting=args.pool_splitting,
        seed=args.seed,
        pareto_param=args.pareto_param,
        max_iterations=args.max_iterations,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        cost_factor=args.cost_factor,
        agent_activation_order=args.agent_activation_order.capitalize(),
        #total stake
        steps_for_convergence=args.steps_for_convergence,
        extra_cost_type=args.extra_cost_type,
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

    equilibrium_steps = sim.equilibrium_steps
    pivot_steps = sim.pivot_steps
    print('equilibrium steps: ', equilibrium_steps)
    print('pivot steps: ', pivot_steps)


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
