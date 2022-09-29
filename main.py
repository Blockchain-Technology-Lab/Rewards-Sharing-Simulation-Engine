import logic.sim as simulation
import logic.helper as hlp

import pathlib
import argparse


def main():
    print("Let the Pooling Games begin!")

    parser = argparse.ArgumentParser(description='Pooling Games')
    hlp.add_script_arguments(parser)
    args = parser.parse_args()

    # todo deal with invalid inputs, e.g. negative n
    # todo make it possible to run more simulations w/o having to rerun the program (e.g. press any key to continue)

    sim = simulation.Simulation(
        n=args.n,
        k=args.k,
        a0=args.a0,
        stake_distr_source=args.stake_distr_source,
        agent_profile_distr=args.agent_profile_distr,
        inactive_stake_fraction=args.inactive_stake_fraction,
        inactive_stake_fraction_known= args.inactive_stake_fraction_known,
        relative_utility_threshold=args.relative_utility_threshold,
        absolute_utility_threshold=args.absolute_utility_threshold,
        seed=args.seed,
        pareto_param=args.pareto_param,
        max_iterations=args.max_iterations,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        extra_pool_cost_fraction=args.extra_pool_cost_fraction,
        agent_activation_order=args.agent_activation_order,
        #total stake
        iterations_after_convergence=args.iterations_after_convergence,
        reward_scheme = args.reward_scheme,
        execution_id=args.execution_id,
        #seq_id
        #parent_dir
        metrics=args.metrics,
        generate_graphs=args.generate_graphs,
        input_from_file=args.input_from_file
    )

    profiling = args.profile_code
    if profiling:
        import cProfile
        from pstats import Stats
        pr = cProfile.Profile()
        pr.enable()

    sim.run_model()

    if sim.has_converged():
        print('Reached equilibrium at step: ', sim.equilibrium_steps)
    else:
        print('Maximum iterations reached without convergence to an equilibrium.')

    if profiling:
        pr.disable()
        stats = Stats(pr)
        print("\n----- Profiling results -----")
        stats.sort_stats('tottime').print_stats(10)


if __name__ == "__main__":
    main()

