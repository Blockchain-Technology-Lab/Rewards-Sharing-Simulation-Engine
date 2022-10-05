# Configuration

The simulation engine is highly configurable. From the reward scheme parameters to be used to the output files to be 
generated, there are numerous variables that can vary from execution to execution. This customization is performed
using command-line arguments when running the ```main.py``` or ```batch-run.py``` scripts. We will go through all the 
available options here, but it's also possible to get an overview of the arguments and their default values by running 
the corresponding help commands:

	python main.py --help
    python batch-run.py --help

Remember though that all arguments are optional, so it is not mandatory to manually set values for any of them. 
If not value is explicitly provided, then the corresponding default value is used.

## Command-line options

These are all the arguments that can be configured during execution from the command line: 

---
**--n**: The number of stakeholders / agents in the simulation. The default value is 1000, but any natural number 
is accepted. Note that the higher the value of **n** the slower the simulation becomes.
---
**--k**: The target number of pools of the system (reward sharing scheme parameter). The default value is 100, but 
any natural number is accepted. Note that the higher the value of **k** the slower the simulation becomes.
---
**--a0**: Stake influence factor (reward sharing scheme parameter). The default value is 0.3, but any non-negative 
real number is accepted. 
---
**--reward_scheme**: The function that is used to calculate the rewards of a pool. There are currently five options,
which correspond to the original reward function of Cardano (0), a simplified version of the original function (1), a 
version that uses flat pledge benefit (2), one that uses curve pledge benefit (3) which corresponds to the one 
proposed in [CIP-7](https://cips.cardano.org/cips/cip7/) and one that corresponds to 
[CIP-50](https://github.com/michael-liesenfelt/CIPs/blob/CIP-Liesenfelt-Shelleys_Voltaire_decentralization_update/CIP-Liesenfelt-Shelleys_Voltaire_decentralization_update/README.md)
(4). More reward schemes can be easily added by extending the relevant class. Please note, however, that our methodology 
includes heuristics that were designed based on the current reward scheme of Cardano, so it might need some adjustments 
to yield meaningful results when used with different reward schemes.
---
**--agent_profile_distr**: List of weights that determine how agents are distributed across the different behavioral 
profiles. Currently, there are three different profiles: non-myopic stakeholder, myopic stakeholder and abstainer. When 
using this argument, one must provide a number of values equal to the number of profiles. The default is [1, 0, 0], i.e. 
there are only non-myopic agents. Any non-negative real numbers can be used for the weights (they don't have to 
explicitly sum to 1), e.g. [4, 2, 1] dictates that there will be (roughly) twice as many non-myopic agents compared to 
myopic ones and that the myopic ones will be (roughly) twice as many as the abstainers. 
---
**--cost_min**: The lowest initial cost that a stakeholder can have (acts as the lower bound of the Uniform distribution 
that cost values are drawn from). The default value is 10<sup>-5</sup>, but any non-negative real number is accepted. 
---
**--cost_max**: The highest initial cost that a stakeholder can have (acts as the upper bound of the Uniform 
distribution that cost values are drawn from). The default value is 10<sup>-4</sup>, but any non-negative real number > 
**cost_min** is accepted. 
---
**extra_pool_cost_fraction**: When an agent operates one pool, then the cost of the pool is equal to the cost of the 
agent. However, in our simulation it's possible for agents to operate multiple pools, so we assume that each additional 
pool an agent operates costs a fraction of their initial cost, and that fraction is the same for all agents and dictated 
by this argument. The default value is 0.4 (i.e. that a second pool costs 0.4 times as much as the first), but any 
non-negative real number is accepted (if we assume economies of scale then this value must be < 1, but if we assume some 
sort of Sybil cost then it can also be >= 1).
---
**--agent_activation_order**: The order in which agents get activated. The default option is "Random", meaning that the 
order changes for each round, while the other options are "Sequential" for activating agents in the same order every 
time (based on their ids) and "Semisimultaneous" for activating a number of them simultaneously before moving on.
---
**--absolute_utility_threshold**: The absolute utility threshold for accepting new moves (relates to inertia). If an 
agent develops a new strategy whose utility does not exceed that of its current one by at least this threshold, then the 
new strategy is rejected. The default value is 10<sup>-9</sup>, but any non-negative real number is accepted.
---
**--relative_utility_threshold**: The relative utility threshold for accepting new moves (relates to inertia). If an 
agent develops a new strategy whose utility does not exceed that of its current one by at least this fraction, then the 
new strategy is rejected. The default value is 0, i.e. no relative utility threshold exists, but any non-negative real 
number is accepted. For example, if this threshold is 0.1 then it means that a new move has to yield utility at least 
10% higher than that of the current move in order to be selected.
---
**--stake_distr_source**: The distribution to use for the initial allocation of stake to the agents. The default choice 
is "Pareto", but other options include "Flat" for a distribution where all agents start with equal stake and "File" 
where a custom distribution is read from a csv file. In the latter case, the relevant file is expected to be at the root 
directory of the project, contain only stake values separated by commas and be named 
synthetic-stake-distribution-X-agents.csv where X is the number of agents used.
---
**--pareto_param**: The parameter that determines the shape of the Pareto distribution that the stake is sampled from
(only relevant if stake_distr_source is set to "pareto"). The default value is 2 but any positive real number is 
accepted.
---
**--inactive_stake_fraction**: The fraction of the total stake of the system that remains inactive (is not allocated to
any of the agents). The default value is 0, meaning that the active stake of the system is equal to the total stake, but 
any value between 0 and 1 is accepted.
---
**--inactive_stake_fraction_known**: Flag that determines whether the fraction of the system's stake that is inactive is 
known upon the launch of the simulation (only relevant when there is inactive stake). If this fraction is known, then 
the simulation automatically adjusts the target number of pools k. The default setting is for it to remain unknown.
---
**--iterations_after_convergence**: The minimum consecutive idle iterations that are required before the simulation 
declares convergence to an equilibrium and terminates. The default value is 10, but any natural number is accepted.
---
**--max_iterations**: The maximum number of iterations that the simulation will perform before terminating. The default 
is 2000, but any natural number is accepted (it is recommended to keep this number high, in order to give the 
opportunity for simulations to converge to equilibria).
---
**--metrics**: A list of ids that correspond to metrics that are tracked during the simulation. Default is [1, 2, 3, 4, 
6, 17, 18, 24, 25]. For the full list of metrics and their corresponding identifiers, see the [Metrics](metrics.md) page.
---
**--generate_graphs**: A flag that determines whether graphs relating to the tracked metrics are generated upon 
termination of the simulation. By default, this is activated.
---
**--seed**: The seed to be used by the pseudorandom generator - can be specified to allow for reproducibility of the
results. The default value is 'None', which means that a seed is chosen at random (can be accessed through the output 
of the simulation). Any non-negative integer can be accepted as the seed.
---
**--execution_id**: An optional identifier for the specific execution of the simulation run, which is used for naming 
the output folder / files. If no identifier is provided, then one is generated automatically.
---
**--input_from_file**: A flag that determines whether the input is read from a file (must be named "args.json" and be 
placed in the root directory of the project). If this is activated, then any other command line arguments are discarded.
By default, this flag is not activated. 
---
