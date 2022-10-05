# Examples
Here, we provide some examples that can help better understand the capacities of the simulation engine. 
We divide the examples in sections, depending on the type of execution (single run or batch run). Note that in all 
examples below, we assume that the ```python``` command corresponds to a Python 3.9 installation and that the commands 
are executed from the root directory of the project. Recall that when an argument is not set explicitly then its default 
value is used (for all arguments and their default values see the [Configuration](configuration.md)) page).

## Single runs

Run with 1000 agents, k = 100 and a0 = 0.3: 
    
    python main.py --n=1000 --k=100 --a0=0.3 --execution_id=baseline

Run with two phases, first with k = 100 and then k = 200: 

    python main.py --k 100 200 --execution_id=increasing-k

Run with 3,000 agents, k = 500 and a specified seed (42):

    python main.py --n=3000 --k=500 --seed=42 --execution_id=n-3K-k-500-seed-42

Run with 50% of the agents being myopic:

    python main.py --agent_profile_distr 1 1 0


## Batch runs

Batch run with 1000 agents and 5 different values for k (100, 200, 300, 400, 500): 

    python batch-run.py --n=1000 --k 100 200 300 400 500 --execution_id=batch-run-varying-k

Batch run with 1000 agents, k = 100 and 3 different values for a0 (0.01, 0.1, 1): 

    python batch-run.py --n=1000 --k=100 --a0 0.01 0.1 1 --execution_id=batch-run-varying-a0

Batch run with two variables, using 3 values for k and 3 values for a0 (total of 9 combinations):

    python batch-run.py --n=500 --k 50 100 150 --a0 0.05 0.1 0.3 --execution_id=batch-run-varying-k-a0-3x3
