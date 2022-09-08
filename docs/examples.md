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

Run with 10,000 agents, k = 500 and a specified seed (42):

    python main.py --n=10000 --k=1000 --seed=42 --execution_id=n-10K-k-1000-seed-42


## Batch runs

Batch run with 1000 agents and a range of 5 different values for k (100, 200, 300, 400, 500): 

    python batch-run.py --n=1000 --k 100 501 100 --execution_id=batch-run-varying-k

Batch run with 1000 agents, k = 100 and a range of 5 values for a0 (0.001, 0.01, 0.1, 1, 10): 

    python batch-run.py --n=1000 --k=100 --a0 -3 1 5 --execution_id=batch-run-varying-a0-log

