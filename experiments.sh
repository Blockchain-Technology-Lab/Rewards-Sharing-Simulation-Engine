#!/bin/bash

# All experiments from the Examples page of the project's documentation are listed here
# If your Python 3.9 installation has an alias different from python3.9 you have to replace it in all commands

echo 'Run with 1000 agents, k = 100 and a0 = 0.3:'
python3.9 main.py --n=1000 --k=100 --a0=0.3 --execution_id=baseline

echo 'Run with two phases, first with k = 100 and then k = 200:'
python3.9 main.py --k 100 200 --execution_id=increasing-k

echo 'Batch run with 1000 agents and a range of 5 different values for k (100, 200, 300, 400, 500):'
python3.9 batch-run.py --n=1000 --k 100 501 100 --execution_id=batch-run-varying-k

echo 'Batch run with 1000 agents, k = 100 and a range of 5 values for a0 (0.001, 0.01, 0.1, 1, 10):'
python3.9 batch-run.py --n=1000 --k=100 --a0 -3 1 5 --execution_id=batch-run-varying-a0-log


echo 'Run with 5,000 agents, k = 500 and a specified seed (42):'
python3.9 main.py --n=5000 --k=500 --seed=42 --execution_id=n-5K-k-500-seed-42