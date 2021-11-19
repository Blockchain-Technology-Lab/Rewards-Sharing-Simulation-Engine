#!/bin/bash

python3.9 main.py --simulation_id=baseline-100-10 --n=100 --k=10 --alpha=0.3

python3.9 main.py --simulation_id=baseline-1000-100 --n=1000 --k=100 --alpha=0.3

python3.9 main.py --simulation_id=abstention-30-100-10 --n=100 --k=10 --alpha=0.3 --abstention_rate=0.3

python3.9 main.py --simulation_id=parameter-change-k-100 --n=100 --k 10 20 --alpha=0.3

python3.9 main.py --simulation_id=parameter-change-alpha-100-10 --n=100 --k=10 --alpha 0.03 0.3

python3.9 main.py --simulation_id=parameter-change-k-1000 --n=1000 --k 100 200 --alpha=0.3

python3.9 main.py --simulation_id=parameter-change-alpha-1000-100 --n=1000 --k=100 --alpha 0.03 0.3

python3.9 main.py --simulation_id=alpha-zero-100-10 --n=100 --k=10 --alpha=0

python3.9 main.py --simulation_id=alpha-zero-1000-100 --n=1000 --k=100 --alpha=0

python3.9 batch-run.py --execution_id=varying-k-100 --n=100 --k 1 51 1 --alpha=0.3

python3.9 batch-run.py --execution_id=varying-alpha-1000-100 --n=1000 --k=100 --alpha 0 10 0.25

python3.9 batch-run.py --execution_id=varying-abstention-rate-1000-100 --n=1000 --k=100 --alpha=0.3 --abstention_rate 0 0.91 0.01

python3.9 batch-run.py --execution_id=varying-k-1000 --n=1000 --k 1 251 2 --alpha=0.3




