#!/bin/bash

# n = 1000
echo 'python3.9 main.py --execution_id=single-run-baseline-n-1000-k-10 --n=1000 --k=10 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-baseline-n-1000-k-10 --n=1000 --k=10 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 main.py --execution_id=single-run-parameter-change-k-10-20-n-1000 --n=1000 --k 10 20 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-parameter-change-k-10-20-n-1000 --n=1000 --k 10 20 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 main.py --execution_id=single-run-parameter-change-alpha-0.03-0.3-n-1000-k-10 --n=1000 --k=10 --alpha 0.03 0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-parameter-change-alpha-0.03-0.3-n-1000-k-10 --n=1000 --k=10 --alpha 0.03 0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 batch-run.py --execution_id=batch-run-alpha-log-n-1000-k-100 --n=1000 --k=100 --alpha -3 1 50  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 batch-run.py --execution_id=batch-run-alpha-log-n-1000-k-100 --n=1000 --k=100 --alpha -4 1 50  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 batch-run.py --execution_id=batch-run-k-5-500-n-1000 --n=1000 --k 5 501 5 --alpha=0.3  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 batch-run.py --execution_id=batch-run-k-5-500-n-1000 --n=1000 --k 5 501 5 --alpha=0.3  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 main.py --execution_id=single-run-abstention-30-n-1000-k-10 --n=1000 --k=10 --alpha=0.3 --abstention_rate=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-abstention-30-n-1000-k-10 --n=1000 --k=10 --alpha=0.3 --abstention_rate=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 batch-run.py --execution_id=batch-run-abstention-rate-0-90-n-1000-k-100 --n=1000 --k=100 --alpha=0.3 --abstention_rate 0 0.91 0.05  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 batch-run.py --execution_id=batch-run-abstention-rate-0-90-n-1000-k-100 --n=1000 --k=100 --alpha=0.3 --abstention_rate 0 0.91 0.05  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 batch-run.py --execution_id=batch-run-k-5-500-n-1000-flat-stk-dstr --n=1000 --k 5 501 5 --alpha=0.3  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 batch-run.py --execution_id=batch-run-k-5-500-n-1000-flat-stk-dstr --n=1000 --k 5 501 5 --alpha=0.3  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'optional'

echo 'python3.9 main.py --execution_id=single-run-baseline-n-1000-k-100 --n=1000 --k=100 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-baseline-n-1000-k-100 --n=1000 --k=100 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 main.py --execution_id=single-run-parameter-change-k-100-200-n-1000 --n=1000 --k 100 200 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-parameter-change-k-100-200-n-1000 --n=1000 --k 100 200 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 main.py --execution_id=single-run-parameter-change-alpha-0.03-0.3-n-1000-k-100 --n=1000 --k=100 --alpha 0.03 0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-parameter-change-alpha-0.03-0.3-n-1000-k-100 --n=1000 --k=100 --alpha 0.03 0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42

echo 'python3.9 main.py --execution_id=single-run-alpha-0-n-1000-k-10 --n=1000 --k=10 --alpha=0 --cost_min=1e-4 --cost_max=1e-3 common_cost=9e-5 --cost_factor=0.6'
python3.9 main.py --execution_id=single-run-alpha-0-n-1000-k-10 --n=1000 --k=10 --alpha=0 --cost_min=1e-4 --cost_max=1e-3 common_cost=9e-5 --cost_factor=0.6

echo 'python3.9 main.py --execution_id=single-run-alpha-0-n-1000-k-100 --n=1000 --k=100 --alpha=0 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42'
python3.9 main.py --execution_id=single-run-alpha-0-n-1000-k-100 --n=1000 --k=100 --alpha=0 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42



# n = 10 000

echo 'n = 10 000'
python3.9 main.py --execution_id=single-run-baseline-n-10000-k-10 --n=10000 --k=10 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-baseline-n-10000-k-100 --n=10000 --k=100 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-abstention-30-n-10000-k-10 --n=10000 --k=10 --alpha=0.3 --abstention_rate=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-parameter-change-k-10-20-n-10000 --n=10000 --k 10 20 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-parameter-change-alpha-0.03-0.3-n-10000-k-10 --n=10000 --k=10 --alpha 0.03 0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-parameter-change-k-100-200-n-10000 --n=10000 --k 100 200 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-parameter-change-alpha-0.03-0.3-n-10000-k-100 --n=10000 --k=100 --alpha 0.03 0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-alpha-0-n-10000-k-10 --n=10000 --k=10 --alpha=0 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 main.py --execution_id=single-run-alpha-0-n-10000-k-100 --n=10000 --k=100 --alpha=0 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 batch-run.py --execution_id=batch-run-k-1-50-n-10000 --n=10000 --k 1 51 2 --alpha=0.3 --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 batch-run.py --execution_id=batch-run-abstention-rate-0-90-n-10000-k-100 --n=10000 --k=100 --alpha=0.3 --abstention_rate 0 0.91 0.01  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 batch-run.py --execution_id=batch-run-k-1-250-n-10000 --n=10000 --k 1 251 3 --alpha=0.3  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42
python3.9 batch-run.py --execution_id=batch-run-alpha-log-n-10000-k-100 --n=10000 --k=100 --alpha -3 1 30  --cost_min=1e-4 --cost_max=1e-3 --cost_factor=0.7 --seed=42