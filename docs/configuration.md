# Configuration

The simulation engine is highly configurable. From the reward scheme parameters to be used to the output files to be 
generated, there are numerous variables that can vary from execution to execution. This customisation is performed
using command-line arguments when running the main or batch-run script. We will go through all the available options 
here, but it's also possible to get an overview of the arguments and their default values by running the help command:

	python main.py --help
    python batch-run.py --help


| Argument | Description | Accepted values | Default value |
| -------- | ----------- | --------------- | ------------- |
| --n      | The number of stakeholders / agents<br>in the simulation. | Any natural number | 1000 |
| --k      | The target number of pools of the<br>system (reward sharing scheme parameter) | Any natural number < n | 100 |
| --a0     | Stake infuence / Sybil resilience<br>factor (reward sharing scheme parameter) | Any non-negative real number | 0.3 |