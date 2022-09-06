# Configuration

The simulation engine is highly configurable. From the reward scheme parameters to be used to the output files to be 
generated, there are numerous variables that can vary from execution to execution. This customisation is performed
using command-line arguments when running the ```main.py``` or ```batch-run.py``` scripts. We will go through all the 
available options here, but it's also possible to get an overview of the arguments and their default values by running 
the corresponding help commands:

	python main.py --help
    python batch-run.py --help

## Command-line options

These are all the arguments that can be configured during execution from the command line. 

**--n**: The number of stakeholders / agents in the simulation. The default value is **1000**, but any natural number 
is accepted. Note that the higher the value of **n** the slower the simulation becomes.
---
**--k**: The target number of pools of the system (reward sharing scheme parameter). The default value is **100**, but 
any natural number < **n** is accepted. Note that the higher the value of **k** the slower the simulation becomes.
---
**--a0**: Stake influence factor (reward sharing scheme parameter). The default value is **0.3**, but any non-negative 
real number is accepted. 
---



