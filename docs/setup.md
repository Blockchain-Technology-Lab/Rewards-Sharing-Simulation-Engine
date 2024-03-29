# Setup

## Dependencies
The simulation engine is written in Python 3.9, so the first thing to ensure when running it on some machine is 
that [Python 3.9](https://www.python.org/downloads/release/python-390/) is installed there. The remaining dependencies 
of the project can be found in the [requirements file](requirements.txt). Running the following from the root directory 
of the project installs all required packages in one go (assuming that the ```python``` command corresponds to a 
Python 3.9 installation):
    
    python -m pip install -r requirements.txt

## Installation
Installing the simulation engine is very simple, as it only involves cloning the relevant 
[Github project](https://github.com/Blockchain-Technology-Laboratory/Rewards-Sharing-Simulation-Engine):

    git clone https://github.com/Blockchain-Technology-Lab/Rewards-Sharing-Simulation-Engine.git

## Execution
The simulation engine is a CLI tool, i.e. it can be executed through a terminal. In the future, a user-friendly 
interface may be added, but for now the best way to interact with the simulation is by running a python script that 
invokes it.

To run the simulation, navigate to the directory of the project and run the ```main.py``` script from a terminal:

    python main.py

This executes the simulation with the default options (1000 agents, k = 100, a0 = 0.3, and so on). It is also possible 
to run the simulation with different parameters. For example, to if we want a simulation with 10,000 agents and target 
number of pools k = 500, we can run the following:

    python main.py --n=10000 --k=500

For the full list of options that the simulation accepts, refer to the [Configuration](configuration.md) page, and for 
examples of using the simulation engine in different ways see the [Examples](examples.md) page.
