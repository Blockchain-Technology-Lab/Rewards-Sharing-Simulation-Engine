# Cardano Pooling Simulator
Tool to simulate staking behaviour in Proof-of-Stake blockchains, especially tailored towards Cardano.

Written in Python 3.9 by Christina Ovezik, Research Engineer at the University of Edinburgh's [Blockchain Technology
Laboratory](https://www.ed.ac.uk/informatics/blockchain).

Refer to the project's [documentation](https://blockchain-technology-lab.github.io/Cardano-Pooling-Simulator/) for 
detailed instructions and examples on how to interact with the simulation engine. 

## Installation
To install the simulation engine, simply clone this project:

    git clone https://github.com/Blockchain-Technology-Lab/Cardano-Pooling-Simulator.git

Take note of the [requirements file](requirements.txt), which lists all the dependencies of the project, and make
sure you have all of them installed before running the simulation. To install all of them in one go, you can run the 
following command from the root directory of the project (assuming that the ```python``` command corresponds to a Python 
3.9 installation):

    python -m pip install -r requirements.txt

## Using the simulation engine

There are 2 main options to execute the simulation with user-defined options.

The first option is to run the ```main.py``` script through the terminal. 
An example command to run from the project's root directory is:

    python main.py --n=2000 --execution_id=2000-agents 

which executes the simulation with 2000 agents instead of the default 1000.
If no command-line arguments are passed, then the simulation is run with all its default values.
To see all argument options and their default values, you can run the following command:

    python main.py --help 

Running the simulation with the default settings is not expected to take too long, but different configurations (e.g. 
higher number of agents or higher target number of pools) may lead to an increased running time.

The output of a simulation execution is a folder within the "output" directory (created automatically the first time 
the simulation is run) that contains multiple files that describe the initial and final state of the system, and 
optionally files that track metrics on each round, and more.

The other option is to use the ```batch-run.py``` script to run multiple instances of the simulation at once, using 
multiprocessing. An example of such a command is: 

    python batch-run.py --n=1000 --k 100 501 100 --alpha=0.3 --execution_id=batch-run-varying-k

which runs the simulation for 5 different values of k (100, 200, 300, 400, 500). The output of all the runs is saved in 
a relevant folder, with subfolders for each execution.

Again, running: 
    
    python batch-run.py --help 

will show all the different options and the corresponding default values for batch running simulations.

## License
This project is licensed under the terms and conditions of the Apache 2.0 [license](LICENSE). Contributions are welcome 
and will be covered by the same license.
