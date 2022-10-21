# Rewards Sharing Simulation Engine
Tool to simulate staking behaviour in Proof-of-Stake blockchains, especially tailored towards Cardano.

Written in Python 3.9 by Christina Ovezik, Research Engineer at the University of Edinburgh's [Blockchain Technology
Laboratory](https://www.ed.ac.uk/informatics/blockchain).

Refer to the project's [documentation](https://blockchain-technology-lab.github.io/Rewards-Sharing-Simulation-Engine/) for 
detailed instructions and examples on how to interact with the simulation engine. 

## Installation
To install the simulation engine, simply clone this project:

    git clone https://github.com/Blockchain-Technology-Lab/Rewards-Sharing-Simulation-Engine.git

Take note of the [requirements file](requirements.txt), which lists all the dependencies of the project, and make
sure you have all of them installed before running the simulation. To install all of them in one go, you can run the 
following command from the root directory of the project (assuming that the ```python``` command corresponds to a Python 
3.9 installation):

    python -m pip install -r requirements.txt

### Ubuntu 20.04 specific Installation

The default Ubuntu 20.04 installation does not meet Python version 3.9 requirement of the engine.
However, Python v.3.9 was backported to Ubuntu 20.04, which allows the engine run on 20.04.

Follow the steps below to use the engine on Ubuntu 20.04.

``` bash
# 1. Install Python v3.9 from the official repo
$ sudo apt update
$ sudo apt install python3.9

# 2. Install the non-backporteed pip3.9 from PyPA
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3.9 get-pip.py --user
$ pip3 --version
pip 22.2.2 from /home/ilap/.local/lib/python3.9/site-packages/pip (python 3.9)

# 2.1 If it shows:
# pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)
# then use pip3.9 in the cli/terminal instead.

# 3. Install simulations engine's pre-requisities
$ pip3.9 install -U Pillow --user
$ python3.9 -m pip install -r requirements.txt
```
Follow the steps below for using/running the engine on Ubuntu 20.04.

## Using the simulation engine

The simulation engine is a CLI tool, i.e. it can be executed through a terminal. In the future, a user-friendly 
interface may be added, but for now the best way to interact with the simulation is by running a python script that 
invokes it.

The first option is to run the ```main.py``` script, which also accepts user-defined options. 
An example command to run from the project's root directory is:

    python main.py --n=2000 --execution_id=2000-agents 

which executes the simulation with 2000 agents instead of the default 1000.
If no command-line arguments are passed, then the simulation is run with all its default values.
To see all argument options and their default values, one can run the following command:

    python main.py --help 

Running the simulation with the default settings is not expected to take too long, but different configurations (e.g. 
higher number of agents or higher target number of pools) may lead to an increased running time.

The output of a simulation execution is a folder within the "output" directory (created automatically the first time 
the simulation is run) that contains multiple files that describe the initial and final state of the system, and 
optionally files that track metrics on each round, and more.

The other option is to use the ```batch-run.py``` script to run multiple instances of the simulation at once, using 
multiprocessing. An example of such a command is: 

    python batch-run.py --n=1000 --k 100 200 300 --a0=0.3 --execution_id=batch-run-varying-k

which runs the simulation for 3 different values of k (100, 200, 300). The output of all the runs is saved in 
a relevant folder, with subfolders for each execution.

Again, running: 
    
    python batch-run.py --help 

will show all the different options and the corresponding default values for batch running simulations.

## Contributing
Everyone is welcome to contribute to our Rewards Sharing Simulation Engine! 

For changes in the code, please fork the repo first, commit your changes and then issue a pull request with your commits. 
For reporting a bug or other issue, please head over to our 
[Issues](https://github.com/Blockchain-Technology-Lab/Rewards-Sharing-Simulation-Engine/issues) page, and also feel free 
to engage in our [Disccussions](https://github.com/Blockchain-Technology-Lab/Rewards-Sharing-Simulation-Engine/discussions) 
by starting a new thread or responding to existing ones.

## License
This project is licensed under the terms and conditions of the Apache 2.0 [license](LICENSE). Contributions are welcome 
and will be covered by the same license. 
