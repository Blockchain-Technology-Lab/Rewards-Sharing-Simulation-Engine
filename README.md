# PoS-Pooling-Games-Simulation
Tool to simulate staking behaviour in Proof-of-Stake Blockchains.

Python 3.9+ required

Other dependencies: 
- mesa 
- matplotlib

---------------------

How to use:

There are 2 main options to execute the simulation with user-defined options.

The first option is to run the file "main.py" through the terminal. 
An example command to run from the directory of that file is:

python ./main.py --execution_id=200-players --n=200 

which would execute the simulation with 200 players instead of the default 100.
If no arguments are passed, then the simulation is run with all its default values.
To see all argument options and their default values, you can run the following command:

python ./main.py --help 

Running the simulation with the default settings is not expected to take too long, but 
different parametrizations may lead to an increased running time.

The output of the simulation is a csv file with the final configuration of the system, and it is saved in the "output" directory 
(will be created automatically the first time the simulation is run).

The other option is to use the "batch-run.py" file to run multiple instances of the 
simulation at once, using multiprocessing. An example of such a command would be: 

python3.9 batch-run.py --execution_id=batch-run-k-10-100 --n=1000 --k 10 101 10 --seed=42

which runs the simulation for 10 different values of k. The output of all the runs is again saved in a csv file.

Again, running: python ./batch-run.py --help will show all the different options and the default values of the parameters.


This project is licensed under the terms of the Apache 2.0 license.