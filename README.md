# PoS-Pooling-Games-Simulation
Tool to simulate staking behaviour in Proof-of-Stake Blockchains.

Python 3.9+ required

Other dependencies: 
- mesa 
- matplotlib

---------------------

How to use:

There are 2 main options to execute the simulation with user-defined options.

The preferred (faster) method is to run the file "main.py" through the terminal. 
An example command to run from the directory of that file is:

python ./main.py --simulation_id=200-players --n=200

which would execute the simulation with 200 players instead of the default 100.
If no arguments are passed, then the simulation is run with all its default values.
To see all argument options and their default values, you can run the following command:

python ./main.py --help 

Note that if the simulation_id parameter is not set, then the simulation will construct an identifier on its own based 
on the values of the other parameters. This identifier is typically quite long, therefore, depending on the directory 
from where the project is run, the full path may or may not exceed the character limit for file names. To be *sure* 
that such an issue is avoided, it is recommended to provide a simulation id for each run.

Running the simulation with the default settings is not expected to take too long (takes about 43'' on my laptop), but 
different parametrisations may lead to an increased running time.

The output of the simulation (plots, a csv file and in some cases a pickled file) is saved in the "output" directory 
(will be created automatically the first time the simulation is run), with the figures specifically in the "figures" 
subdirectory.

The other option is to execute the "runViz" file, which launches an interactive simulation in the browser.
One can accomplish that with the following command:

python -m ./interactiveViz.runViz

After the window is launched, the user can change any parameter values through sliders and buttons and then run
the simulation with them. Note that after changing any parameter values, the user needs to click on the "Reset" button
to save these options *and then* click on the "Start" button to execute the simulation. While the simulation is running,
the user can understand what's going on in every step through the dynamic charts that are in place.

Another (non-user-configurable for now) option is  to use the "batch-run.py" file to run multiple instances of the 
simulation at once, using multiprocessing. The current behaviour for now is that the simulation is run for different 
values of one parameter (k), while the rest are kept fixed (to their default values).
Note that, depending on the operating system, there is a line of code in that file that should be
commented / uncommented, because of the different requirements with multiprocessing (see that file for details).
