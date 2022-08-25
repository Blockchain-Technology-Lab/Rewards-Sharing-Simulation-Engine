# Setup

## Dependencies
The simulation engine is written in Python 3.9, so the first thing to ensure if you want to run it on your machine is 
that you have Python 3.9 installed. 

Having the right version of Python, you can now install the requirements for the project, which can be found in the 
[requirements file](requirements.txt). To install all of them in one go, you can run the following from the root 
directory of the project (assuming that the ```python``` command corresponds to a Python 3.9 installation):
    
    python -m pip install -r requirements.txt

## Installation
Installing the simulation engine is very simple, as it only involves cloning the relevant github project:

    git clone https://github.com/Blockchain-Technology-Lab/Cardano-Pooling-Simulator.git

## Execution
To run the simulation, navigate to the directory of the project and run the main script:
    
    cd Cardano-Pooling-Simulator
    python main.py

It is also possible to run the simulation with different parameters (e.g. number of agents). For a full list of options
refer to the [Configuration](configuration.md) page, and for examples of using the simulation engine in different ways 
see the [Examples](examples.md) page.
