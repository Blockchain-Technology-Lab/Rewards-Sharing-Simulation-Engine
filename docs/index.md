# Cardano Pooling Simulator’s Documentation

This is the documentation for the Cardano Pooling Simulation Engine built by the University of Edinburgh's Blockchain 
Technology Lab. 
The source code is available on [Github](https://github.com/Blockchain-Technology-Laboratory/Cardano-Pooling-Simulator).

## Overview

The simulation models the behaviour of stakeholders in a Proof-of-Stake system, i.e. the way they use their stake to 
engage with the protocol depending on the rewards they expect to receive. It focuses particularly on the way different 
stakeholders combine their resources and create stake pools (it assumes an on-chain pooling mechanism like the one in 
Cardano) and follows the dynamics of the system until it (potentially) reaches an equilibrium. The implementation is 
based on the Cardano blockchain, but can be generalized to systems that use similar reward sharing schemes (e.g. 
[Nym](https://nymtech.net/)).

The simulation engine can be used to play out different scenarios and better understand the relationship between the
system's input (e.g. parameters of the reward scheme or initial stake distribution) and its convergence properties (does
it converge to an equilibrium and if yes how quickly, how decentralized is the final allocation of stake to the 
different stakeholders, and so on).

For details on how to install the engine and run simulations, see the [Setup](setup.md) page; for a complete guide on
how to customize the simulation, see the [Configuration](configuration.md) page; for a description of the different 
output files that a simulation produces, see the [Output](output.md) page, and for examples to get started with and 
draw inspiration from, see the [Examples](examples.md) page.


## Contributions
This is an open source project licensed  under the terms of the Apache 2.0 [license](LICENSE). Everyone is welcome to 
contribute to it by proposing or implementing their ideas. Example contributions include, but are not limited to, 
adding a new feature to the simulation engine (e.g. integration of exchange rates), improving the performance of the 
simulations, or creating a user-friendly interface for configuring and running simulations. 

