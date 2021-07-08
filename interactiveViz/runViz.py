# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:51:21 2021

@author: chris
"""
import mesa.visualization.ModularVisualization
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer

from logic.sim import Simulation

# from myModularVisualization import MyModularServer
from interactiveViz.stackedChartModule import StackedChartModule
from interactiveViz.scatterChartModule import ScatterChartModule
from interactiveViz.myChartModule import MyChartModule

max_num_agents = 500
poolsChart = MyChartModule([{"Label": "#Pools",
                             "Color": "Blue"}],
                           data_collector_name='datacollector')

poolDynamicsStackedChart = StackedChartModule([{"Label": "Pool",
                                                "Num_agents": max_num_agents}],
                                              data_collector_name='datacollector')

poolScatterChart = ScatterChartModule([{"Label": "StakePairs"}],
                                      data_collector_name='datacollector')

model_params = {
    "n": UserSettableParameter(
        "slider", "Number of stakeholders", 100, 2, max_num_agents,
        description="The number of stakeholders in the system."
    ),
    "k": UserSettableParameter(
        "slider", "k", 10, 1, 100,
        description="The k value of the system."
    ),
    "alpha": UserSettableParameter(
        "slider", "α", 0.3, 0, 1, 0.01,
        description="The alpha value of the system."
    ),

    "max_iterations": UserSettableParameter(
        "slider", "Max iterations", 100, 1, 300, 2,
        description="The maximum number of iterations of the system."
    ),

    "cost_min": UserSettableParameter(
        "slider", "Minimum cost", 0.001, 0.001, 0.05, 0.001,
        description="The minimum possible cost for operating a stake pool."
    ),

    "cost_max": UserSettableParameter(
        "slider", "Maximum cost", 0.002, 0.002, 0.1, 0.001,
        description="The maximum possible cost for operating a stake pool."
    ),

    "pareto_param": UserSettableParameter(
        "slider", "Pareto shape value", 2, 0.1, 5, 0.1,
        description="The parameter that determines the shape of the distribution that the stake will be sampled from"
    ),

    "player_activation_order": UserSettableParameter("choice", "Player activation order", value="Random",
                                              choices=list(Simulation.player_activation_orders.keys())),

    "seed": UserSettableParameter(
        "number", "Random seed", 42, description="Seed for reproducibility"
    ),

    "idle_steps_after_pool": UserSettableParameter(
        "slider", "Idle steps", 10, 1, 20, 1
    )
}

# figure out why MyModularServer was not working at some point
# figured out: it only works when I use the ModularServer first so it probably caches some necessary files
server = ModularServer(Simulation,
                       [poolsChart, poolDynamicsStackedChart, poolScatterChart],
                       "PoS Pooling Games",
                       model_params)
# todo investigate why there are missing steps from the charts
server.port = 8521
server.launch()
