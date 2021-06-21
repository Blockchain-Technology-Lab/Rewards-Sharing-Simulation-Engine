# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:51:21 2021

@author: chris
"""

from sim import Simulation
from myModularVisualization import  MyModularServer
from mesa.visualization.modules import ChartModule
from stackedChartModule import StackedChartModule
from mesa.visualization.UserParam import UserSettableParameter

num_agents = 100
poolsChart = ChartModule([{"Label": "#Pools",
                      "Color": "Blue"}],
                    data_collector_name='datacollector')

#todo check why this chart misses one step
poolDynamicsStackedChart = StackedChartModule([{"Label": "Pool",
                      "Num_agents": num_agents}],  data_collector_name='datacollector')

model_params = {
    "n": UserSettableParameter(
        "slider", "Number of stakeholders", 10, 2, 100,
        description="The number of stakeholders / players in the system."
    ),
    "k": UserSettableParameter(
        "slider", "k", 3, 1, 100,
        description="The k value of the system."
    ),
    "alpha": UserSettableParameter(
        "slider", "α", 0.3, 0, 1, 0.01,
        description="The alpha value of the system."
    ),
    "pool_splitting": UserSettableParameter("checkbox", "Pool Splitting", False),

    "max_iterations": UserSettableParameter(
        "slider", "Max iterations", 100, 1, 300, 2,
        description="The maximum number of iterations of the system."
    )

    # user input options: TYPES = (NUMBER, CHECKBOX, CHOICE, SLIDER, STATIC_TEXT)

}

server = MyModularServer(Simulation,
                       [poolsChart, poolDynamicsStackedChart],
                       "PoS Pooling Games",
                       model_params)
server.port = 8521 # The default
server.launch()
