# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:51:21 2021

@author: chris
"""

from sim import Simulation
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from stackedChartModule import StackedChartModule

num_agents = 5
poolsChart = ChartModule([{"Label": "#Pools",
                      "Color": "Blue"}],
                    data_collector_name='datacollector')
poolDynamicsStackedChart = StackedChartModule([{"Label": "Pool",
                      "Datasets": num_agents}],  data_collector_name='datacollector')
server = ModularServer(Simulation,
                       [poolsChart, poolDynamicsStackedChart],
                       "PoS Pooling Games",
                       {"n":num_agents})
server.port = 8521 # The default
server.launch()
