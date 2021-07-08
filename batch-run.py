from multiprocessing import freeze_support

from mesa.batchrunner import BatchRunner
from mesa.batchrunner import BatchRunnerMP
import time
import matplotlib.pyplot as plt

from logic.sim import Simulation
from logic.sim import get_number_of_pools

if __name__ == '__main__':
    freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux)

    fixed_params = {"n": 100,
                    "k": 10,
                    "total_stake": 1,
                    "max_iterations": 50,
                    "cost_min": 0.001,
                    "player_activation_order": "Random"}

    variable_params = {"alpha": [0.1, 0.3, 0.5],
                       "cost_max": [0.002, 0.02, 0.1],
                       "pareto_param": [1, 1.5, 2]}
    # todo figure out how to run the model with only certain combinations of the variable params

    # Run the models as a single process (takes almost 2 mins on my laptop with current settings)
    batch_run = BatchRunner(Simulation,
                            variable_params,
                            fixed_params,
                            iterations=1,
                            max_steps=50,
                            model_reporters={"#Pools": get_number_of_pools},
                            display_progress=True)
    start_time = time.time()
    batch_run.run_all()
    print("Batch run without multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    run_data = batch_run.get_model_vars_dataframe()
    print(run_data.head())
    plt.figure()
    plt.scatter(run_data['alpha'], run_data['#Pools'])
    plt.xlabel("α")
    plt.ylabel("#pools")
    plt.show()

    # Run the models using multiprocessing (takes less than 30'')
    batch_run_MP = BatchRunnerMP(Simulation,
                                 nr_processes=12,
                                 variable_parameters=variable_params,
                                 fixed_parameters=fixed_params,
                                 iterations=1,
                                 max_steps=100,
                                 model_reporters={"#Pools": get_number_of_pools})
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    # Extract data from the batch runner
    run_data_MP = batch_run_MP.get_model_vars_dataframe()
    print(run_data_MP.head())
    plt.figure()
    plt.scatter(run_data_MP['alpha'], run_data_MP['#Pools'])
    plt.xlabel("α")
    plt.ylabel("#pools")
    plt.show()

    # ordered dicts with data from each step of each run (the combinations of variable params act as the keys)
    # for example data_collector_model[(0.1, 0.02, 1)] shows the values of the parameters collected at model level
    # when α=0.1 and cost_max=0.02 (which happened to be the run with index 1)
    data_collector_agents = batch_run_MP.get_collector_agents()
    data_collector_model = batch_run_MP.get_collector_model()

# to save the state of a model and resume it later on we can pickle it
# (useful for models that take a lot of time to run)
'''
import pickle
with open("filename.p", "wb") as f:
    pickle.dump(model, f)   
     
...

with open("filename.p", "rb") as f:
    model = pickle.load(f)
'''
