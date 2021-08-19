from multiprocessing import freeze_support

from mesa.batchrunner import BatchRunner
from mesa.batchrunner import BatchRunnerMP
import time
import matplotlib.pyplot as plt

import logic.sim as sim

if __name__ == '__main__':
    # freeze_support()  # needed for multiprocessing to work on windows systems (comment out line to run on linux / uncomment for windows)

    fixed_params = {"n": 100}

    variable_params = {"k": [k for k in range(1, 31)]}
    #variable_params = {"alpha": [a/10 for a in range(11)]}
    # todo figure out how to run the model with only certain combinations of the variable params

    # only use the non-multiprocessing batch-run (uncomment the lines above and comment the lines below) if the multiprocessing one doesn't work for some reason
    '''batch_run = BatchRunner(Simulation,
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
    plt.show()'''
    # only use the non-multiprocessing batch-run (uncomment the lines above and comment the lines below) if the multiprocessing one doesn't work for some reason

    batch_run_MP = BatchRunnerMP(sim.Simulation,
                                 nr_processes=12,
                                 variable_parameters=variable_params,
                                 fixed_parameters=fixed_params,
                                 iterations=1,
                                 max_steps=100,
                                 model_reporters={"#Pools": sim.get_number_of_pools, "avgPledge": sim.get_avg_pledge})
    start_time = time.time()
    batch_run_MP.run_all()
    print("Batch run with multiprocessing:  {:.2f} seconds".format(time.time() - start_time))

    # Extract data from the batch runner
    run_data_MP = batch_run_MP.get_model_vars_dataframe()
    # print(run_data_MP.head())
    plt.figure()
    plt.scatter(run_data_MP['k'], run_data_MP['#Pools'])
    plt.xlabel("k")
    plt.ylabel("#pools")
    '''plt.scatter(run_data_MP['alpha'], run_data_MP['avgPledge'], c='r')
    plt.xlabel("alpha")
    plt.ylabel("Average pledge")'''
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
