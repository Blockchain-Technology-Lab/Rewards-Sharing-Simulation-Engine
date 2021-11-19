from mesa.batchrunner import BatchRunnerMP
import pickle as pkl
from tqdm import tqdm

class MyBatchRunner(BatchRunnerMP):
    """
    Child class of BatchRunnerMP, exactly the same functionality but saves results to a pickle file to avoid data loss
    """

    def __init__(self, model_cls, nr_processes=None, **kwargs):
        super().__init__(model_cls, nr_processes, **kwargs)

    def run_all(self):
        """
        Run the model at all parameter combinations and store results,
        overrides run_all from BatchRunner.
        """

        run_iter_args, total_iterations = self._make_model_args_mp()
        # register the process pool and init a queue
        # store results in ordered dictionary
        results = {}

        if self.processes > 1:
            with tqdm(total_iterations, disable=not self.display_progress) as pbar:
                for params, model in self.pool.imap_unordered(
                    self._run_wrappermp, run_iter_args
                ):
                    results[params] = model
                    pbar.update()

                #  Added by LadyChristina
                print("doooone")
                pickled_batch_run_results = "output/batch-run-raw-results.pkl"
                with open(pickled_batch_run_results, "wb") as pkl_file:
                    pkl.dump(results, pkl_file)

                self._result_prep_mp(results)
        # For debugging model due to difficulty of getting errors during multiprocessing
        else:
            for run in run_iter_args:
                params, model_data = self._run_wrappermp(run)
                results[params] = model_data

            self._result_prep_mp(results)

        # Close multi-processing
        self.pool.close()

        return (
            getattr(self, "model_vars", None),
            getattr(self, "agent_vars", None),
            getattr(self, "datacollector_model_reporters", None),
            getattr(self, "datacollector_agent_reporters", None),
        )
