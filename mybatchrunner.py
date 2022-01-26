from mesa.batchrunner import BatchRunnerMP
from tqdm import tqdm
import pandas as pd
import csv


def write_to_csv(filepath, header, row):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell()==0:
            writer.writerow(header)
        writer.writerow(row)


class MyBatchRunner(BatchRunnerMP):
    """
    Child class of BatchRunnerMP, modified to save intermediate output (after every run) to file
    """

    def __init__(self, model_cls, execution_id, nr_processes=None, **kwargs):
        super().__init__(model_cls, nr_processes, **kwargs)
        self.execution_id = execution_id

    def _intermediate_result_prep_mp(self, params, model):
        """
        Save intermediate results to csv file.
        Useful for preventing data loss, even if some executions of the batch run don't terminate.
        The specific file to be used depends on the execution_id. If the file already exists,
        new results will be appended to it (no overwriting).
        @param params: tuple of the parameter values (variable and fixed) used for this run
        @param model: the simulation model produced for this run
        """

        extra_cols = ["Run"]
        header = []
        if self.parameters_list:
            header = list(self.parameters_list[0].keys())
        if self.fixed_parameters:
            fixed_param_cols = list(self.fixed_parameters.keys())
            header += fixed_param_cols
        header += extra_cols

        if self.model_reporters:
            current_model_vars = self.collect_model_vars(model)
            self.model_vars[params] = current_model_vars
            header.extend(current_model_vars.keys())

        row = [param for param in params]
        row.extend([value for value in current_model_vars.values()])

        this_batch_run_intermediate_results = "output/" + self.execution_id + "-intermediate-results.csv"
        all_batch_run_intermediate_results = "output/batch-run-all-intermediate-results.csv"
        write_to_csv(this_batch_run_intermediate_results, header, row)
        write_to_csv(all_batch_run_intermediate_results, header, row)

    def _result_prep_mp(self, results):
        """
        Helper Function
        :param results: Takes results dictionary from Processpool and single processor debug run and fixes format to
        make compatible with BatchRunner Output
        :updates model_vars and agents_vars so consistent across all batchrunner
        """
        # Take results and convert to dictionary so dataframe can be called
        for model_key, model in results.items():
            '''if self.model_reporters:
                self.model_vars[model_key] = self.collect_model_vars(model)'''
            if self.agent_reporters:
                agent_vars = self.collect_agent_vars(model)
                for agent_id, reports in agent_vars.items():
                    agent_key = model_key + (agent_id,)
                    self.agent_vars[agent_key] = reports
            if hasattr(model, "datacollector"):
                if model.datacollector.model_reporters is not None:
                    self.datacollector_model_reporters[
                        model_key
                    ] = model.datacollector.get_model_vars_dataframe()
                if model.datacollector.agent_reporters is not None:
                    self.datacollector_agent_reporters[
                        model_key
                    ] = model.datacollector.get_agent_vars_dataframe()

        # Make results consistent
        if len(self.datacollector_model_reporters.keys()) == 0:
            self.datacollector_model_reporters = None
        if len(self.datacollector_agent_reporters.keys()) == 0:
            self.datacollector_agent_reporters = None

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
                    self._intermediate_result_prep_mp(params, model)

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

    def _prepare_report_table(self, vars_dict, extra_cols=None):
        """
        Creates a dataframe from collected records and sorts it using 'Run'
        column as a key.
        """
        extra_cols = ["Run"] + (extra_cols or [])
        index_cols = []
        if self.parameters_list:
            index_cols = list(self.parameters_list[0].keys())
        if self.fixed_parameters:
            fixed_param_cols = list(self.fixed_parameters.keys())
            index_cols += fixed_param_cols
        index_cols += extra_cols

        records = []
        for param_key, values in vars_dict.items():
            record = dict(zip(index_cols, param_key))
            record.update(values)
            records.append(record)

        df = pd.DataFrame(records)
        rest_cols = set(df.columns) - set(index_cols)
        ordered = df[index_cols + list(sorted(rest_cols))]
        #ordered.sort_values(by="Run", inplace=True)
        return ordered

