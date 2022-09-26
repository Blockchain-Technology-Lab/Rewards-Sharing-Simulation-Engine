"""
Copy of mesa's batchrunner, slightly modified for our purposes (e.g. save results to csv file as they come)
"""
from functools import partial
import itertools
from tqdm import tqdm
from copy import copy
from multiprocessing import Pool
from typing import (
    Any,
    Counter,
    Dict,
    Iterable,
    List,
    Mapping,
    Tuple,
    Type,
    Union,
)
import pathlib
from mesa.model import Model
import logic.helper as hlp

def custom_batch_run(
    model_cls,
    parameters,
    batch_run_id,
    number_processes=1,
    iterations=1,
    data_collection_period=-1,
    max_steps=1000,
    display_progress= True,
    initial_seed=0
):
    """Batch run a mesa model with a set of parameter values.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to batch-run
    parameters : Mapping[str, Union[Any, Iterable[Any]]],
        Dictionary with model parameters over which to run the model. You can either pass single values or iterables.
    number_processes : int, optional
        Number of processes used, by default 1. Set this to None if you want to use all CPUs.
    iterations : int, optional
        Number of iterations for each parameter combination, by default 1
    data_collection_period : int, optional
        Number of steps after which data gets collected, by default -1 (end of episode)
    max_steps : int, optional
        Maximum number of model steps after which the model halts, by default 1000
    display_progress : bool, optional
        Display batch run process, by default True

    Returns
    -------
    List[Dict[str, Any]]
        [description]
    """

    seq_id = hlp.read_seq_id() + 1
    hlp.write_seq_id(seq_id)
    batch_run_id = str(seq_id) + '-' + batch_run_id
    path = pathlib.Path.cwd() / "output" / batch_run_id
    pathlib.Path(path).mkdir(parents=True)

    parameters.update({'parent_dir': path})
    kwargs_list, fixed_params = _make_model_kwargs(parameters)
    final_kwargs_list = [
        copy(kwarg_dict) | {'seed': initial_seed + i} # add different seed for each iteration with same param combination
        for kwarg_dict in kwargs_list
        for i in range(iterations)
    ]
    for i, params in enumerate(final_kwargs_list):
        seq_id = i + 1
        execution_id = '-'.join([str(key) + '-' + str(value) for key, value in params.items() if key not in fixed_params])
        params.update({'seq_id': seq_id, 'execution_id': execution_id})
        print(execution_id)
    process_func = partial(
        _model_run_func,
        model_cls,
        max_steps=max_steps,
        data_collection_period=data_collection_period,
        batch_run_id=batch_run_id,
        fixed_params=fixed_params
    )

    total_iterations = len(final_kwargs_list)
    run_counter = itertools.count()

    results: List[Dict[str, Any]] = []

    with tqdm(total_iterations, disable=not display_progress) as pbar:
        iteration_counter: Counter[Tuple[Any, ...]] = Counter()

        def _fn(paramValues, rawdata):
            iteration_counter[paramValues] += 1
            iteration = iteration_counter[paramValues]
            run_id = next(run_counter)
            data = []
            for run_data in rawdata:
                out = {"RunId": run_id, "iteration": iteration - 1}
                out.update(run_data)
                data.append(out)
            results.extend(data)
            pbar.update()

        if number_processes == 1:
            for kwargs in final_kwargs_list:
                paramValues, rawdata = process_func(kwargs)
                _fn(paramValues, rawdata)
        else:
            with Pool(number_processes) as p:
                for paramValues, rawdata in p.imap_unordered(process_func, final_kwargs_list):
                    _fn(paramValues, rawdata)

    return results, path


def _make_model_kwargs(
    parameters: Mapping[str, Union[Any, Iterable[Any]]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Create model kwargs from parameters dictionary.

    Parameters
    ----------
    parameters : Mapping[str, Union[Any, Iterable[Any]]]
        Single or multiple values for each model parameter name

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        A list of all kwargs combinations and a list of all variable parameter names
    """
    parameter_list = []
    fixed_params = []
    for param, values in parameters.items():
        if isinstance(values, str):
            # The values is a single string, so we shouldn't iterate over it.
            all_values = [(param, values)]
            fixed_params.append(param)
        else:
            try:
                all_values = [(param, value) for value in values]
            except TypeError:
                all_values = [(param, values)]
                fixed_params.append(param)
        parameter_list.append(all_values)
    all_kwargs = itertools.product(*parameter_list)
    kwargs_list = [dict(kwargs) for kwargs in all_kwargs]
    return kwargs_list, fixed_params


def _model_run_func(
    model_cls: Type[Model],
    kwargs: Dict[str, Any],
    max_steps: int,
    data_collection_period: int,
    batch_run_id: str,
    fixed_params: List[str]
) -> Tuple[Tuple[Any, ...], List[Dict[str, Any]]]:
    """Run a single model run and collect model and agent data.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to batch-run
    kwargs : Dict[str, Any]
        model kwargs used for this run
    max_steps : int
        Maximum number of model steps after which the model halts, by default 1000
    data_collection_period : int
        Number of steps after which data gets collected

    Returns
    -------
    Tuple[Tuple[Any, ...], List[Dict[str, Any]]]
        Return model_data, agent_data from the reporters
    """
    model = model_cls(**kwargs)
    while model.running and model.schedule.steps <= max_steps:
        model.step()

    data = []

    steps = list(range(0, model.schedule.steps, data_collection_period))
    if not steps or steps[-1] != model.schedule.steps - 1:
        steps.append(model.schedule.steps - 1)

    for step in steps:
        model_data, all_agents_data = _collect_data(model, step)

        # If there are agent_reporters, then create an entry for each agent
        if all_agents_data:
            stepdata = [
                {**{"Step": step}, **kwargs, **model_data, **agent_data}
                for agent_data in all_agents_data
            ]
        # If there is only model data, then create a single entry for the step
        else:
            stepdata = [{**{"Step": step}, **kwargs, **model_data}]

            if step == model.schedule.steps - 1:
                # write end-of-run results to file
                header = [key for key in stepdata[0].keys() if key not in fixed_params]
                row = [value for key, value in stepdata[0].items() if key not in fixed_params]
                filename = 'aggregate-results-' + batch_run_id + '.csv'
                path = pathlib.Path.cwd() / "output" / batch_run_id
                filepath = path / filename
                hlp.write_to_csv(filepath, header, row)
        data.extend(stepdata)

    return tuple(kwargs.values()), data


def _collect_data(
    model: Model,
    step: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Collect model and agent data from a model using mesas datacollector."""
    dc = model.datacollector
    model_data = {param: values[step] for param, values in dc.model_vars.items()}

    all_agents_data = []
    raw_agent_data = dc._agent_records.get(step, [])
    for data in raw_agent_data:
        agent_dict = {"AgentID": data[1]}
        agent_dict.update(zip(dc.agent_reporters, data[2:]))
        all_agents_data.append(agent_dict)
    return model_data, all_agents_data
