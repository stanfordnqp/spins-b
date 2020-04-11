"""Functions to load data from the log files output during spins optimization."""

import collections
import os
import pickle
from typing import Dict, List, NamedTuple, Optional, Set, Union

import flatdict
import numpy as np
import pandas as pd

from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
from spins.invdes.problem_graph.log_tools import monitor_spec

LOG_EXTENSION = "pkl"
LOG_COUNTER = "log_counter"
LOG_TRANSFORMATION_KEY = "transformation"
LOG_TRANSFORMATION_INFO_KEY = "event:state"
LOG_ITERATION_KEY = "event:iteration"
LOG_MONITOR_DATA_KEY = "monitor_data"

SCALAR_TYPE = "scalar"
PLANAR_TYPE = "planar"
VOLUME_TYPE = "volume"

TransformMonitorData = NamedTuple("TransformMonitorData",
                                  [("data", List), ("iteration", List[int])])
"""
NamedTuple representing the monitor data associated with an event in a transformation.

Attributes:
    data: List containing the monitor data for the specified event in the specified
        transformation. Each entry in the list is the data for 1 iteration.
    iteration: List of the iterations for each monitor data value.
"""

MonitorNames = NamedTuple("MonitorNames",
                          [("scalars", monitor_spec.MonitorDescriptionList),
                           ("planars", monitor_spec.MonitorDescriptionList),
                           ("volumes", monitor_spec.MonitorDescriptionList)])
"""
NamedTuple representing `MonitorDescriptionList` objects and the type of monitor contained.

Attributes:
    scalars: List of monitor descriptions for 1-D scalar monitors (e.g. monitors for objective function, powers etc.).
    planars: List of monitor descriptions for 2-D planar monitors (e.g. monitors for 2D field slices or 2D epsilon slices).
    volumes: List of monitor descriptions for 3-D volume monitors (e.g. monitors for full 3D fields or epsilon).
"""

ScalarPlotData = NamedTuple("ScalarPlotData",
                            [("data", List[float]),
                             ("transformation_changes", List[int]),
                             ("iterations", List[int])])
"""
NamedTuple representing data used for plotting scalar transformation data versus iteration.

Attributes:
    data: List of the monitor data for each iteration.
    transformation_changes: List of indices at which the transformation has changed.
    iterations: List of iterations. Iterations are cumulative across transformations.

"""


def create_log_data_frame(log_dict_list: List[Dict]) -> pd.DataFrame:
    """Creates a single pandas dataframe containing all the spins optimization output data.

    Uses flatdict.FlatDict to transform the hierarchical dictionary data from the SPINS output
    pickle files into a flat dictionary with colon-separated keys.

    Args:
        log_dict_list: List of dictionaries contained in the spins optimization pickle file outputs.

    Returns:
        Single pandas dataframe with all the data in the list of dictionaries sorted
        according to the order in which the log data was written.
    """
    log_df = pd.DataFrame()
    for log_dict in log_dict_list:
        # Flatten the log dictionary.
        flat_log = flatdict.FlatDict(log_dict)

        # Replace all entries in the flattened log dictionaries that are not strings
        # with a list containing the entry to allow array data to be stored
        # in the pandas dataframe cell.
        for key in flat_log.keys():
            if not isinstance(flat_log[key], str):
                flat_log[key] = [flat_log[key]]

        # Create a pandas dataframe from the flattened log dictionary and
        # concatenate it with the existing pandas dataframe which will eventually
        # store all the log information.
        single_log_df = pd.DataFrame(dict(flat_log), index=[0])
        log_df = pd.concat([log_df, single_log_df],
                           axis=0,
                           ignore_index=True,
                           sort=False)
    if LOG_COUNTER in log_df.columns:
        log_df = log_df.sort_values(by=[LOG_COUNTER])

    return log_df


def load_all_logs(log_dir: str) -> Dict:
    """Loads all the spins optimization pickle files in the log directory.

    Args:
        log_dir: Directory containing the pickle file outputs from optimization.

    Returns:
        List of the dictionaries contained in the optimization pickle files.
    """
    # Create a list of all the log files in the log directory.
    logs_name_list = []
    for dir_file in os.listdir(log_dir):
        if dir_file.endswith(LOG_EXTENSION):
            logs_name_list.append(dir_file)

    # Create a list of all the log dictionaries loaded from each log file.
    log_dict_list = []
    for file_name in logs_name_list:
        with open(os.path.join(log_dir, file_name), "rb") as log_file:
            data = pickle.load(log_file)
            # Ignore files that do not have any monitors.
            if "monitor_data" not in data:
                continue
            log_dict_list.append(data)
    return log_dict_list


def get_monitor_data(log_df: pd.DataFrame, monitor_names: Union[str, List[str]],
                     event_name: str) -> collections.OrderedDict:
    """Retrieves all the monitor data associated with the given event.

    Args:
        log_df: Pandas dataframe containing all optimization output information.
        monitor_names: List of monitor names to grab. If `monitor_names` is
            a single string, it is converted to a list with one element.
        event_name: String specifying the event within the transformation for
            which the monitor data should be returned.

    Returns:
        `OrderedDict` whose keys are all transformations for which
        `monitor_name` has data during event_name and whose values are the
        monitor data and iteration data in `TransformMonitorData`.
        The keys in the `OrderedDict` are in the order of transformation
        occurrence (the last key is the last transformation performed).
    """
    if isinstance(monitor_names, str):
        monitor_names = [monitor_names]

    # First, filter by event name.
    log_df_event = log_df[log_df[LOG_TRANSFORMATION_INFO_KEY] == event_name]
    all_transformation_names = log_df[LOG_TRANSFORMATION_KEY].unique().tolist()

    # Store monitor data in an ordered dict so transformations in correct order.
    monitor_data = collections.OrderedDict()

    for transformation_name in all_transformation_names:
        # Filter again by the transformation name.
        log_df_event_trans = log_df_event[log_df_event[LOG_TRANSFORMATION_KEY]
                                          == transformation_name]
        for name in monitor_names:
            full_name = LOG_MONITOR_DATA_KEY + ":" + name
            if full_name not in log_df_event_trans:
                continue

            mask = ~log_df_event_trans[full_name].isnull()
            mon_dat = log_df_event_trans[full_name][mask].tolist()

            # Check that there is iteration data for this monitor.
            if LOG_ITERATION_KEY in log_df_event_trans:
                iter_data = log_df_event_trans[LOG_ITERATION_KEY][
                    mask &
                    ~log_df_event_trans[LOG_ITERATION_KEY].isnull()].tolist()
            else:
                iter_data = []
            if mon_dat:
                monitor_data[transformation_name] = TransformMonitorData(
                    mon_dat, list(np.int_(iter_data)))
                break
    return monitor_data


def get_single_monitor_data(
        log_df: pd.DataFrame,
        monitor_names: Union[str, List[str]],
        transformation_name: Optional[str] = None,
        iteration: Optional[int] = None,
        event_name: Optional[str] = None) -> Union[float, List]:
    """Gets a single set of monitor data corresponding to the specified event,
        transformation, and iteration.

    Args:
        log_df: Pandas dataframe containing all spins optimization output
            information.
        monitor_names: List of monitor names to fetch.
        transformation_name: String specifying the transformation of interest.
        iteration: Int specifying the iteration of interest.
        event_name: String specifying the event within the transformation for
            which the monitor data should be returned.

    Returns:
        The single set of monitor data for specified event, transformation, and
        iteration. If no event is given, data for the last event is returned.
        If no transformation name is given, data for the last transformation
        performed is returned. If no iteration is given, data for the last
        iteration available is returned.
    """

    # If no transformation name given, extract all transformation names to find the
    # last one with desired event data.
    if transformation_name is None:
        all_transformation_names = log_df[LOG_TRANSFORMATION_KEY].unique(
        ).tolist()
    else:
        all_transformation_names = [transformation_name]

    # If no event name given, extract all event names to find the last one with
    # data.
    if event_name is None:
        all_event_names = log_df[LOG_TRANSFORMATION_INFO_KEY].unique().tolist()
    else:
        all_event_names = [event_name]

    # Get all data for the given monitor name associated with the event_name.
    for name_transformation in reversed(all_transformation_names):
        for name_event in reversed(all_event_names):
            data_all = get_monitor_data(log_df, monitor_names, name_event)

            # Check if data for the given transformation name was found.
            if not data_all or name_transformation not in data_all.keys():
                continue

            transformation_key = name_transformation

            if iteration in data_all[transformation_key].iteration:
                iteration_index = data_all[transformation_key].iteration.index(
                    iteration)
            else:
                iteration_index = -1
            return data_all[transformation_key].data[iteration_index]

    return []


def get_monitors_by_type(
        log_df: pd.DataFrame,
        monitor_descriptions: monitor_spec.MonitorDescriptionList
) -> MonitorNames:
    """Sorts the monitor information by the type of monitor data contained.

    Args:
        log_df: Pandas dataframe containing all spins optimization output information.
        monitor_descriptions: MonitorDescriptionList object.

    Returns:
            MonitorNames NamedTuple.

    """

    scalar_monitors = []
    planar_monitors = []
    volume_monitors = []
    for m in monitor_descriptions.monitor_list:
        monitor_type = m.monitor_type
        monitor_joiner_id = m.joiner_id
        if monitor_type == SCALAR_TYPE:
            scalar_monitors.append(m)
        elif monitor_type == PLANAR_TYPE:
            planar_monitors.append(m)
        elif monitor_type == VOLUME_TYPE:
            volume_monitors.append(m)
    return MonitorNames(
        monitor_spec.MonitorDescriptionList(monitor_list=scalar_monitors),
        monitor_spec.MonitorDescriptionList(monitor_list=planar_monitors),
        monitor_spec.MonitorDescriptionList(monitor_list=volume_monitors))


def get_overlap_monitor_names(log_df: pd.DataFrame) -> List[str]:
    """Returns a list of names of overlap monitors.

    Args:
        log_df: Pandas dataframe containing all spins optimization output information.

    Returns:
        List of overlap monitor name strings.
    """
    monitor_names = get_monitor_name_by_type(log_df)
    overlap_names = []
    for name in monitor_names.scalars:
        # Currently, all overlap monitors have an autogen name including "overlap".
        if "overlap" in name.lower():
            overlap_names.append(name)
    return overlap_names


def get_joined_scalar_monitors(
        log_df: pd.DataFrame,
        monitor_names: Union[str, List[str]],
        event_name: str,
        scalar_operation: Optional[str],
) -> ScalarPlotData:
    """Joins scalar monitors data across all transformations.

    Args:
        log_df: Pandas dataframe containing all optimization output information.
        monitor_names: List of monitor names to grab. If `monitor_names` is
            a single string, it is converted to a list with one element.
        event_name: String specifying the event within the transformation for
            which the monitor data should be returned.
        scalar_operation: Scalar operation to apply on the data elements.
            See `process_scalar` for details.

    Returns:
       `ScalarPlotData` containing the joined data.
    """
    data = []
    transformation_changes = []
    iterations = []
    monitor_data = get_monitor_data(log_df, monitor_names, event_name)
    for transformation in monitor_data.keys():
        transformation_data = monitor_data[transformation].data
        transformation_iters = monitor_data[transformation].iteration
        dat_length = len(data)
        if dat_length > 0:
            iterations.extend(transformation_iters + iterations[-1])
            transformation_changes.append(dat_length)
        else:
            iterations.extend(transformation_iters)
        data.extend(transformation_data)

    data = process_scalar(data, scalar_operation=scalar_operation)
    return ScalarPlotData(data, transformation_changes, iterations)


def process_scalar(data: Union[float, List],
                   scalar_operation: Optional[str]) -> Union[float, List]:
    """Performs scalar operations on monitor data.

    Args:
        data: Float or list of complex monitor data.
        scalar_operation: String indicating scalar operation to perform.

    Returns:
        Data where each element has the corresponding scalar operation applied.
        Does nothing is `scalar_operation` is `None`.
    """
    if not scalar_operation:
        return data
    elif scalar_operation.lower() == "magnitude_squared":
        return np.absolute(data)**2
    elif scalar_operation.lower() == "magnitude":
        return np.absolute(data)
    elif scalar_operation.lower() == "phase":
        return np.angle(data)
    elif scalar_operation.lower() == "real":
        return np.real(data)
    elif scalar_operation.lower() == "imag":
        return np.imag(data)
    else:
        raise ValueError(
            "Unknown scalar operation, got {}".format(scalar_operation))


def process_field(field: List,
                  vector_operation: Optional[str] = None,
                  scalar_operation: Optional[str] = None) -> np.ndarray:
    """Calculates the magnitude of the inputted field data.

    Args:
        field: Field slice data as a list of x, y, and z 2-dimensional data.
        vector_operation: String specifying the component of the field data.
            Choices are "magnitude", "x", "y", or "z".
        scalar_operation: String specifying how to process each element of the
            field data. Choices are "power", "magnitude", "phase", "real",
            or "imag".

    Returns:
        List containing the processed field data.
    """
    if vector_operation.lower() == "magnitude":
        mag = (np.sqrt(
            np.abs(field[0])**2 + np.abs(field[1])**2 + np.abs(field[2])**2))
    elif vector_operation.lower() == "x":
        mag = field[0]
    elif vector_operation.lower() == "y":
        mag = field[1]
    elif vector_operation.lower() == "z":
        mag = field[2]
    mag = process_scalar(mag, scalar_operation=scalar_operation)
    return mag


def display_summary(
        log_df: pd.DataFrame,
        monitor_descriptions: monitor_spec.MonitorDescriptionList) -> None:
    """Displays to the command line the most recent values for all scalar monitors.

    Args:
        log_df: Pandas dataframe containing all spins optimization output information.

        monitor_descriptions: MonitorDescriptionList object.
    """

    monitor_list = get_monitors_by_type(log_df, monitor_descriptions)
    scalar_monitors = monitor_list.scalars
    for mon in scalar_monitors.monitor_list:
        # Get all the transformation data for the monitor name and join it.
        monitor_data = get_single_monitor_data(log_df, mon.monitor_names)
        monitor_data = process_scalar(
            monitor_data, scalar_operation=mon.scalar_operation)
        print("{name}: {value:1.4E}".format(
            name=mon.joiner_id, value=monitor_data))
