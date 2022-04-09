"""Defines schema for storing monitor information."""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from schematics import models
from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils

import yaml

MONITOR_SPEC_NAME = "monitor_spec.yml"


class MonitorDescription(schema_utils.Model):
    """Stores individual monitors and their relevant processing information.

    Attributes:
        monitor: Monitor object.
        joiner_id: String that is used to join monitors across transformations.
        monitor_type: String describing the type of data contained in the monitor.
        scalar_operation: String specifying how to process scalar monitor data.
        vector_operation: String specifying how to process planar monitor data.

    If both scalar_operation and vector_operation are specified, the vector_operation
        is performed first and then the scalar operation is performed.
    """
    monitor = optplan.ReferenceType(optplan.Monitor)
    joiner_id = types.StringType()
    monitor_type = types.StringType(choices=("scalar", "planar", "volume"))
    scalar_operation = types.StringType(
        choices=("magnitude_squared", "magnitude", "phase", "real", "imag"))
    vector_operation = types.StringType(choices=("magnitude", "x", "y", "z"))


class JoinedMonitorDescription(schema_utils.Model):
    """Stores joiner id's and their corresponding monitor information.

    Attributes:
        joiner_id: String that is used to join monitors across transformations.
        monitor_names: List of monitor objects corresponding to the joiner_id.
        monitor_type: String describing the type of data contained in the monitor.
        scalar_operation: String specifying how to process scalar monitor data.
        vector_operation: String specifying how to process planar monitor data.
    """
    joiner_id = types.StringType()
    monitor_names = types.ListType(optplan.ReferenceType(optplan.Monitor))
    monitor_type = types.StringType(choices=("scalar", "planar", "volume"))
    scalar_operation = types.StringType(
        choices=("magnitude_squared", "magnitude", "phase", "real", "imag"))
    vector_operation = types.StringType(choices=("magnitude", "x", "y", "z"))

    def __init__(self, *args, **kwargs) -> None:
        """Creates a new `JoinedMonitorDescription`.

        This allows `JoinedMonitorDescription` to be created in these additional
        ways:
        1) If `joiner_id` is not specified, the first entry in `monitor_names`
           is used.
        2) `monitor_name` can be specified as a single string, which will be
           converted to a list and put into `monitor_names`.
        """
        if "monitor_name" in kwargs:
            kwargs["monitor_names"] = [kwargs["monitor_name"]]
            del kwargs["monitor_name"]
        super().__init__(*args, **kwargs)
        if not self.joiner_id:
            self.joiner_id = self.monitor_names[0]


class MonitorDescriptionList(schema_utils.Model):
    """Stores a list of `JoinedMonitorDescription` objects

    Attributes:
        monitor_list: List of `JoinedMonitorDescription` objects.
    """
    monitor_list = types.ListType(types.ModelType(JoinedMonitorDescription))


def get_monitor_objects(
        monitor_list: List[MonitorDescription],
        filter_by_type: Optional[str] = None) -> List[optplan.Monitor]:
    """Extracts a list of monitor objects.

    Args:
        monitor_list: List of `MonitorDescription` objects.
        filter_by_type: String specifying type of monitor object to return.
            If None, returns a list of all monitor objects.
    Returns:
        List of the monitor objects contained in the `MonitorDescription` objects.
    """
    if filter_by_type:
        return [
            m.monitor for m in monitor_list if m.monitor_type == filter_by_type
        ]
    else:
        return [m.monitor for m in monitor_list]


def convert_element_list(monitor_descriptions: List[MonitorDescription]
                        ) -> MonitorDescriptionList:
    """Joins together monitor description info by joiner_id.

    Args:
        monitor_descriptions: List of `MonitorDescription` objects.

    Returns:
        MonitorDescriptionList object that now has monitor info grouped by joiner_id.

    """
    monitor_dict = {}
    for m in monitor_descriptions:
        if m.joiner_id in monitor_dict:
            # Check that all the other fields are the same for this joined monitor.
            m_joined = monitor_dict[m.joiner_id]
            if not m.monitor_type == m_joined.monitor_type:
                raise ValueError("Monitor type for " + m.monitor.name +
                                 " is inconsistent for joiner id " +
                                 m.joiner_id)
            if not m.scalar_operation == m_joined.scalar_operation:
                raise ValueError("Scalar operation for " + m.monitor.name +
                                 " is inconsistent for joiner id " +
                                 m.joiner_id)
            if not m.vector_operation == m_joined.vector_operation:
                raise ValueError("Vector operation for " + m.monitor.name +
                                 " is inconsistent for joiner id " +
                                 m.joiner_id)

            # All fields are consistent with the joined monitor fields, so add this monitor name.
            monitor_dict[m.joiner_id].monitor_names.append(m.monitor.name)
        else:
            monitor_dict[m.joiner_id] = JoinedMonitorDescription(
                joiner_id=m.joiner_id,
                monitor_names=[m.monitor.name],
                monitor_type=m.monitor_type,
                scalar_operation=m.scalar_operation,
                vector_operation=m.vector_operation)

    return MonitorDescriptionList(monitor_list=list(monitor_dict.values()))


def get_joined_monitor_description(monitor_descriptions: MonitorDescriptionList,
                                   joiner_id: str) -> JoinedMonitorDescription:
    """Gets the `JoinedMonitorDescription` object corresponding to the joiner_id.

    Args:
        monitor_descriptions: MonitorDescriptionList object.
        joiner_id: String specifying the joiner_id.

    Returns:
        JoinedMonitorDescription object.
    """
    for mon in monitor_descriptions.monitor_list:
        if mon.joiner_id == joiner_id:
            return mon


def save_to_yml(monitor_descriptions: MonitorDescriptionList,
                filename: str) -> None:
    """Saves monitor information grouped by joiner_id to yml file.

    Args:
        monitor_descriptions: MonitorDescriptionList object.
        filename: String specifying the yml file name to save to.

    """
    with open(filename, "w") as fp:
        yaml.dump(monitor_descriptions.to_native(), fp)


def load_from_yml(filename: str) -> MonitorDescriptionList:
    """Reads monitor spec information into schema.

    Args:
        filename: String specifying the monitor spec yml filename.

    Returns:
        MonitorDescriptionList object containing the monitor spec information.
    """
    with open(filename) as fp:
        monitor_descriptions = MonitorDescriptionList(yaml.load(fp))
    return monitor_descriptions
