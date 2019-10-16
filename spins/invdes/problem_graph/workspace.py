"""The workspace manages the creation and execution of the optplan.

The workspace has the following responsibilities:
    1) Creating and caching problem graph node objects.
    2) Handling execution of transformations.
"""
import collections
from datetime import datetime
import glob
import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Union

from spins.invdes import parametrization
from spins.invdes import problem
from spins.invdes.problem import graph_executor
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace

# Special node to denote the structure variable.
VARIABLE_NODE = "__variable"


class Workspace:
    """Manages optplan node creation and caching.

    Nodes that are added to the workspace are considered immutable. Changes made
    to the node specification after adding to the workspace may or may not
    have any effect.
    """

    def __init__(self,
                 filepath: str = ".",
                 savepath: str = ".",
                 nodes: Optional[Union[optplan.ProblemGraphNode, List[
                     optplan.ProblemGraphNode]]] = None) -> None:
        """Initializes the workspace.

        Args:
            filepath: Root folder for the workspace. This is used as the root
                folder for loading any auxiliary files.
            savepath: Path to save any logs.
            nodes: Optional list of nodes that exist in the graph.
        """
        self._filepath = filepath

        # Mapping from node name to the actual node. This is for node
        # definition lookup as well as to quickly see which nodes are part of
        # the graph.
        self._nodes = {}
        # List of all the objects that have been instantiated.
        self._objects = {}

        # TODO(logansu): Accept argument-less variable.
        self._objects[VARIABLE_NODE] = problem.Variable(1)

        if isinstance(nodes, collections.Iterable):
            for node in nodes:
                self._add_node(node)
        elif nodes:
            self._add_node(nodes)

        # Setup logging if applicable.
        self._logger = None
        if savepath:
            self._logger = Logger(savepath, self)

    def _add_node(self, node: optplan.ProblemGraphNode) -> None:
        """Adds a node to the workspace.

        If the node has already been added, then it is ignored.

        Args:
            node: The node to add.
        """

        def add_node(node: optplan.ProblemGraphNode) -> bool:
            if node.name in self._nodes:
                if id(node) != id(self._nodes[node.name]):
                    raise ValueError("Node with same name added, got {}".format(
                        node.name))
                return False

            self._nodes[node.name] = node
            return True

        def process_field(
                unused_parent: optplan.ProblemGraphNode,
                child: optplan.ProblemGraphNode) -> optplan.ProblemGraphNode:
            add_node(child)

        if add_node(node):
            visited = set(id(node_) for node_ in self._nodes)
            # TODO(logansu): Deal with this private import.
            from spins.invdes.problem_graph.optplan.io import _iter_optplan_fields
            _iter_optplan_fields(node, visited, process_field)

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def logger(self) -> "Logger":
        return self._logger

    def run(self,
            node: optplan.Transformation,
            event_data: Optional[Dict] = None) -> None:
        """Runs a transformation.

        All nodes required for the transformation are automatically created.

        Args:
            node: Transformation to run.
            event_data: Event data used to restore transformations.
        """
        # Create the transformation.
        creator = optplan.GLOBAL_CONTEXT_STACK.get_node_creator(
            optplan.NodeMetaType.TRANSFORMATION, node.transformation.type)
        if creator is None:
            raise ValueError("Unable to find creator for transformation with "
                             "type {}".format(node.type))

        if self._logger:
            self._logger.set_transformation_name(node.name)

        transform = creator(node.transformation, self)
        # TOOD(logansu): Factor this out somehow.
        # Load parameters from `event_data` if provided.
        if not event_data:
            _set_parameters(self, node.parameter_list)
        transform(self.get_object(node.parametrization), event_data)

        if self._logger:
            self._logger.set_transformation_name(None)

    def get_object(self,
                   name_or_node: Union[str, optplan.ProblemGraphNode],
                   return_graph_node: bool = False):
        """Creates or retrieves a problem graph object.

        Based on `name_or_node`, if the corresponding object does not exist,
        then it is created. If it was previously created, then the previously
        created object is returned.

        When a node is created, all nodes that it depends on are also created.
        An error is raised if a node by the same name but different id is
        added.

        Args:
            name_or_node: Either the name of a node that has been added or the
                node description itself.
            return_graph_node: If `True`, the node information is returned
                as well.

        Returns:
            If `return_graph_node` is `True`, a tuple `(obj, node)` where
            `obj` is the object and `node` is the node description. Otherwise,
            the object is returned.

        Raises:
            ValueError: If `name_or_node` is the wrong type.
        """
        if name_or_node == VARIABLE_NODE:
            return self._objects[VARIABLE_NODE]

        # Figure out if we are given the name of the node or the actual node
        # itself.
        if isinstance(name_or_node, str):
            node = self._nodes[name_or_node]
        elif isinstance(name_or_node, optplan.ProblemGraphNode):
            node = name_or_node
        else:
            raise ValueError(
                "`name_or_node` must be string or `ProblemGraphNode`"
                ", got {}".format(type(name_or_node)))

        # Return the cached object. Note that we do this before `_add_node`
        # just in case someone decided to modify `node` after adding it to
        # the workspace.
        if node.name in self._objects:
            if return_graph_node:
                return self._objects[node.name], self._nodes[node.name]
            else:
                return self._objects[node.name]

        # Update the node list with all the nodes that are part of the graph.
        self._add_node(node)

        # Create the actual object.
        creator = optplan.GLOBAL_CONTEXT_STACK.get_node_creator(
            optplan.NodeMetaType.OPTPLAN_NODE, node.type)
        if creator is None:
            self._objects[node.name] = node
        else:
            self._objects[node.name] = creator(node, self)

        if return_graph_node:
            return self._objects[node.name], self._nodes[node.name]
        else:
            return self._objects[node.name]

    def get_objects_by_type(
            self,
            model_type: optplan.ProblemGraphNode,
            return_graph_node: bool = False,
    ) -> Dict:
        """Returns a dictionary of all the nodes with a given type.

        Args:
            model_type: Type of node to return.
            return_graph_node: Return the original graph node as well.

        Returns:
            A dictionary with keys corresponding to node names and values
            corresponding to the objects. If `return_graph_node` is `True`,
            returns `Dict[str, Tuple[object, optplan.ProblemGraphNode]]`.
        """
        objects = {}
        for node_name, node in self._nodes.items():
            if isinstance(node, model_type):
                objects[node.name] = self.get_object(node, return_graph_node)

        return objects


def _set_parameters(work: workspace.Workspace,
                    parameter_list: Optional[List[optplan.SetParam]]) -> None:
    """Sets parameters in `parameter_list`.

    Args:
        work: Workspace.
        parameter_list: List of parameter set commands to execute.
    """
    if not parameter_list:
        return

    # TODO(logansu): Parallelize the objective function evaluation.
    for set_parameter in parameter_list:
        parameter = work.get_object(set_parameter.parameter)
        function = work.get_object(set_parameter.function)
        function_value = function.calculate_objective_function(
            work.get_object(set_parameter.parametrization))

        if set_parameter.inverse:
            parameter.set_parameter_value(function_value**-1)
        else:
            parameter.set_parameter_value(function_value)


class Logger:
    """Handles logging for transformations.

    The logger saves the states of monitors and all the parameters for
    transformations. Currently a new Pickle file is created for each log entry.
    """

    def __init__(self, log_path: str, work: Workspace) -> None:
        """Initilizes a logger.

        Args:
            log_path: Directory in which to save the monitors. If the path does
                not exist, it will be automatically created.
            work: Workspace containing all the monitors.
        """
        self._path = log_path
        self._work = work

        if not os.path.exists(self._path):
            os.makedirs(self._path)

        # Name of the current transformation.
        self._transform_name = None

        # Keeps track of the number of logs saved.
        self._log_counter = 0

        # Logging to screen.
        self._logger = logging.getLogger(__name__)

    def set_transformation_name(self, name: str) -> None:
        """Set the name of the transformation.

        Args:
            name: String specifying the tranformation.
        """
        self._transform_name = name

    def write_checkpoint(self, filename: str) -> None:
        """Writes a checkpoint file.

        A checkpoint file contains the full state, including all the
        parametrizations and all the parameters. This is used to restore
        the full state of the optimization.

        Args:
            filename: Name of the checkpoint file.
        """
        # Get workspace parameters.
        parameter_data = {}
        parameter_list = self._work.get_objects_by_type(optplan.Parameter)
        for param_name, param_obj in parameter_list.items():
            parameter_data[param_name] = param_obj.value

        # Get parametrizations.
        parametrization_data = {}
        param_list = self._work.get_objects_by_type(optplan.Parametrization)
        for name, obj in param_list.items():
            parametrization_data[name] = obj.serialize()

        data = {
            "time": str(datetime.now()),
            "parameters": parameter_data,
            "parametrizations": parametrization_data,
        }

        checkpoint_file = filename + ".chkpt.pkl"
        self._logger.info("Saving checkpoint file: %s", checkpoint_file)

        # Save the data.
        file_path = os.path.join(self._path, checkpoint_file)
        with open(file_path, "wb") as handle:
            pickle.dump(data, handle)

    def write(self,
              event: Dict,
              param: parametrization.Parametrization,
              monitor_list: List[str] = None) -> None:
        """Write monitor data to log file.

        Args:
            event: Transformation-specific information about the event.
            param: Parametrization that has to be evaluated.
            monitor_list: List of monitor names to be evaluated.
        """
        # Increment log_counter.
        self._log_counter += 1

        # Get monitor data.
        monitor_data = {}
        if monitor_list:
            mon_vals = graph_executor.eval_fun(
                [self._work.get_object(mon) for mon in monitor_list], param)
            for mon, mon_val in zip(monitor_list, mon_vals):
                monitor_data[mon.name] = mon_val

        # Get workspace parameters.
        parameter_data = {}
        parameter_list = self._work.get_objects_by_type(optplan.Parameter)
        for param_name, param_obj in parameter_list.items():
            parameter_data[param_name] = param_obj.calculate_objective_function(
                param)

        # Make a log entry.
        data = {
            "transformation": self._transform_name,
            "event": event,
            "time": str(datetime.now()),
            "parametrization": param.serialize(),
            "parameters": parameter_data,
            "monitor_data": monitor_data,
            "log_counter": self._log_counter
        }

        self._logger.info(
            "Saving monitors for transformation %s with event info %s [%d].",
            self._transform_name, event, self._log_counter)

        # Save the data.
        file_path = os.path.join(
            self._path, os.path.join("step{}.pkl".format(self._log_counter)))
        with open(file_path, "wb") as handle:
            pickle.dump(data, handle)


def get_latest_log_step(folder: str) -> int:
    """Retrieves the latest log file step.

    This is done by finding all files in `folder` with name of the form
    "stepXX.pkl" where `XX` is a number, and then returning the largest
    number.

    Args:
        folder: Folder containing the log files.

    Returns:
        Last log file step.
    """
    # Find the last logged Pickle file.
    filenames = glob.glob(os.path.join(folder, "step*.pkl"))
    # Keep track of the file with the largest step.
    max_step = 0
    for name in filenames:
        match = re.search(r"step(?P<step>\d+)\.pkl$", name)
        if match:
            max_step = max(max_step, int(match.group("step")))
    return max_step


def get_latest_log_file(folder: str) -> Optional[str]:
    """Retrieves the latest log filename.

    This performs the same operation as `get_max_log_step` except it returns
    the name of the last log file rather than the actual log step number.

    Args:
        folder: Folder containing log files.

    Returns:
        Filename of the last log step if it exists. Else `None`.
    """
    max_step = get_latest_log_step(folder)
    if max_step == 0:
        return None
    return os.path.join(folder, "step{}.pkl".format(max_step))
