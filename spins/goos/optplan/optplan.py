from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

import copy
import dataclasses
import datetime
import inspect
import json
import logging
import os
import pickle

import numpy as np

from spins.goos import flows
from spins.goos import optplan


class ProblemGraphNodeMeta(type):

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        # Check that the node has a name.
        if not hasattr(cls, "node_type"):
            raise ValueError(
                "Problem graph node {} did not define `node_type`.".format(cls))

        # `ProblemGraphNode` has special implementation. Do not treat as normal
        # node.
        if cls.node_type == "goos.problem_graph_node" or cls.node_type == "goos.action":
            return cls

        # Validate that the constructor arguments are valid.
        signature = inspect.signature(cls.__init__)
        # Skip the first parameter, which is always "self".
        for param in list(signature.parameters.values())[1:]:
            # Error out if parameter is named for items with meaning.
            if param.name == "name":
                raise TypeError("Parameter cannot be named 'name'.")
            elif param.name == "type":
                raise TypeError("Parameter cannot be named 'type'.")

            # Parameters should have type annotation.
            if param.annotation == param.empty:
                raise TypeError("Parameter `{}` has no type annotation.".format(
                    param.name))

            # Default values need to be serializable.
            if param.default != param.empty and False:
                raise ValueError(
                    "Parameter `{}` has unserializable default value,"
                    " got {}".format(param.name, param.default))

            # We cannot handle variational parameters.
            if (param.kind == param.VAR_POSITIONAL or
                    param.kind == param.VAR_KEYWORD):
                raise TypeError("Problem graph node \"{}\" has a variational"
                                " parameter.".format(cls.node_type))

        # Create the schema class.
        # Check if a class called `Schema` is declared in the class itself
        # (as opposed to one of its parents).
        if "Schema" not in cls.__dict__:
            # We need to construct a schema inheritance hierarchy that parallels
            # the inheritance hierarchy in the actual nodes.
            schema_base_classes = []
            for base in bases:
                if issubclass(base, ProblemGraphNode):
                    schema_base_classes.append(base.Schema)

            # If the class explicitly defines `__init__`, then build up a schema
            # adding arguments from its constructor. Else, build a schema
            # for the sake of preserving hierarchy but do not add any fields.
            if "__init__" in cls.__dict__:
                cls.Schema = optplan.schema.construct_schema(
                    cls.node_type,
                    cls.__init__,
                    skip_first_arg=True,
                    base_classes=schema_base_classes,
                    other_fields={
                        "type": optplan.types.StringType(default=cls.node_type)
                    })
            else:
                cls.Schema = type(cls.node_type, tuple(schema_base_classes), {})
        if not cls.goos_no_register:
            optplan.GLOBAL_CONTEXT_STACK.register_node(cls.node_type,
                                                       cls.Schema, cls)
        return cls

    def __call__(cls, *args, **kwargs):
        # `ProblemGraphNode` has special implementation. Do not treat as normal
        # node.
        if cls.node_type == "goos.problem_graph_node" or cls.node_type == "goos.action":
            return super(ProblemGraphNodeMeta, cls).__call__(*args, **kwargs)

        if "name" in kwargs:
            node_name = kwargs["name"]
            del kwargs["name"]
        else:
            node_name = None

        # Verify that all arguments are serializable and save argument values.
        signature = inspect.signature(cls.__init__)
        schema_data = {}
        # Index of the next signature parameter to check.
        # Start at 1 to ignore `self`.
        param_ind = 1
        parameters = list(signature.parameters.values())
        # Parse out position arguments.
        for arg in args:
            schema_data[parameters[param_ind].name] = arg
            param_ind += 1
        # Parse out keyword arguments.
        for param in parameters[param_ind:]:
            if param.name in kwargs:
                schema_data[param.name] = kwargs[param.name]
            else:
                schema_data[param.name] = param.default

        obj = super(ProblemGraphNodeMeta, cls).__call__(*args, **kwargs)

        # Name the node and set the schema.
        if node_name:
            obj._goos_name = node_name
        else:
            obj._goos_name = optplan.schema.generate_name(cls.node_type)

        # Allow the node to custom override the way the schema is set.
        if not hasattr(obj, "_goos_schema") or not obj._goos_schema:
            schema_data["name"] = obj._goos_name
            schema_data = _replace_node_with_schema(schema_data)
            obj._goos_schema = cls.Schema(**schema_data)
        else:
            obj._goos_schema.name = obj._goos_name
        # TODO(logansu): Validate the schema.

        default_plan = get_default_plan()
        if default_plan:
            default_plan.add_node(obj)
        return obj


def _replace_node_with_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_replace_node_with_schema(item) for item in value]
    elif isinstance(value, tuple):
        # TODO
        pass
    elif isinstance(value, dict):
        # TODO: check key of dictionary
        return {
            key: _replace_node_with_schema(item) for key, item in value.items()
        }
    elif isinstance(value, ProblemGraphNode):
        return value._goos_schema
    return value


@dataclasses.dataclass
class NodeFlags:
    """Contains parameters useful for function and gradient evaluation.

    This dataclass provides additional information useful for function and
    gradient evaluation during backprop. They are passed as arguments to `eval`
    and `grad` by the graph execturo. Using this additional information is
    not necessary but can lead to performance enhancements.

    Attributes:
        const_flags: List of constant flags for each input argument. The flag
            indicates whether the argument value is constant, i.e. cannot
            possibly change.
        frozen_flags: List of constant flags for each input argument.
            The flag indicates whether the value of the input is frozen, i.e.
            the derivative with respect to the node is zero.
    """
    const_flags: flows.Flow.ConstFlags
    frozen_flags: flows.Flow.ConstFlags


@dataclasses.dataclass
class EvalContext:
    input_flags: List[NodeFlags]


class ProblemGraphNode(metaclass=ProblemGraphNodeMeta):
    node_type = "goos.problem_graph_node"
    Schema = optplan.ProblemGraphNodeSchema
    goos_no_register = False

    def __init__(self,
                 deps: Iterable["ProblemGraphNode"] = None,
                 heavy_compute: bool = False) -> None:
        if not deps:
            deps = []
        if isinstance(deps, ProblemGraphNode):
            deps = [deps]
        self._goos_inputs = deps
        self._goos_schema = None
        self._goos_heavy = heavy_compute

    def parallelize(self, val: bool = True):
        self._goos_heavy = val

    def get(self, run: bool = False):
        if run:
            get_default_plan().run()
        return get_default_plan().eval_node(self)

    def get_grad(self, wrt_nodes: List["ProblemGraphNode"], run: bool = False):
        if run:
            get_default_plan().run()
        return get_default_plan().eval_grad(self, wrt_nodes)

    def eval(self, inputs: Iterable["ProblemGraphNode"],
             eval_context: EvalContext):
        raise NotImplemented()

    def __hash__(self):
        return id(self)


class Action(ProblemGraphNode):
    node_type = "goos.action"
    Schema = optplan.ActionNodeSchema


# Decorator to mark methods in `OptimizationPlan` that modify state.
def _requires_action(method):

    def decorator(self, *args, **kwargs):
        if not self._modifiable:
            raise ValueError("Attempted to modify plan outside of an action")
        return method(self, *args, **kwargs)

    return decorator


class OptimizationPlan:

    def __init__(self,
                 root_path: str = ".",
                 save_path: Optional[str] = None,
                 autorun: bool = False) -> None:
        """Creates a new optimization plan.

        Args:
            root_path: Path relative to which any files will be loaded.
            save_path: Path for saving any data. If `None`, logging data will
                not be saved.
            autorun: If `True`, the optimization plan will run immediately
                when an action is added.
        """
        # Flag indicating whether the optimization plan state can be modified.
        # This flag is enabled only if `run` is being called, thus enabling
        # only actions to modify state.
        self._modifiable = False

        self._root_path = root_path
        self._autorun = autorun

        # Handle logging.
        self._save_path = save_path
        self._log_counter = 0

        # Logging to screen.
        self._logger = logging.getLogger(__name__)

        self._node_map = {}
        self._actions = []
        # Index of next action to take.
        self._action_ptr = 0
        # Used to store current event data for resuming.
        self._action_data = None

        # TODO(logansu): Refactor into a single variable state dataclass.
        # Map from variable name to current state.
        self._var_value = {}
        # Map from variable name to whether it is frozen.
        self._var_frozen = {}
        # Map from variable name to lower and upper bounds. Bounds are stored
        # as a tuple `(lower, upper)` where `lower` is an array for lower bounds
        # and `upper` is an array for upper bounds.
        self._var_bounds = {}

    def save(self, save_folder: Optional[str] = None) -> None:
        """Saves the optimization plan.

        Args:
            save_folder: Folder in which to save the optimization plan.
                If `None`, the optimization plan save path is used.
        """
        if not save_folder:
            save_folder = self._save_path

        nodes = [item._goos_schema for item in self._node_map.values()]
        actions = [item._goos_schema for item in self._actions]
        plan = optplan.OptimizationPlanSchema(nodes=nodes, actions=actions)
        with open(os.path.join(save_folder, "optplan.json"), "w") as fp:
            fp.write(optplan.schema.dumps(plan))

    def load(self, save_folder: Optional[str] = None) -> None:
        """Loads the optimization plan.

        Args:
            save_folder: Folder in which to save the optimization plan.
                If `None`, the optimization plan save path is used.
        """
        if not save_folder:
            save_folder = self._save_path

        with open(os.path.join(save_folder, "optplan.json"), "r") as fp:
            plan = optplan.schema.loads(fp.read())

        from schematics import models

        def process_field(model: models.Model,
                          child_model: models.Model) -> ProblemGraphNode:
            return _create_node(child_model)

        visited = set()

        def _create_node(model: models.Model) -> ProblemGraphNode:
            if model.name in self._node_map:
                return self._node_map[model.name]

            optplan.schema._iter_optplan_fields(model, visited, process_field)
            params = {}
            for key, value in model.items():
                if key == "type":
                    continue
                params[key] = value
            return optplan.GLOBAL_CONTEXT_STACK.get_node(
                model.type).creator(**params)

        for node in plan.nodes:
            # TODO(logansu): Check for clashes in names.
            self.add_node(_create_node(node))

        for model in plan.actions:
            # TODO(logansu): Check for clashes in names.
            self.add_action(_create_node(model))

    def get_state_dict(self) -> Dict:
        """Creates a dictionary with the current state of the plan."""
        # Setup variable data.
        var_data = {}
        for var_name, var_val in self._var_value.items():
            var_data[var_name] = {
                "value": var_val,
                "frozen": self._var_frozen[var_name],
                "bounds": self._var_bounds[var_name],
            }

        data = {
            "version": "0.2.0",
            "action_ptr": self._action_ptr,
            "action": self._actions[self._action_ptr]._goos_name,
            "time": str(datetime.datetime.now()),
            "variable_data": var_data,
        }
        # TODO(logansu): Remove "transformation" and "parametrization".
        # For backwards capatibility, keep the name "transformation" and
        # add an empty parametrization.
        data["transformation"] = data["action"]
        data["parametrization"] = {}
        return data

    @property
    def logger(self):
        return self._logger

    @_requires_action
    def write_event(self,
                    event: Dict,
                    monitor_list: List[ProblemGraphNode] = None) -> None:
        """Write monitor data to log file.

        This function is intended to be used within actions only. If the
        save path is not set, this function still performs the same steps
        but will not actually save the log file at the end.

        Args:
            event: Action-specification information about the event.
            monitor_list: List of nodes to save.
        """
        self._log_counter += 1

        self._logger.info(
            "Evaluating monitors for action %d (%s) with event info %s [%d].",
            self._action_ptr, self._actions[self._action_ptr]._goos_name, event,
            self._log_counter)

        # Get the monitor data.
        # TODO(logansu): Make flows serializable.
        monitor_data = {}
        if monitor_list:
            monitor_values = self.eval_nodes(monitor_list)
            for mon, val in zip(monitor_list, monitor_values):
                monitor_data[mon._goos_name] = val.array

        data = self.get_state_dict()
        data.update({
            "event": event,
            "monitor_data": monitor_data,
            "log_counter": self._log_counter,
        })
        self._action_data = event

        # Print out scalar monitor data.
        for mon_name, mon_val in monitor_data.items():
            mon_val = np.atleast_1d(mon_val)
            if mon_val.ndim == 1:
                mon_val = mon_val[0]
                if np.isreal(mon_val):
                    self._logger.info("Monitor {}: {}".format(
                        mon_name, mon_val))
                else:
                    self._logger.info(
                        "Monitor {}: {} (mag={}, phase={})".format(
                            mon_name, mon_val, np.abs(mon_val),
                            np.angle(mon_val)))

        # Save the data.
        if self._save_path:
            file_path = os.path.join(
                self._save_path,
                os.path.join("step{}.pkl".format(self._log_counter)))
            with open(file_path, "wb") as handle:
                pickle.dump(data, handle)
            self._logger.info("Data saved to %s", file_path)

    def write_checkpoint(self, filename: str) -> None:
        """Writes a checkpoint file.

        A checkpoint file contains the full state, including all the
        parametrizations and all the parameters. This is used to restore
        the full state of the optimization.

        Args:
            filename: Name of the checkpoint file.
        """
        data = self.get_state_dict()

        checkpoint_file = filename
        self._logger.info("Saving checkpoint file: %s", checkpoint_file)

        # Save the data.
        with open(checkpoint_file, "wb") as handle:
            pickle.dump(data, handle)

    def read_checkpoint(self, filename: str) -> None:
        """Reads a chekcpoint file.

        This file loads data from a checkpoint file into the optimization plan.
        See `write_checkpoint` for details.

        Args:
            filename: Name of the checkpoint file.
        """
        with open(filename, "rb") as handle:
            data = pickle.load(handle)

        self._action_ptr = data["action_ptr"]
        self._action_data = data.get("event", None)

        # Unpack variables.
        for var_name, var_data in data["variable_data"].items():
            self._var_value[var_name] = var_data["value"]
            self._var_frozen[var_name] = var_data["frozen"]
            self._var_bounds[var_name] = var_data["bounds"]

    def add_action(self, action: Action):
        self._actions.append(action)

        for dep_node in action._goos_inputs:
            self.add_node(dep_node)

        if self._autorun:
            self.run()

    # TODO(logansu): Refactor into a single variable state dataclass.
    def add_node(self, node: ProblemGraphNode):
        if node._goos_name in self._node_map:
            return
        self._node_map[node._goos_name] = node

        from spins.goos import math
        if isinstance(node, math.Variable):
            self._var_value[node._goos_name] = node._value
            self._var_frozen[node._goos_name] = node._is_param

            # Make `lower_bounds` array have same size as node value.
            lower_bounds = node._lower_bounds
            if lower_bounds is None:
                lower_bounds = -np.inf
            if np.isscalar(lower_bounds):
                lower_bounds = np.ones_like(node._value) * lower_bounds

            upper_bounds = node._upper_bounds
            if upper_bounds is None:
                upper_bounds = np.inf
            if np.isscalar(upper_bounds):
                upper_bounds = np.ones_like(node._value) * upper_bounds

            self._var_bounds[node._goos_name] = [lower_bounds, upper_bounds]

        for dep_node in node._goos_inputs:
            self.add_node(dep_node)

    def get_node(self, name: str) -> ProblemGraphNode:
        return self._node_map[name]

    @_requires_action
    def set_var_value(self,
                      node: "goos.Variable",
                      value: np.ndarray,
                      check_frozen: bool = True) -> None:
        if check_frozen and self._var_frozen[node._goos_name]:
            raise ValueError(
                "Attempting to set value of frozen variable {}".format(
                    node._goos_name))
        self._var_value[node._goos_name] = np.array(value)

    @_requires_action
    def set_var_bounds(self, node: "goos.Variable",
                       bounds: List[np.ndarray]) -> List[np.ndarray]:
        self._var_bounds[node._goos_name] = copy.copy(bounds)

    def get_var_value(self, node: "goos.Variable") -> np.ndarray:
        return self._var_value[node._goos_name].copy()

    def get_var_bounds(self, node: "goos.Variable") -> List[np.ndarray]:
        return self._var_bounds[node._goos_name].copy()

    @_requires_action
    def freeze_var(self, node: "goos.Variable") -> None:
        self._var_frozen[node._goos_name] = True

    @_requires_action
    def thaw_var(self, node: "goos.Variable") -> None:
        self._var_frozen[node._goos_name] = False

    @_requires_action
    def get_thawed_vars(self) -> List["goos.Variable"]:
        return [
            self._node_map[var_name]
            for var_name in self._var_value.keys()
            if not self._var_frozen[var_name]
        ]

    def run(self, auto_checkpoint: bool = True) -> None:
        """Runs the optimization plan.

        Args:
            auto_checkpoint: If `True`, checkpoint files are automatically
                generated. Checkpoints will only be created if a save path
                is specified.
        """
        self._modifiable = True
        while self._action_ptr < len(self._actions):
            self.logger.info("Running action {} ({}).".format(
                self._action_ptr, self._actions[self._action_ptr]._goos_name))
            self._actions[self._action_ptr].run(self)

            if auto_checkpoint and self._save_path:
                self.write_checkpoint(
                    os.path.join(self._save_path,
                                 "action{}.chkpt".format(self._action_ptr)))

            self._action_ptr += 1
            self._action_data = None
        self._modifiable = False

    def resume(self,
               checkpoint: str = None,
               auto_checkpoint: bool = True) -> None:
        """Resumes the optimization.

        This function will resume an optimization. It is assumed that the
        optimization plan from before has already been loaded using `load`.
        The function will resume executation based on the state in the
        last loaded checkpoint file (a call to `read_checkpoint`). Optionally,
        `read_checkpoint` will be called before resuming if the `checkpoint`
        parameter is specified.

        Actions must know how to handle a resume by implementing a `resume`
        function. Actions that do not implement such a function will simply
        be executed from the checkpoint file state.

        Args:
            checkpoint: Checkpoint file to load before resuming.
            auto_checkpoint: If `True`, checkpoint files are automatically
                generated. Checkpoints will only be created if a save path
                is specified.
        """
        if checkpoint:
            self.read_checkpoint(checkpoint)

        self._modifiable = True

        # Attempt to resume if there is event data to resume from.
        if self._action_data:
            self.logger.info("Resuming action {} ({}).".format(
                self._action_ptr, self._actions[self._action_ptr]._goos_name))

            if hasattr(self._actions[self._action_ptr], "resume"):
                self._actions[self._action_ptr].resume(self, self._action_data)
            else:
                self.logger.warning(
                    "Action has no resume capabilities {}, re-running.".format(
                        self._action_ptr))
                self._actions[self._action_ptr].run(self)
        else:
            self.logger.info("Running action {} ({}).".format(
                self._action_ptr, self._actions[self._action_ptr]._goos_name))
            self._actions[self._action_ptr].run(self)

        if auto_checkpoint and self._save_path:
            self.write_checkpoint(
                os.path.join(self._save_path,
                             "action{}.chkpt".format(self._action_ptr)))

        self._action_ptr += 1
        self._action_data = None
        self._modifiable = False

        self.run()

    def eval_nodes(self, nodes: List[ProblemGraphNode]) -> List[flows.Flow]:
        """Evaluates nodes.

        If the node is not already in the optplan, it is added.

        Args:
            nodes: List of nodes to evaluate.

        Returns:
            List of flows, one for each node.
        """
        for node in nodes:
            self.add_node(node)

        from spins.goos import graph_executor
        from spins.goos import flows
        override_map = {}
        for var_name, var_value in self._var_value.items():
            # Setup the context.
            const_flags = flows.NumericFlow.ConstFlags()

            frozen_flags = flows.NumericFlow.ConstFlags(False)
            frozen_flags.set_all(self._var_frozen[var_name])

            context = NodeFlags(const_flags=const_flags,
                                frozen_flags=frozen_flags)

            override_map[self._node_map[var_name]] = (
                flows.NumericFlow(var_value), context)
        return graph_executor.eval_fun(nodes, override_map)

    def eval_node(self, node: ProblemGraphNode) -> flows.Flow:
        """Evalutes a single node.

        The node is added to the optplan if it has not already been added.

        Args:
            node: Node to evaluate.

        Returns:
            Flow corresponding to the node.
        """
        return self.eval_nodes([node])[0]

    def eval_grad(self, node: ProblemGraphNode,
                  wrt_nodes: List[ProblemGraphNode]):
        self.add_node(node)
        from spins.goos import graph_executor
        from spins.goos import flows
        override_map = {}
        for var_name, var_value in self._var_value.items():
            # Determine the gradient.
            if self._var_frozen[var_name]:
                grad_value = flows.NumericFlow(np.zeros_like(var_value))
            else:
                grad_value = flows.NumericFlow(np.ones_like(var_value))

            # Setup the context.
            const_flags = flows.NumericFlow.ConstFlags()

            frozen_flags = flows.NumericFlow.ConstFlags(False)
            frozen_flags.set_all(self._var_frozen[var_name])

            context = NodeFlags(const_flags=const_flags,
                                frozen_flags=frozen_flags)

            override_map[self._node_map[var_name]] = (
                flows.NumericFlow(var_value), grad_value, context)
        return graph_executor.eval_grad(node, wrt_nodes, override_map)

    def __enter__(self):
        push_plan(self)
        return self

    def __exit__(self, type, value, traceback):
        pop_plan()


def push_plan(plan: OptimizationPlan) -> None:
    optplan.GLOBAL_PLAN_STACK.append(plan)


def pop_plan() -> OptimizationPlan:
    optplan.GLOBAL_PLAN_STACK.pop()


def get_default_plan() -> OptimizationPlan:
    if optplan.GLOBAL_PLAN_STACK:
        return optplan.GLOBAL_PLAN_STACK[-1]
    return None
