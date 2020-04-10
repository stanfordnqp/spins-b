"""Module responsible for parsing schemas.

This module is responsible for parsing and generating schemas.
"""
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import copy
import inspect
import json
import numbers
import sys
import typing
import warnings

import numpy as np
from schematics import models
from schematics import types

# `typing_inspect` officially only works with Python 3.7+.
# We can kind of get away with using 3.6.
if sys.version_info[1] >= 6:
    import typing_inspect
else:
    # Create a fake module.
    import imp
    typing_inspect = imp.new_module("typing_inspect")

    def is_union_type(cls):
        # Our codebase does not support anything below Python 3.5.

        # Python 3.5
        if hasattr(cls, "__union_params__"):
            return True

        # Python 3.6
        if not hasattr(cls, "__origin__"):
            return False
        if cls.__origin__ is Union:
            return True
        return False

    typing_inspect.is_union_type = is_union_type


def get_type_class(cls):
    try:
        # Python 3.5 and 3.6
        return cls.__extra__
    except AttributeError:
        # Python 3.7+
        return cls.__origin__


typing_inspect.get_type_class = get_type_class

from spins.goos import optplan
from spins.goos.optplan import schema_types

# Backport `typing.get_args` from Python 3.8
if not hasattr(typing, "get_args"):

    def get_args(cls) -> Tuple:
        if typing_inspect.is_union_type(cls):
            try:
                return cls.__union_params__
            except AttributeError:
                pass
            return cls.__args__
        elif issubclass(cls, List):
            return cls.__args__
        else:
            raise ValueError("Cannot get type arguments for {}".format(cls))

    typing.get_args = get_args


def construct_schema(schema_name: str,
                     fun: Callable,
                     skip_first_arg: bool = False,
                     base_classes=(models.Model,),
                     other_fields: Dict = None) -> models.ModelMeta:
    """Constructs a `schematics.models.Model` type based on a function.

    This is used to automatically generate schemas from functions with type
    annotations. We use reflection to determine the names of the variables, type
    annotations to create schema types, and the default values.

    The function must obey the following rules:
    - All parameters must be type-annotated with an acceptable type. Acceptable
      types include the following (an exhaustive list can be found in the
      implementation of `_convert_type_to_schematics`):
      - ints
      - floats
      - strings
      - booleans
      - numpy arrays
      - problem graph nodes (`ProblemGraphNode`)
      - lists
      - unions (`Union`)
      - other schematics models inheriting `optplan.Model`
    - Schematics model types will be typed as `PolyModelType` so the same
      precautions with using `PolyModelType` must be followed.
    - Variable length parameters are not permitted (i.e. *args, **kwargs).
    - Default values must match the type annotation.

    Example:

        def fun(x: int, y: float = 3):
            ...
        FunSchema = construct_schema("FunSchema", fun)

    is equivalent to

        class FunSchema(models.Model):
            x = types.IntType()
            y = types.FloatType(default=3)

    Args:
        schema_name: Name to give the schema.
        fun: Function from which to generate schema.
        skip_first_arg: If `True`, the generated schema ignores the first
            function argument. This is useful when generating schemas from
            member functions, which always take `self` as the first argument.
        base_classes: A tuple of base classes from which the schema inherits.
            At least one of the base classes must be a subclass of
            `schematics.models.Model`.
        other_fields: A dictionary mapping from string to schematic type
            objects. These fields are included in the generated schema type.
            Note that they will override any parameters with the same name.

    Returns:
        A `schematics` model with name `schema_name` that inherits from
        `base_classes` and has a schema according to the parameters of `fun`.
    """
    signature = inspect.signature(fun)
    param_list = list(signature.parameters.values())
    if skip_first_arg:
        param_list = param_list[1:]

    model_args = {}
    for param in param_list:
        model_args[param.name] = _convert_param_to_schematics(param)

    if other_fields:
        model_args.update(other_fields)
    return type(schema_name, tuple(base_classes), model_args)


def _convert_type_to_schematics(typ) -> Tuple:
    """Converts a type object into its equivalent `schematics` schema type.

    Note that this function returns a tuple `(typ, args)` so construct the
    actual type object for `schematics`, one must pass `args` to the constructor
    i.e. `typ(*args)`. This is done so that the caller of the function may
    choose to add additional arguments before constructing the type.

    Example:
        _convert_type_to_schematics(int) == (types.IntType, [])
        _convert_type_to_schematics(List[int]) == (types.ListType,
            [types.IntType()])

    Args:
        typ: Python type from a type annotation.

    Returns:
        A tuple `(typ, args)` where `typ` is a `schematics` type (inherits from
        `types.TypeMeta`) and `args` is a list of arguments that should be
        passed into `typ` to construct the type (see example).
    """
    schema_type = None
    schema_args = []
    # TODO(logansu): Add support for tuples and dictionaries.
    if typ == int:
        schema_type = types.IntType
    elif typ == float:
        schema_type = types.FloatType
    elif typ == str:
        schema_type = types.StringType
    elif typ == bool:
        schema_type = types.BooleanType
    elif typ == complex:
        schema_type = schema_types.ComplexNumberType
    elif typ == np.ndarray:
        schema_type = schema_types.NumpyArrayType
    elif typing_inspect.is_union_type(typ):
        schema_type = types.UnionType
        union_types = []
        for arg in typing_inspect.get_args(typ, evaluate=True):
            arg_typ, arg_args = _convert_type_to_schematics(arg)
            union_types.append(arg_typ(*arg_args))
        schema_args.append(union_types)
    elif typing_inspect.is_generic_type(typ):
        if issubclass(typing_inspect.get_type_class(typ), list):
            schema_type = types.ListType
            for arg in typing_inspect.get_args(typ, evaluate=True):
                arg_typ, arg_args = _convert_type_to_schematics(arg)
                schema_args.append(arg_typ(*arg_args))
        else:
            raise ValueError(
                "Cannot convert type to schema type, got {}".format(typ))
    elif issubclass(typ, optplan.ProblemGraphNode):
        schema_type = optplan.ReferenceType
        schema_args = [typ.Schema]
    elif issubclass(typ, optplan.Model):
        schema_type = types.PolyModelType
        schema_args = [typ]
    elif issubclass(typ, models.Model):
        # TODO(logansu): Remove this block once old spins fully deprecated.
        schema_type = types.PolyModelType
        schema_args = [typ]
    else:
        raise ValueError(
            "Cannot convert type to schema type, got {}".format(typ))
    return schema_type, schema_args


def _convert_param_to_schematics(param):
    """Converts a parameter into schematics schema.

    This function does the following:
    1) Converts the Python type into a schematics type.
    2) Sets any default values.
    3) Assumes any parameter without a default value is a required field.

    Args:
        param: A parameter object from `inspect.signature`.

    Returns:
        A `schematics` type object.
    """
    # Store the class, position arguments, and keyword arguments used to
    # construct the type.
    schema_type, schema_args = _convert_type_to_schematics(param.annotation)
    schema_kwargs = {}

    # Next determine the attributes of the type.
    if param.default != param.empty:
        schema_kwargs["default"] = param.default
    else:
        schema_kwargs["required"] = True

    return schema_type(*schema_args, **schema_kwargs)


def generate_name(model_type: str) -> str:
    """Generates a name for a model with type `model_type`.

    This is used by auto-name generation. The name will be of the form
    `model_type.num` where `num` is a number that is tracked by
    `problem_graph_name_map`. Note that this does NOT guarantee name uniqueness
    unless the following are true:
        1) All nodes created before the most recent call to `reset_graph` are
           not present in the current graph (since `reset_graph` resets the
           counters).
        2) All user-defined names do not carrying the period (.).

    Args:
        model_type: String model type.

    Returns:
        A unique automatically generated name subject the constraints stated
        above.
    """
    if model_type not in optplan.PROBLEM_GRAPH_NAME_MAP:
        optplan.PROBLEM_GRAPH_NAME_MAP[model_type] = 0

    name = "{}.{}".format(model_type,
                          optplan.PROBLEM_GRAPH_NAME_MAP[model_type])

    optplan.PROBLEM_GRAPH_NAME_MAP[model_type] += 1

    return name


def _iter_optplan_fields(
        model: models.Model,
        visited: Set[int],
        process_field: Callable[
            [models.Model, Union[str, optplan.ProblemGraphNodeSchema]], None],
        pass_field_info: bool = False) -> None:
    """Iterate through every reference field recursively in the model.

    This function operates by performing a depth-first search through every
    model and calling `process_field` on every field in the model. This is
    useful, for example, for finding all the reference nodes and replacing them
    with their names (see `_replace_ref_nodes_with_names`).

    Args:
        model: The model to recrusively search.
        visited: A list of model ids (computed from `id`) to indicate which
            nodes have been visited before.
        process_field: Function to call on every reference field. This function
            should accept the following arguments:
                - A `models.Model` model corresponding to the model with the
                  field.
                - The model (or name of the model) of the reference field.
            The function should return an object that should replace the
            reference field with. If it returns `None`, the optplan will be
            left intact.
        pass_field_info: If `True`, the field information is passed as a third
            argument to `process_field`.
    """
    # Handle only dictionaries, lists, and models. Every other type is assumed
    # to not contain a `optplan.ReferenceType`.
    if not isinstance(model, models.Model):
        return

    # Avoid double processing nodes.
    if id(model) in visited:
        return
    visited.add(id(model))

    # Wrap `process_field` so that returning `None` is same as returning the
    # child.
    def process_field_wrapped(
        parent: models.Model,
        child: Union[str, optplan.ProblemGraphNodeSchema],
        field_type: optplan.ReferenceType,
    ) -> optplan.ProblemGraphNodeSchema:
        if pass_field_info:
            return_val = process_field(parent, child, field_type)
        else:
            return_val = process_field(parent, child)

        if return_val is None:
            return child
        return return_val

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for field_name, field_type in model.fields.items():
            if model[field_name] is None:
                continue
            # Recursively check every field for `optplan.ReferenceType` in
            # compound types.
            if isinstance(model[field_name], models.Model):
                _iter_optplan_fields(model[field_name],
                                     visited,
                                     process_field,
                                     pass_field_info=pass_field_info)
            elif isinstance(field_type, types.ListType):
                for item in model[field_name]:
                    _iter_optplan_fields(item,
                                         visited,
                                         process_field,
                                         pass_field_info=pass_field_info)
            elif isinstance(field_type, types.DictType):
                for _, value in model[field_name].items():
                    _iter_optplan_fields(value,
                                         visited,
                                         process_field,
                                         pass_field_info=pass_field_info)

            # Actually process any reference fields.
            if isinstance(field_type, optplan.ReferenceType):
                model[field_name] = process_field_wrapped(
                    model, model[field_name], field_type)
            elif (isinstance(field_type, types.ListType) and
                  isinstance(field_type.field, optplan.ReferenceType)):
                model[field_name] = [
                    process_field_wrapped(model, m, field_type.field)
                    for m in model[field_name]
                ]
            elif (isinstance(field_type, types.DictType) and
                  isinstance(field_type.field, optplan.ReferenceType)):
                model[field_name] = {
                    key: process_field_wrapped(model, m, field_type.field)
                    for key, m in model[field_name].items()
                }


def validate(plan: optplan.OptimizationPlanSchema) -> None:
    """Validates an optimization plan.

    Args:
        plan: Plan to validate.

    Raises:
        Exception if plan is not valid.
    """
    # Do schema validation.
    # Rather than validate the entire plan all at once, we first validate
    # node-by-node because `schematics` is very bad at giving informative
    # errors.
    node_list = []
    if plan.nodes:
        node_list += plan.nodes
    if plan.actions:
        node_list += plan.actions

    for node in node_list:
        try:
            # Store the node name before hand because `validate` will
            # actually destroy the node if the validation fails.
            node_name = node.name
            node.validate()
        except Exception as exc:
            raise ValueError("Error encountered when validating node {}".format(
                node_name)) from exc

    # Now validate the plan schema itself just in case we missed something
    # from the previous checks.
    plan.validate()

    # Check for non-unique nodes.
    names = set()
    for node in plan.nodes:
        if node.name in names:
            raise ValueError("Nonunique name found: {}".format(node.name))
        names.add(node.name)


def validate_references(model: models.Model) -> None:
    """Validates that all `optplan.ReferenceType` are valid.

    This function recursively checks to see if every `ReferenceType` field
    is actually holding a reference to the appropriate type.

    Args:
        model: The model to check.

    Raises:
        ValueError: If a bad type is encountered.
    """

    def process_field(parent: models.Model,
                      child: Union[str, optplan.ProblemGraphNodeSchema],
                      field_type: optplan.ReferenceType) -> None:
        if not child:
            return

        if (not isinstance(child, (str, field_type.reference_type)) and
                not isinstance(child, optplan.WildcardSchema)):
            raise ValueError("Expected type {} for node {}, got {}.".format(
                field_type.reference_type,
                child,
                type(child),
            ))

    _iter_optplan_fields(model, set(), process_field, pass_field_info=True)


def _validate_optplan_version(version: str) -> None:
    """Validates an optplan has a compatible version.

    Raises:
        ValueError: On incompatible version.
    """
    version_parts = [int(part) for part in version.split(".")]
    if version_parts[0] < 0 or version_parts[1] < 3:
        raise ValueError(
            "Optplan must be at least version 0.3.0, got {}".format(version))


def loads(serialized_plan: str) -> optplan.OptimizationPlanSchema:
    """Loads a serialized optimization plan.

    Args:
        serialized_plan: Serialized plan.

    Returns:
        Stored `optplan.OptimizationPlanSchema`.
    """
    plan = optplan.OptimizationPlanSchema(json.loads(serialized_plan))
    validate(plan)
    _validate_optplan_version(plan.version)

    # Unpack the optimization plan.
    # First create a dictionary mapping strings to the node.
    node_map = {}
    for node in plan.nodes:
        node_map[node.name] = node

    # Now, recursively fill any node names with the actual node.
    def process_field(model: models.Model, child_model: str) -> models.Model:
        return node_map[child_model]

    visited = set()
    if plan.actions:
        for action in plan.actions:
            _iter_optplan_fields(action, visited, process_field)
    for node in plan.nodes:
        _iter_optplan_fields(node, visited, process_field)

    validate_references(plan)
    return plan


def dumps(plan: optplan.OptimizationPlanSchema) -> str:
    """Serializes `plan` into a string.

    This is useful for saving `optplan.OptimizationPlanSchema` as a string.

    Args:
        plan: Plan.

    Raises:
        ValueError: If any `optplan.ProblemGraphNodeSchema` names are duplicated.
    """
    # Make a deepcopy so we do not touch the original.
    plan = copy.deepcopy(plan)
    validate_references(plan)

    # Next extract out every graph node.
    model_list = []
    _extract_nodes(plan, model_list)

    # Now replace all reference types with the correct names.
    _replace_ref_nodes_with_names(plan, model_list)

    # Update the list of nodes.
    # Remove actions from the list.
    plan.nodes = model_list

    validate(plan)
    return json.dumps(plan.to_primitive())


def _extract_nodes(value: Any,
                   model_list: List[optplan.ProblemGraphNodeSchema]) -> None:
    """Finds any nodes recursively in `value` and appends them to `model_list`.

    This is used by `dumps` to find all relevant `optplan.ProblemGraphNodeSchema`
    instances.

    Args:
        value: Value to recurse on.
        model_list: List of `optplan.ProblemGraphNodeSchema` models found so far.
    """
    if isinstance(value, optplan.ProblemGraphNodeSchema):
        if value not in model_list:
            model_list.append(value)
        else:
            # This node has been processed before so quit.
            return

    if isinstance(value, models.Model):
        for item in value.keys():
            _extract_nodes(value[item], model_list)
    elif isinstance(value, list):
        for item in value:
            _extract_nodes(item, model_list)
    elif isinstance(value, dict):
        for _, item in value.items():
            _extract_nodes(item, model_list)


def _replace_ref_nodes_with_names(
        model: models.Model,
        model_list: List[optplan.ProblemGraphNodeSchema]) -> None:
    """Replaces `optplan.ProblemGraphNodeSchema` references with names.

    This is used by `dumps` to replace all `optplan.ProblemGraphNodeSchema` instances
    with their names for serialization.

    Args:
        model: Model to recurse on.
        model_list: List of `optplan.ProblemGraphNodeSchema` models.
    """

    # Replace model reference with a name.
    def process_field(model: models.Model, child_model: models.Model) -> str:
        if isinstance(child_model, str):
            return child_model

        ind = model_list.index(child_model)
        return model_list[ind].name

    _iter_optplan_fields(model, set(), process_field)
