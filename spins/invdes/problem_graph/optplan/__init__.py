"""Defines the optimization plan schema and associated functions.

Note that the order of the imports here is important! This is a package rather
than a single module merely to split up code amongst multiple files to make it
more manageable. It behaves like a single module though.
"""
import enum

# First setup optplan contexts. This is required before any node registration
# and hence we do it first.
from spins.invdes.problem_graph.optplan.context import *  # pylint: disable=wildcard-import

# Global stack context is created. With this, model registration can be defined.
GLOBAL_CONTEXT_STACK = OptplanContextStack()
GLOBAL_CONTEXT_STACK.push(OptplanContext())


class NodeMetaType(enum.Enum):
    """Defines metatypes used by spins.

    Attributes:
        OPTPLAN_NODE: Indicates a `ProblemGraphNode`.
        TRANSFORMATION: Indicates a `Transformation`.
    """
    OPTPLAN_NODE = "optplan_node"
    TRANSFORMATION = "transformation"


def register_node_type(
        node_meta_type: str = NodeMetaType.OPTPLAN_NODE,
        context_stack: OptplanContextStack = GLOBAL_CONTEXT_STACK):
    """Returns a decorator to register a model.

    The model is assumed to be a `schematics.models.Model` with a `StringType`
    field named "type" that contains the node type.

    Args:
        node_meta_type: Metatype of the node model.
        context: Context stack to register model.

    Returns:
        A decorator for a `models.Model` class.
    """

    def decorator(cls):
        # Immediately fail if there is more than one possible choice for field
        # value as this would imply that this is not a proper polymorphic model
        # as we have defined here.

        assert len(cls._schema.fields["type"].choices) == 1  # pylint: disable=protected-access

        node_type = cls._schema.fields["type"].choices[0]  # pylint: disable=protected-access

        def not_implemented(unused_params, unused_workspace):
            raise NotImplementedError(
                "Node type has no creator implemented: {}".format(node_type))

        context_stack.peek().register_node_type(node_meta_type, node_type, cls,
                                                not_implemented)

        return cls

    return decorator


def register_node(model_class: models.ModelMeta,
                  node_meta_type: str = NodeMetaType.OPTPLAN_NODE,
                  context_stack: OptplanContextStack = GLOBAL_CONTEXT_STACK):
    """Returns a decorator to register a model/creator pair.

    This registers a model with the corresponding creator function. The creator
    function should have accept the following arguments:
        *  model: A `schematics.models.Model` with type `model_class`.
        *  workspace: A `spins.invdes.Workspace` workspace.
    The creator function should then construct the object corresponding to the
    node information and return it.

    Args:
        model_class: Class of model to associate with the creator function.
        node_meta_type: Metatype of the node model.
        context: Context stack to register model.

    Returns:
        A decorator for the creator function.
    """
    # Adds the `_claim_polymorphic` method to `model_class`, which allows
    # `schematics` to decide whether a given dictionary of data corresponds
    # to the schema given by `model_class`.
    assert len(model_class._schema.fields["type"].choices) == 1  # pylint: disable=protected-access
    node_type = model_class._schema.fields["type"].choices[0]  # pylint: disable=protected-access

    def _claim_polymorphic(data):
        return data["type"] == node_type

    model_class._claim_polymorphic = _claim_polymorphic  # pylint: disable=protected-access

    # The creator function decorator performs the actual registration.
    def decorator(fun: optplan.CreatorFunction):
        context_stack.peek().register_node_type(node_meta_type, node_type,
                                                model_class, fun)
        return fun

    return decorator


def register_transformation(
        model_class: models.ModelMeta,
        context_stack: OptplanContextStack = GLOBAL_CONTEXT_STACK):
    """Registers a transformation.

    This is a wrapper made just for convenience.

    Args:
        model_class: Class corresponding to the transformation. Note that this
            is NOT the `optplan.Transformation` type.
        context_stack: Context stack to register with.
    """
    return register_node(model_class, NodeMetaType.TRANSFORMATION,
                         context_stack)


# pylint: disable=wildcard-import, wrong-import-position
# Next, import optplan.py, which defines model registration and basic optplan
# types.
from spins.invdes.problem_graph.optplan.optplan import *

# The schemas can be imported arbitrary order now that all the setup work is
# all done/imported.
from spins.invdes.problem_graph.optplan.schema_em import *
from spins.invdes.problem_graph.optplan.schema_function import *
from spins.invdes.problem_graph.optplan.schema_monitor import *
from spins.invdes.problem_graph.optplan.schema_opt import *
from spins.invdes.problem_graph.optplan.schema_param import *

from spins.invdes.problem_graph.optplan.io import *
# pylint: enable=wildcard-import, wrong-import-position

# Use this to allow overriding of the default registrations (which have no
# creator function).
GLOBAL_CONTEXT_STACK.push(OptplanContext())

# Make it easier to setup a schema by importing the required components into
# `optplan` module.
from spins.invdes.problem_graph.schema_utils import polymorphic_model_type as define_schema_type
from schematics import types
