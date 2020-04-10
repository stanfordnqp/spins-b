"""Defines base optplan optplan.types and utility functions."""
import numbers
import warnings

from schematics import models

from spins.goos import optplan


class ReferenceType(optplan.types.StringType):
    """Represents a top-level object.

    The `schematics` define herein is the serialization schema. During
    serialization, the optplan graph is collapsed into a list of nodes
    and the references from one node to another is replaced by a string (the
    name of the node). During deserialization, the string names are once again
    replaced by their objects.

    This reference type helps this process by (1) acting a string object for
    verification by `schematics` and (2) keeping track of the desired target
    type of a node. This typing information is used to validate that the
    graph was constructed properly.
    """

    def __init__(self, model_class: models.ModelMeta, **kwargs) -> None:
        """Creates a new reference type.

        Args:
            model_class: This reference should refer to an object with type
                `model_class`.
        """
        self._reference_type = model_class
        super().__init__(**kwargs)

    @property
    def reference_type(self) -> models.ModelMeta:
        return self._reference_type


class NodeSchema(optplan.Model):
    """Represents a top-level parameter graph node.

    Attributes:
        name: Name of the node.
    """
    name = optplan.types.StringType(required=True)

    def __init__(self, *args, **kwargs) -> None:
        """Creates a new problem graph node.

        Raises:
            ValueError: If `name` is invalid.
        """
        super().__init__(*args, **kwargs)

        # Set a default name if it was not set already.
        if not self.name:
            # TODO(logansu): Remove this recursive import.
            from .schema import generate_name
            self.name = generate_name(self.type)  # pylint: disable=no-member

        # Validate the name.
        if self.name.startswith("__"):
            raise ValueError(
                "Name cannot start with two underscores (__), got {}".format(
                    self.name))

        # Verify that reference fields have been appropriately set. This is
        # actually a redundant check since `optplan.loads` and `optplan.dumps`
        # performs the check as well. However, erroring out as soon as possible
        # makes debugging easier. Moreover, this is check is not strict enough
        # because it only checks the immediate fields (does not recurse on
        # lists, dictionaries, and models).
        # TODO(logansu): Remove warnings once `schematics` has fixed its own
        # warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Iterate through every field and, if it is a reference field,
            # check that the value is of the right type. Again, we stress that
            # this is not the definitive error checker and is here purely
            # to aid debugging. Consequently, this is not a thorough check.
            for field_name, field_type in self.fields.items():
                if not isinstance(field_type, ReferenceType):
                    continue

                field_value = self[field_name]
                if field_value is None:
                    continue
                elif isinstance(field_value, (str, field_type.reference_type)):
                    continue
                elif isinstance(field_value, optplan.ProblemGraphNode):
                    if issubclass(field_value.Schema,
                                  field_type.reference_type):
                        continue
                elif isinstance(field_value, optplan.WildcardSchema):
                    continue

                raise ValueError("Expected type {} for field {}, got {}".format(
                    field_type.reference_type, field_name, type(field_value)))


class ProblemGraphNodeSchema(NodeSchema):
    """Represents a `ProblemGraphNode`."""


class ActionNodeSchema(NodeSchema):
    """Represents a `Action`."""


class WildcardSchema:
    """Indicates that the schema is a wildcard and is valid for all references.

    Any schema that inherits this will bypass `ReferenceField` validation. This
    is used by `CastSchema` to declare itself as a valid schema for all
    `ReferenceField`.
    """


class OptplanPolyModelType(optplan.types.CompoundType):
    """Defines a polymorphic optplan node.

    This class mimics the behavior of `optplan.types.PolyModelType` but uses
    the global context stack to dynamically resolve model types rather than
    statically as in `optplan.types.PolyModelType`. This enables the
    `OptimizationPlanSchema` to be defined before other optplan node schemas
    to be defined. This is necessary to add custom optplan nodes.
    """

    def __init__(self, node_meta_type: str, **kwargs):
        """Creates a new optplan polymorphic model type.

        Args:
            node_meta_type: The metatype of acceptable models.
        """
        self._node_meta_type = node_meta_type

        optplan.types.CompoundType.__init__(self, **kwargs)

    def _convert(self, value, context):
        if value is None:
            return None

        model_class = optplan.GLOBAL_CONTEXT_STACK.get_node(
            value["type"]).schema

        if not issubclass(model_class, self._node_meta_type):
            raise ValueError(
                "Unknown node, got node type '{}' with metatype '{}'".format(
                    value["type"], self._node_meta_type))

        return model_class(value, context=context)

    def _export(self, value, format, context):  # pylint: disable=redefined-builtin
        # TODO: Understand _export
        #if (not value.__class__ in optplan.GLOBAL_CONTEXT_STACK.get_node_map().
        #        values()):
        #    raise ValueError("Cannot export model with type '{}'".format(
        #        value.__class__))

        return value.export(context=context)


class OptimizationPlanSchema(optplan.Model):
    """Defines an optimization plan.

    The optimization plan schema defines a series of nodes (e.g. simulation
    spaces, sources, overlaps). Each node must have a unique name amongst all
    nodes. Node names must contain only alphanumeric characters or an underscore
    (_).

    Attributes:
        version: Version information. Should be dot (.) separated.
        nodes: List of problem graph nodes.
        actions: List of actions to apply.
    """
    version = optplan.types.StringType(required=True, default="0.3.0")
    nodes = optplan.types.ListType(OptplanPolyModelType(ProblemGraphNodeSchema))
    actions = optplan.types.ListType(OptplanPolyModelType(ActionNodeSchema),
                                     default=[])
