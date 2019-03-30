"""Defines base optplan types and utility functions."""
import numbers
import warnings

from schematics import models
from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils


class ReferenceType(types.StringType):
    """Represents a top-level object.

    The `schematics` define herein is the serialization schema. During
    serialization, the optplan graph is collapsed into a list of nodes
    and the references from one node to another is replaced by a string (the
    name of the node). During deserialization, the string names are once again
    replaced by their objects. See `io.loads` and `io.dumps` for more details.

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


class OptplanPolyModelType(types.CompoundType):
    """Defines a polymorphic optplan node.

    This class mimics the behavior of `types.PolyModelType` but uses
    the global context stack to dynamically resolve model types rather than
    statically as in `types.PolyModelType`. This enables the `OptimizationPlan`
    schema to be defined before other optplan node schemas to be defined. This
    is necessary to add custom optplan nodes.
    """

    def __init__(self, node_meta_type: str, **kwargs):
        """Creates a new optplan polymorphic model type.

        Args:
            node_meta_type: The metatype of acceptable models.
        """
        self._node_meta_type = node_meta_type

        types.CompoundType.__init__(self, **kwargs)

    def _convert(self, value, context):
        if value is None:
            return None

        model_class = optplan.GLOBAL_CONTEXT_STACK.get_node_model(
            self._node_meta_type, value["type"])

        if model_class is None:
            raise ValueError(
                "Unknown node, got node type '{}' with metatype '{}'".format(
                    value["type"], self._node_meta_type))

        return model_class(value, context=context)

    def _export(self, value, format, context):  # pylint: disable=redefined-builtin
        if (not value.__class__ in optplan.GLOBAL_CONTEXT_STACK.
                get_node_model_dict(self._node_meta_type).values()):
            raise ValueError("Cannot export model with type '{}'".format(
                value.__class__))

        return value.export(context=context)


class ProblemGraphNode(schema_utils.Model):
    """Represents a top-level parameter graph node.

    Attributes:
        name: Name of the node.
    """
    name = types.StringType(required=True)

    def __init__(self, *args, **kwargs) -> None:
        """Creates a new problem graph node.

        Raises:
            ValueError: If `name` is invalid.
        """
        super().__init__(*args, **kwargs)

        # Set a default name if it was not set already.
        if not self.name:
            # TODO(logansu): Remove this recursive import.
            from .io import generate_name
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

                raise ValueError("Expected type {} for field {}, got {}".format(
                    field_type.reference_type, field_name, type(field_value)))


class EmOverlap(ProblemGraphNode):
    """Represents an overlap.

    An overlap object is a callable that accepts as input a
    `SimulationSpaceBase` object and a wavelength and produces as output
    a vector field corresponding to the overlap vector. The overlap vector
    is used to take an inner product on the electric field.
    """


class EmSource(ProblemGraphNode):
    """Represents a source.

    A source object is a callable that accepts as input a
    `SimulationSpaceBase` object and a wavelength and produces as output
    a vector field corresponding to the electric field source. It may optionally
    produce as a second output a Bloch vector corresponding to the source.
    Sources are used for `FdfdSimulation` objects.
    """


class Function(ProblemGraphNode):
    """Represents a node that describes an `spins.invdes.OptimizationFunction`.

    Functions have a well-defined input-output relationship and are
    differentiable. They are used as objective functions and constraints.
    """

    def __add__(self, obj) -> "optplan.Sum":
        if isinstance(obj, optplan.Sum):
            return obj.__add__(self)
        if isinstance(obj, optplan.Constant):
            return optplan.Sum(functions=[self, obj])
        if isinstance(obj, (numbers.Number, optplan.ComplexNumber)):
            return optplan.Sum(functions=[self, optplan.make_constant(obj)])
        if isinstance(obj, Function):
            return optplan.Sum(functions=[self, obj])
        raise TypeError("Attempting to add node with type {}".format(type(obj)))

    def __mul__(self, obj) -> "optplan.Product":
        if isinstance(obj, optplan.Product):
            return obj.__mul__(self)
        if isinstance(obj, optplan.Constant):
            return optplan.Product(functions=[self, obj])
        if isinstance(obj, (numbers.Number, optplan.ComplexNumber)):
            return optplan.Product(functions=[self, optplan.make_constant(obj)])
        if isinstance(obj, Function):
            return optplan.Product(functions=[self, obj])
        raise TypeError("Attempting to multiply node with type {}".format(
            type(obj)))

    def __pow__(self, obj) -> "optplan.Power":
        if isinstance(obj, numbers.Real):
            return optplan.Power(function=self, exp=obj)
        if isinstance(obj, optplan.Constant):
            if obj.value.imag != 0:
                raise ValueError(
                    "Attempting to raise object to a complex power.")
            return optplan.Power(function=self, exp=obj.value.real)
        raise TypeError(
            "Attempting to raise objective to complex or non-constant power.")

    def __sub__(self, obj) -> "optplan.Sum":
        return self + (-obj)

    def __truediv__(self, obj) -> "optplan.Product":
        return self * obj**-1

    def __neg__(self) -> "optplan.Product":
        return -1 * self

    def __radd__(self, obj) -> "optplan.Sum":
        return self.__add__(obj)

    def __rmul__(self, obj) -> "optplan.Product":
        return self.__mul__(obj)

    def __rsub__(self, obj) -> "optplan.Sum":
        return -self + obj

    def __rdiv__(self, obj) -> "optplan.Product":
        return self**(-1) * obj


class Monitor(ProblemGraphNode):
    """Represents a monitor.

    A monitor object implements
    `spins.invdes.problem_graph.creator_monitor.Monitor` object and are used
    to monitor the values of optimization functions during the optimization
    process.
    """


class Parametrization(ProblemGraphNode):
    """Represents a parametrization.

    A parametrization object must implement
    `spins.invdes.parametrization.Parametrization` and represents a
    photonic structcture.
    """


class SimulationSpaceBase(ProblemGraphNode):
    """Represents a simulation space."""
    # TODO(logansu): Define that characteristics of a `SimulationSpaceBase`.


class TransformationBase(schema_utils.Model):
    """Represents a base class of a transformation.

    A transformation object is a callable that accepts a single parameter
    (the parametrization) and executes arbitrary modification on that
    parametrization.
    """


class SetParam(schema_utils.Model):
    """Defines info in how to change a parameter.

    Attributes:
        parameter: Parameter that needs to be set.
        function: Function to evaluate.
        inverse: take the inverse of the function.
    """
    parameter = ReferenceType(Function)
    function = ReferenceType(Function)
    parametrization = ReferenceType(Parametrization)
    inverse = types.BooleanType()


class Transformation(schema_utils.Model):
    """Defines a transformation.

    A transformation are actions that have side effects (whereas a `Function`
    is a pure function that has no side effects). That is, a transformation
    can carry state.

    Attributes:
        name: Name of transformation. Must be unique.
        transformation: Transformation details.
        parameter_list: List of parameters to set before running the
            transformation.
        monitors: List of monitors.
    """
    name = types.StringType()
    parametrization = ReferenceType(Parametrization)
    parameter_list = types.ListType(types.ModelType(SetParam))
    transformation = OptplanPolyModelType(optplan.NodeMetaType.TRANSFORMATION)


class OptimizationPlan(schema_utils.Model):
    """Defines an optimization plan.

    The optimization plan schema defines a series of nodes (e.g. simulation
    spaces, sources, overlaps). Each node must have a unique name amongst all
    nodes. Node names must contain only alphanumeric characters or an underscore
    (_).

    Attributes:
        version: Version information. Should be dot (.) separated.
        nodes: List of problem graph nodes. These include the top-level types
            "simulation_space", "source", "overlap", and "parametrization".
        transformations: List of transformations to apply. Transformations are
            applied in order as given by the list.
    """
    version = types.StringType(required=True, default="0.2.3")
    nodes = types.ListType(
        OptplanPolyModelType(optplan.NodeMetaType.OPTPLAN_NODE))
    transformations = types.ListType(types.ModelType(Transformation))


def vec2d(**kwargs) -> types.ListType:
    """Returns a 2D vector type."""
    return types.ListType(types.FloatType(), min_size=2, max_size=2, **kwargs)


def vec3d(**kwargs) -> types.ListType:
    """Returns a 3D vector type."""
    return types.ListType(types.FloatType(), min_size=3, max_size=3, **kwargs)


class Box3d(schema_utils.Model):
    """Represents an axis-aligned 3D rectangular prism.

    Attributes:
        center: Center of the box.
        extents: Length, width, and height of the box.
    """
    center = vec3d()
    extents = vec3d()


class ComplexNumber(schema_utils.Model):
    """Represents a complex number.

    Attributes:
        real: The real part.
        imag: The imaginary part.
    """
    real = types.FloatType(default=0.0)
    imag = types.FloatType(default=0.0)
