"""Defines schema for monitors."""
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils


@optplan.register_node_type()
class SimpleMonitor(optplan.Monitor):
    """Defines a monitor.

    A monitor takes the output of a scalar function.

    Attributes:
        type: Must be "monitor.scalar".
        function: Name of function to monitor.
    """
    type = schema_utils.polymorphic_model_type("monitor.simple")
    function = optplan.ReferenceType(optplan.Function)


@optplan.register_node_type()
class FieldMonitor(optplan.Monitor):
    """Defines a field monitor.

    A field monitor represents a 3D vector field, e.g. the electric field or
    permittivity distribution. The characteristic of a field monitor is that
    is unpacks the output of a function (which is a vector) into a 3-component,
    3D array, each of which represents a component of the field at a given point
    in space (or in technical terms, the vector is unvec'ed).

    Because 3D fields are expensive to store, a 2D slice can be specified
    by a point in the slice and a normal vector in the slice.

    Attributes:
        type: Must be "monitor.field"
        function: Name of function to monitor.
        center: A point in the monitored slice.
        normal: Normal of the slice plane (must be axis-aligned).
    """
    type = schema_utils.polymorphic_model_type("monitor.field")
    function = optplan.ReferenceType(optplan.Function)
    center = optplan.vec3d()
    normal = optplan.vec3d()
