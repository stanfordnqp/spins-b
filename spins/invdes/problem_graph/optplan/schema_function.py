"""Defines schema for non-EM-related computational graph functions."""
import numbers
from typing import Union

from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils


@optplan.register_node_type()
class Sum(optplan.Function):
    """Defines the sum of a list of functions.

    Attributes:
        type: Must be "function.sum".
    """
    type = schema_utils.polymorphic_model_type("function.sum")
    functions = types.ListType(
        optplan.ReferenceType(optplan.Function), default=[])

    def __add__(self, obj):
        if isinstance(obj, Sum):
            return Sum(functions=self.functions + obj.functions)
        if isinstance(obj, optplan.Function):
            return Sum(functions=self.functions + [obj])
        if isinstance(obj, (numbers.Number, optplan.ComplexNumber)):
            return Sum(functions=self.functions + [make_constant(obj)])
        raise TypeError(
            "Attempting to add a node with type {} to type `Sum`.".format(
                type(obj)))


@optplan.register_node_type()
class Product(optplan.Function):
    """Defines a product function.

    Attributes:
        type: Must be "function.product".
        functions: Functions there are multipied.
    """
    type = schema_utils.polymorphic_model_type("function.product")
    functions = types.ListType(optplan.ReferenceType(optplan.Function))

    def __mul__(self, obj):
        if isinstance(obj, Product):
            return Product(functions=self.functions + obj.functions)
        if isinstance(obj, optplan.Function):
            return Product(functions=self.functions + [obj])
        if isinstance(obj, (numbers.Number, optplan.ComplexNumber)):
            return Product(functions=self.functions + [make_constant(obj)])
        raise TypeError(
            "Attempting to multiply a node with type {} to type `Product`.".
            format(type(obj)))


@optplan.register_node_type()
class Power(optplan.Function):
    """Defines the power of a function.

    Attributes:
        type: Must be "power".
        function: Function to take power of.
        exp: Power.
    """
    type = schema_utils.polymorphic_model_type("function.power")
    function = optplan.ReferenceType(optplan.Function)
    exp = types.FloatType()


@optplan.register_node_type()
class PowerComp(optplan.Function):
    """Defines a penalty function to check if a function is in a certain range.

    penalty = R(f-(value-range/2))**exp+R((value-range/2)-f),
    where R is a ramp function.

    Attributes:
        type: Must be "function.power_comp".
    """
    type = schema_utils.polymorphic_model_type("function.power_comp")
    function = optplan.ReferenceType(optplan.Function)
    value = types.FloatType()
    range = types.FloatType()
    exp = types.FloatType()


@optplan.register_node_type()
class Abs(optplan.Function):
    """Defines absolute value of a function.

    Attributes:
        type: Must be "abs".
        function: The function to take absolute value of.
    """
    type = schema_utils.polymorphic_model_type("function.abs")
    function = optplan.ReferenceType(optplan.Function)

def abs(fun: optplan.Function) -> optplan.Function:
    """Utility function for creating absolute value.

    Args:
        fun: Function to take absolute value.

    Returns:
        `AbsoluteValue` of `fun`.
    """
    return Abs(function=fun)


@optplan.register_node_type()
class Constant(optplan.Function):
    """Defines a constant scalar.

    Attributes:
        type: Must be "function.constant".
        value: The constant value.
    """

    type = schema_utils.polymorphic_model_type("function.constant")
    value = types.ModelType(
        optplan.ComplexNumber, default=optplan.ComplexNumber())


def make_constant(
        value: Union[numbers.Number, optplan.ComplexNumber]) -> Constant:
    """Creates a new constant.

    This is a utility function to quickly create new constants.

    Args:
        value: Value to turn into constant.

    Returns:
        A constant object.
    """
    if isinstance(value, numbers.Number):
        complex_value = complex(value)
        value = optplan.ComplexNumber(
            real=complex_value.real, imag=complex_value.imag)
    return Constant(value=value)


@optplan.register_node_type()
class Parameter(optplan.Function):
    """Defines a constant scalar.

    Attributes:
        type: Must be "function.parameter".
        initial value: Value of the parameters when it is initialized.
    """
    type = schema_utils.polymorphic_model_type("function.parameter")
    initial_value = types.FloatType()
