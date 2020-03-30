from typing import List, Union

import copy
import dataclasses
import numbers

import numpy as np


class FlowMeta(type):
    """Creates a new `Flow` object.

    This metaclass is responsible for the following:
    (1) Turning `Flow` classes into dataclasses. This is merely for usability.
    (2) Automatically generating `ConstFlags` if it does not already exist.
    (3) Automatically generating `Grad` if it does not already exist.

    A `Flow` class must have the following two counterparts:
    (1) `ConstFlags` is a dataclass with flags for every non-constant flow field
        that is used by `NodeFlags` to indicate whether the flags are constant
        or frozen. These are used to optimize function and gradient evaluation.
    (2) `Grad` is a gradient flow object corresponding to the flow class.

    A valid `ConstFlags` must resemble the following:

    ```python

    class ConstFlags:

        def __bool__(self) -> bool:
            # Indicates whether the flow is entirely constant.

        def set_all(self, value: bool) -> None:
            # Sets all the fields to have constantness given by `value`.
    ```
    The default implementation simply makes a field for `ConstFlags` for every
    non-constant field in the flow.

    A valid `Grad` must resemble the following:

    ```python

    # Must inherit from `Flow.Grad`.
    class Grad(flows.Flow.Grad):

        def __iadd__(self, value: Grad) -> Grad:
            # Implements how to sum a version of itself.
            # This is used by backprop.
    ```
    The default implementation makes a field called "XXX_grad" for every
    non-constant field in the flow named "XXX". The default `__iadd__`
    implementation performs `__iadd__` on each of the fields in `Grad`.
    """

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        cls = dataclasses.dataclass(cls, eq=False)

        # Get the fields for inner class autogeneration.
        cls_fields = dataclasses.fields(cls)
        # Keep track of all the fields that can vary.
        nonconst_fields = []
        for field in cls_fields:
            if ("constant_field" in field.metadata) and (
                    field.metadata["constant_field"]):
                continue
            nonconst_fields.append(field)

        # Create constant flag class.
        if "ConstFlags" not in cls.__dict__:
            const_flag_fields = [(field.name, bool,
                                  dataclasses.field(default=False))
                                 for field in nonconst_fields]
            cls.ConstFlags = dataclasses.make_dataclass(name + ".ConstFlags",
                                                        const_flag_fields)

            def __bool__(self) -> bool:
                return all(
                    getattr(self, field.name) for field in nonconst_fields)

            def set_all(self, value: bool) -> None:
                for field in nonconst_fields:
                    setattr(self, field.name, value)

            cls.ConstFlags.__bool__ = __bool__
            cls.ConstFlags.set_all = set_all

        # Create the gradient class.
        if "Grad" not in cls.__dict__:
            grad_fields = [(field.name + "_grad", field.type, np_zero_field(1))
                           for field in nonconst_fields]

            grad_bases = tuple(cls_base.Grad for cls_base in bases)
            cls.Grad = dataclasses.make_dataclass(name + ".Grad",
                                                  grad_fields,
                                                  bases=grad_bases)

            def __iadd__(self, value):
                for field in grad_fields:
                    field_val = getattr(self, field[0])
                    value_val = getattr(value, field[0])
                    try:
                        field_val.__iadd__(value_val)
                    except AttributeError:
                        setattr(self, field[0], value_val + field_val)
                return super(cls.Grad, self).__iadd__(value)

            cls.Grad.__iadd__ = __iadd__

        return cls


def np_zero_field(n: int):
    """Creates a field that defaults to a numpy array with zeros.

    Args:
        n: Number of elements in array.

    Returns:
        Dataclass field that produces an array with `n` zeros.
    """
    return dataclasses.field(default_factory=lambda: np.zeros(n))


def constant_field(**kwargs):
    """Marks a flow field as constant.

    Constant flow fields are not permitted to change value once set, and
    consequently, the gradient for these fields do not exist.

    Args:
        kwargs: Keyword arguments to pass to `dataclasses.field`.

    Returns:
        A dataclasses field where `metadata` has entry `"constant_field": True`.
    """
    if "metadata" not in kwargs:
        kwargs["metadata"] = {}
    kwargs["metadata"].update({"constant_field": True})
    return dataclasses.field(**kwargs)


class Flow(metaclass=FlowMeta):

    class Grad:

        def __iadd__(self, value):
            return self

    def __eq__(self, other: "Flow") -> bool:
        """Checks if two dataclasses are equal to which other.

        We need to implement equality operator separately to handle NumPy
        arrays, which require calling `.all()` to indicate equality.

        Args:
            other: Flow to compare to.

        Returns:
            `True` only if `self` and `other` are the same type and their
            values are equal.

        Raises:
            NotImplemented: If the flow types are different between `self`
                and `other`.
        """
        if self is other:
            return True

        if self.__class__ != other.__class__:
            raise NotImplemented(
                "Cannot compare flow types, got {} and {}".format(self, other))

        for val1, val2 in zip(dataclasses.astuple(self),
                              dataclasses.astuple(other)):

            if val1 is val2:
                equal = True
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                equal = (val1.shape == val2.shape) and (val1 == val2).all()
            else:
                equal = val1 == val2
            if not equal:
                return False
        return True


class NumericFlow(Flow):
    """Represents a numeric value.

    This flow is implemented here because of its special nature (e.g. all
    variables are numeric flows, gradient calculations start with a numeric
    flow).
    """
    array: np.ndarray = 0

    @dataclasses.dataclass
    class Grad(Flow.Grad):
        array_grad: np.ndarray = np_zero_field(1)

        def __iadd__(self, value):
            self.array_grad += value.array_grad
            return super().__iadd__(value)

        def __eq__(self, value) -> bool:
            if type(self) == NumericFlow.Grad:
                if isinstance(value, numbers.Number):
                    return np.all(self.array_grad == value)
                elif isinstance(value, np.ndarray):
                    return np.all(self.array_grad == value)
                elif type(value) == NumericFlow.Grad:
                    return np.all(self.array_grad == value.array_grad)
            return super().__eq__(value)

    def __eq__(self, value) -> bool:
        if type(self) == NumericFlow:
            if isinstance(value, numbers.Number):
                return np.all(self.array == value)
            elif isinstance(value, np.ndarray):
                return np.all(self.array == value)
            elif type(value) == NumericFlow:
                return np.all(self.array == value.array)
        return super().__eq__(value)
