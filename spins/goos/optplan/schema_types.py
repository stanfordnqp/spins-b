"""This module defines additional `schematics` types.

These types are considered primitives in SPINS.
"""
import numbers

import numpy as np
from schematics import models
from schematics import types


class ComplexNumberType(types.BaseType):
    """Defines a `schematics` type for complex numbers.

    Complex numbers `a+bj` are serialized into a list of two numbers `[a, b]`.
    """

    def to_native(self, value, context=None):
        if isinstance(value, numbers.Number):
            return value
        elif isinstance(value, list):
            if len(value) != 2:
                raise ValueError(
                    "Complex number primitive form must be a list of two"
                    " elements, got {}".format(value))
            return value[0] + 1j * value[1]
        raise ValueError(
            "Could not convert to complex number, got {}".format(value))

    def to_primitive(self, value, context=None):
        if not isinstance(value, complex):
            value = value + 0j
        return [value.real, value.imag]


class NumpyArrayType(types.BaseType):
    """Defines a `schematics` type for numpy arrays.

    Numpy arrays are serialized into list form (i.e. `np.ndarray.tolist`).
    Complex arrays are not supported at this time.
    """

    def to_native(self, value, context=None):
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, numbers.Number):
            return value
        elif isinstance(value, dict):
            return np.array(value["real"]) + 1j * np.array(value["imag"])
        return np.array(value)

    def to_primitive(self, value, context=None):
        value = np.array(value)
        if np.iscomplexobj(value):
            return {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        else:
            return np.array(value).tolist()
