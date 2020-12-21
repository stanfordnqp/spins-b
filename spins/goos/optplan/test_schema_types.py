import numpy as np
import pytest
from schematics import models
from schematics import types

from spins.goos.optplan import schema_types


def test_complex_type():

    class Model(models.Model):
        x = schema_types.ComplexNumberType()

    model = Model({"x": 1 + 2j})
    model.validate()
    assert model.to_primitive() == {"x": [1, 2]}
    assert model.to_native() == {"x": 1 + 2j}

    model = Model({"x": [1, 2]})
    model.validate()
    assert model.to_primitive() == {"x": [1, 2]}
    assert model.to_native() == {"x": 1 + 2j}

    model = Model({"x": 1})
    model.validate()
    assert model.to_primitive() == {"x": [1, 0]}
    assert model.to_native() == {"x": 1 + 0j}

    with pytest.raises(ValueError, match="not convert to complex"):
        model = Model({"x": "hi"})


def test_numpy_array_type():

    class Model(models.Model):
        x = schema_types.NumpyArrayType()

    model = Model({"x": 1})
    model.validate()
    assert model.to_primitive() == {"x": 1}
    assert model.to_native() == {"x": 1}

    model = Model({"x": [1]})
    model.validate()
    assert model.to_primitive() == {"x": [1]}
    assert model.to_native() == {"x": [1]}

    model = Model({"x": np.array([1, 2])})
    model.validate()
    assert model.to_primitive() == {"x": [1, 2]}
    model_native = model.to_native()
    assert list(model_native.keys()) == ["x"]
    np.testing.assert_array_equal(model_native["x"], [1, 2])

    model = Model({"x": np.array([[1], [2]])})
    model.validate()
    assert model.to_primitive() == {"x": [[1], [2]]}
    model_native = model.to_native()
    assert list(model_native.keys()) == ["x"]
    np.testing.assert_array_equal(model_native["x"], [[1], [2]])

    model = Model({"x": [1 + 2j, 3 + 4j]})
    model.validate()
    assert model.to_primitive() == {"x": { "real": [1, 3], "imag": [2, 4]}}
    model_native = model.to_native()
    assert list(model_native.keys()) == ["x"]
    np.testing.assert_array_equal(model_native["x"], [1 + 2j, 3 + 4j])
