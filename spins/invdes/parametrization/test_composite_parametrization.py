"""Tests for composite_parametrization.py."""
import unittest

import numpy as np
import scipy.sparse as sparse

from spins.invdes.parametrization import composite_parametrization
from spins.invdes.parametrization import parametrization


class QuadraticParam(parametrization.Parametrization):
    def __init__(self, initial, scale):
        self._scale = scale
        self._vector = initial

    def get_structure(self) -> np.ndarray:
        return self._scale * self._vector[:-1]**2

    def calculate_gradient(self) -> sparse.dia_matrix:
        return 2 * self._scale * sparse.diags(
                self._vector[:-1], shape=(self._vector.size - 1,
                                          self._vector.size))

    def encode(self) -> np.ndarray:
        return self._vector

    def decode(self, vector: np.ndarray) -> None:
        self._vector = vector

    def to_vector(self) -> np.ndarray:
        return self._scale * self._vector

    def from_vector(self, vector: np.ndarray) -> None:
        self._vector = self._scale * vector


class TestCompositeParametrization(unittest.TestCase):
    def test_get_structure(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds = [0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 2, 1]),
                                              bounds = [0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2, 1, 1, 5]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        np.testing.assert_array_almost_equal(
                param.get_structure(),
                np.array([1, 2, 1, 4, 2, 2, 1, 1.5, 6.0, 1.5, 1.5]), 7)

    def test_calculate_gradient(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds = [0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 2, 1]),
                                              bounds = [0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2, 1, 1, 5]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        vec = np.array([1, 2, 1, 2, 4, 1, 1, 3, 3, 4, 5, 1])
        np.testing.assert_array_almost_equal(
                param.calculate_gradient() @ vec,
                np.array([1, 2, 1, 2, 4, 1, 1, 9, 18, 12, 15]), 7)

    def test_encode(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds = [0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 2, 1]),
                                              bounds = [0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2, 1, 1, 5]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        np.testing.assert_array_almost_equal(
                param.encode(),
                np.array([1, 2, 1, 4, 2, 2, 1, 1, 2, 1, 1, 5]), 7)

    def test_decode(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds = [0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 2, 1]),
                                              bounds = [0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2, 1, 1, 5]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        param.decode(np.array([1, 3, 1, 1, 4, 1, 3, 1, 5, 1, 1, 6]))
        np.testing.assert_array_almost_equal(
                param_1.encode(), np.array([1, 3, 1, 1]), 7)
        np.testing.assert_array_almost_equal(
                param_2.encode(), np.array([4, 1, 3]), 7)
        np.testing.assert_array_almost_equal(
                param_3.encode(), np.array([1, 5, 1, 1, 6]), 7)

    def test_project(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds = [0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 3.5, 1]),
                                              bounds = [0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2, 1, 1, 5]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        param.project()
        np.testing.assert_array_almost_equal(
                param_1.encode(), np.array([1, 1, 1, 1]), 7)
        np.testing.assert_array_almost_equal(
                param_2.encode(), np.array([2, 2.1, 1]), 7)
        np.testing.assert_array_almost_equal(
                param_3.encode(), np.array([1, 2, 1, 1, 5]), 7)

    def test_bounds(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds=[0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 3.5, 1]),
                                              bounds=[0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2, 1, 1, 5]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        lower_bounds, upper_bounds = param.get_bounds()
        np.testing.assert_array_equal(
                lower_bounds,
                np.array([0, 0, 0, 0, 0.9, 0.9, 0.9,
                          None, None, None, None, None]))
        np.testing.assert_array_equal(
                upper_bounds,
                np.array([1, 1, 1, 1, 2.1, 2.1, 2.1,
                          None, None, None, None, None]))

    def test_to_vector(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds=[0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 3.5, 1]),
                                              bounds=[0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        np.testing.assert_array_almost_equal(
                param.to_vector(),
                np.array([1, 2, 1, 4, 2, 3.5, 1, 1.5, 3]))

    def test_from_vector(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds=[0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 3.5, 1]),
                                              bounds=[0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        param.from_vector(np.array([1, 1, 2, 1, 4, 3, 1, 2, 2]))
        np.testing.assert_array_almost_equal(
                param_1.to_vector(),
                np.array([1, 1, 1, 1]), 7)
        np.testing.assert_array_almost_equal(
                param_2.to_vector(),
                np.array([2.1, 2.1, 1]), 7)
        np.testing.assert_array_almost_equal(
                param_3.to_vector(),
                np.array([4.5, 4.5]), 7)

    def test_serialize(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds=[0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 3.5, 1]),
                                              bounds=[0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        data = param.serialize()
        np.testing.assert_array_almost_equal(
                data["param_0"]["vector"],
                np.array([1, 2, 1, 4]), 7)
        np.testing.assert_array_almost_equal(
                data["param_1"]["vector"],
                np.array([2, 3.5, 1]), 7)
        np.testing.assert_array_almost_equal(
                data["param_2"]["vector"],
                np.array([1.5, 3]), 7)

    def test_deserialize(self):
        param_1 = parametrization.DirectParam(np.array([1, 2, 1, 4]),
                                              bounds=[0, 1])
        param_2 = parametrization.DirectParam(np.array([2, 3.5, 1]),
                                              bounds=[0.9, 2.1])
        param_3 = QuadraticParam(np.array([1, 2]), 1.5)
        param = composite_parametrization.CompositeParam(
                [param_1, param_2, param_3])
        param.deserialize({"param_0": {"vector": np.array([1, 2, 2, 3])},
                           "param_1": {"vector": np.array([1, 2, 5])},
                           "param_2": {"vector": np.array([1, 5])}})
        np.testing.assert_array_almost_equal(
                param_1.encode(), np.array([1, 1, 1, 1]), 7)
        np.testing.assert_array_almost_equal(
                param_2.encode(), np.array([1, 2, 2.1]), 7)
        np.testing.assert_array_almost_equal(
                param_3.encode(), np.array([1.5, 7.5]), 7)
