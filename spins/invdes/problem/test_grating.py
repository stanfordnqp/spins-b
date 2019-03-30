import numpy as np
import pytest

from spins.invdes import parametrization
from spins.invdes import problem


@pytest.mark.parametrize("param,min_feature,value", [
    (parametrization.GratingParam([1, 2, 5, 6.7], 10), 1.5,
     [0.5, -1.5, -0.2, 2, -0.3]),
    (parametrization.CompositeParam([
        parametrization.GratingParam([1, 2, 5, 6.7], 10),
        parametrization.GratingParam([3, 4], 8)
    ]), 1.5, [0.5, -1.5, -0.2, 2, -0.3, 0.5, 0, -1]),
])
def test_grating_feature_constraint_objective(param, min_feature, value):
    constraint = problem.GratingFeatureConstraint(min_feature)
    np.testing.assert_almost_equal(
        constraint.calculate_objective_function(param), value)


@pytest.mark.parametrize("param,min_feature,value", [
    (parametrization.GratingParam([1, 2, 5, 6.7], 10), 1.5, [[1, -1, 0, 0], [
        0, 1, -1, 0
    ], [0, 0, 1, -1], [-1, 0, 0, 0], [0, 0, 0, 1]]),
    (parametrization.CompositeParam([
        parametrization.GratingParam([1, 2, 5, 6.7], 10),
        parametrization.GratingParam([3, 4], 8)
    ]), 1.5, [[1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0], [0, 0, 1, -1, 0, 0],
              [-1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, -1],
              [0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 1]]),
])
def test_grating_feature_constraint_gradient(param, min_feature, value):
    constraint = problem.GratingFeatureConstraint(min_feature)
    np.testing.assert_almost_equal(
        constraint.calculate_gradient(param), np.array(value))
