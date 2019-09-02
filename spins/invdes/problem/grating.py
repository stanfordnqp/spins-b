"""This module contains structure objectives for grating parametrizations."""
from typing import List

import numpy as np
import scipy

from spins.invdes import parametrization
from spins.invdes import problem


def _diff_matrix(vec_len: int) -> np.ndarray:
    """Constructs the matrix that represents the operation `np.diff`.

    Args:
        vec_len: Number of elements in vector.

    Returns:
        A matrix `A` such that `A @ v == np.diff(v)` for some vector `v`.
    """
    # Construct a matrix that computes the distances between adjacent
    # vector elements. This is simply a `np.diff`.
    return (scipy.linalg.circulant([-1, 1] + [0] * (vec_len - 2)).T)[:-1, :]


class GratingFeatureConstraint(problem.OptimizationFunction):
    """Handles feature constraints for `GratingParam`.

    It also works for `CompositeParam` consisting of `GratingParam`.

    There are three separate constraints:
    1) Minimum distance between two grating edges should be larger than
       the minimum feature size (in pixels).
    2) Minimum distance between the left boundary and leftmost edge should be
       greater than twice the minimum feature size.
    3) Minimum distance between the right boundary and rightmost edge should be
       greater than twice the minimum feature size.

    Constraints 2 and 3 use twice the feature size to ensure that feature
    size constraints are properly satisfied even at the boundary, as the
    constraints do not take into account the selection matrix.
    """

    def __init__(self, min_feature: float,
                 boundary_constraint_scale: float = 2) -> None:
        """Creates new constraint object.

        Args:
            min_feature: Minimum feature size in terms of pixels.
            boundary_constraint_scale: Sets the constraint between the edges
                and the boundary in terms of multiples of the minimum feature
                size. Left at the default, this is exactly constraints 2 and 3
                described above.
        """
        if min_feature < 0:
            raise ValueError(
                "Minimum feature must be positive, got {}".format(min_feature))
        self._min_feature = min_feature
        self._edge_cons_scale = boundary_constraint_scale

    def calculate_objective_function(
            self, param: parametrization.Parametrization) -> np.ndarray:
        # TODO(logansu): Remove hack. This is hack because multiple
        # parametrizations are currently handled as a single composite
        # parametrization.
        if isinstance(param, parametrization.CompositeParam):
            constraints = []
            for subparam in param._params:
                constraints += self._build_constraints(subparam.to_vector(),
                                                       subparam._num_pixels)
        else:
            constraints = self._build_constraints(param.to_vector(),
                                                  param._num_pixels)
        return np.hstack(constraints)

    def calculate_gradient(
            self, param: parametrization.Parametrization) -> np.ndarray:
        # TODO(logansu): Remove hack. This is hack because multiple
        # parametrizations are currently handled as a single composite
        # parametrization.
        if isinstance(param, parametrization.CompositeParam):
            # The gradient is block-diagonal, with one block per
            # sub-parametrization.
            grad_blocks = []

            for subparam in param._params:
                grad_blocks.append(
                    self._build_constraint_grads(subparam.to_vector()))

            return scipy.linalg.block_diag(*grad_blocks)

        return self._build_constraint_grads(param.to_vector())

    def _build_constraints(self, vec: np.ndarray,
                           grating_len: int) -> List[np.ndarray]:
        """Constructs constraints for a single grating.

        The constraints involve three separate inequalities: 1) minimum distance
        between two edges, 2) left edge is not too close to the edge of the
        design region, and 3) right edge is not too close to the right of
        the design region.

        Args:
            vec: Vector corresponding to grating edge parametrization.

        Returns:
            List of constraints.
        """
        constraints = []
        # The minimum feature constraint.
        constraints.append(-(_diff_matrix(len(vec)) @ vec -
                             np.ones(len(vec) - 1) * self._min_feature))
        # Lower bounding the left and right edges. Note that we keep them at
        # least 2 times the minimum feature constraint because the selection
        # matrix at the edges may not be aligned to the grid.
        constraints.append(-vec[0] + self._edge_cons_scale * self._min_feature)
        constraints.append(-grating_len +
                           self._edge_cons_scale * self._min_feature + vec[-1])
        return constraints

    def _build_constraint_grads(self, vec: np.ndarray) -> List[np.ndarray]:
        """Constructs constraint gradients for a single grating.

        These are the gradients corresponding to `_build_constraints`.

        Args:
            vec: Vector corresponding to grating edge parametrization.

        Returns:
            List of constraint gradients.
        """
        constraints = []

        # The minimum feature constraint.
        constraints.append(-_diff_matrix(len(vec)))

        lower_bound = np.zeros(len(vec))
        lower_bound[0] = -1
        constraints.append(lower_bound)

        upper_bound = np.zeros(len(vec))
        upper_bound[-1] = 1
        constraints.append(upper_bound)

        return np.vstack(constraints)
