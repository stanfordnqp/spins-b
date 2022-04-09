""" Test EM objectives. """
import copy
import numpy as np
import unittest

from spins.fdfd_solvers.local_matrix_solvers import DirectSolver
from spins.invdes.parametrization import DirectParam
from spins.invdes.problem import EmObjective, FdfdSimulation
import spins.fdfd_tools as fdfd_tools


class SimpleEmObjective(EmObjective):
    """ A simple 1D EM objective design to test the adjoint calculation.

    The objective is a field-matching objective:
    f(x) = |x - target_efields|^2
    """

    def __init__(self, sim: FdfdSimulation, target_efields) -> None:
        super().__init__(sim)

        self.target_efields = target_efields

    def calculate_f(self, efields, struct):
        return np.sum(np.abs(efields - self.target_efields)**2)

    def calculate_partial_df_dx(self, efields, struct):
        return np.conj(efields - self.target_efields)


class TestGradientCalculation(unittest.TestCase):
    """ Test the adjoint calculation. """

    def test_gradient(self):
        # Create a 3x3 2D grid to brute force check adjoint gradients.
        shape = [3, 3, 1]
        # Setup epsilon (pure vacuum).
        epsilon = [np.ones(shape) for i in range(3)]
        # Setup dxes. Assume dx = 40.
        dxes = [[np.ones(shape[i]) * 40 for i in range(3)] for j in range(2)]
        # Setup a point source in the center.
        J = [np.zeros(shape) for i in range(3)]
        J[2][1, 1, 0] = 1
        # Setup frequency.
        omega = 2 * np.pi / 1500
        # Avoid complexities of selection matrix by setting to the identity.
        # Number of elements: 3 field components * grid size
        S = np.identity(3 * np.prod(shape))
        # Use 2D solver.
        sim = FdfdSimulation(DirectSolver(), shape, omega, dxes, J, S, epsilon)

        # Setup target fields.
        target_fields = [
            np.zeros(shape).astype(np.complex128) for i in range(3)
        ]
        target_fields[2][:, :, 0] = 20j

        objective = SimpleEmObjective(sim, fdfd_tools.vec(target_fields))

        # Check gradient for initial parametrization of 0.5 everywhere.
        param = DirectParam(0.5 * np.ones(np.prod(shape) * 3))

        f = objective.calculate_objective_function(param)
        gradient = objective.calculate_gradient(param)

        # Now brute-force the gradient.
        eps = 1e-7  # Empirically 1e-6 to 1e-7 is the best step size.
        vec = param.encode()
        brute_gradient = np.zeros_like(vec)
        for i in range(len(vec)):
            temp_vec = np.array(vec)
            temp_vec[i] += eps
            new_f = objective.calculate_objective_function(
                DirectParam(temp_vec))
            brute_gradient[i] = (new_f - f) / eps

        np.testing.assert_almost_equal(gradient, brute_gradient, decimal=4)
