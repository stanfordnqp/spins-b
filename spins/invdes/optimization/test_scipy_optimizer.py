""" Test the scipy optimizer. """
import unittest

import numpy as np

from spins.invdes.problem.objective import Sum, Product, Constant, Variable
import spins.invdes.optimization.problems as problems
from spins.invdes.optimization import ScipyOptimizer
from spins.invdes.parametrization import DirectParam


class TestScipyOptimizer(unittest.TestCase):
    """ Test scipy optimizer. """

    def setUp(self):
        # List of methods to test on.
        self.methods = [
            'Nelder-Mead', 'Powell', 'CG', 'L-BFGS-B', 'TNC', 'SLSQP'
        ]

    def test_single_variable_quadratic(self):
        for method in self.methods:
            obj, param, optimum = problems.build_single_variable_quadratic()
            opt = ScipyOptimizer(obj, param, method)
            opt.optimize()
            np.testing.assert_almost_equal(
                opt.param.to_vector(), optimum, decimal=4)

    def test_two_variable_quadratic(self):
        for method in self.methods:
            obj, param, optimum = problems.build_two_variable_quadratic()
            opt = ScipyOptimizer(obj, param, method)
            opt.optimize()
            np.testing.assert_almost_equal(
                opt.param.to_vector(), optimum, decimal=4)

    def test_rosenbrock_function(self):
        for method in self.methods:
            obj, param, optimum = problems.build_rosenbrock_function()
            opt = ScipyOptimizer(obj, param, method)
            opt.optimize()
            np.testing.assert_almost_equal(
                opt.param.to_vector(), optimum, decimal=2)

    def test_constrained_optimization(self):
        # Test that constrained optimization works on SLSQP.
        # Implements f(x) = x^2 + 2y^2 - 5y - 2xy constrainted to x - y >= 1
        var = Variable(2)
        x_var = var[0]
        y_var = var[1]

        obj = Sum([
            Product([x_var, x_var]),
            Product([Constant(2), y_var, y_var]),
            Product([Constant(-5), y_var]),
            Product([Constant(-2), x_var, y_var])
        ])
        param = DirectParam(np.array([0, 0]), bounds=[-10, 10])
        constraints = [{
            'type': 'ineq',
            'fun': lambda z: z[0] - z[1] - 1,
            'jac': lambda z: np.array([1, -1])
        }]

        opt = ScipyOptimizer(obj, param, 'SLSQP', constraints=constraints)
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.to_vector(), [7 / 2, 5 / 2])

    def test_constrained_optimization2(self):
        optimizer = ScipyOptimizer(method='SLSQP')

        for opt, param, ans in problems.build_constrained_problem_list():
            out_param = optimizer(opt, param)
            np.testing.assert_array_almost_equal(
                out_param.to_vector(), ans, decimal=4)
