""" Test gradient descent module. """
import unittest

import numpy as np

import spins.invdes.optimization.problems as problems

from spins.invdes.optimization import (Adam, Adagrad, AdaptiveGradientDescent,
                                       GradientDescent, Nag, RmsProp)
from spins.invdes.problem.objective import Sum, Product, Constant, Variable
from spins.invdes.parametrization import DirectParam


class TestGradientDescent(unittest.TestCase):

    def test_single_variable_step(self):
        # Test that the first step is correct.

        # Test quadratic optimization: f(x) = x^2 - 4x + 1
        # Setup objective.
        x_var = Variable(1)
        x_var_squared = Product([x_var, x_var])
        obj = Sum([x_var_squared, Product([Constant(-4), x_var]), Constant(1)])

        # Optimization with x0 = 0
        param = DirectParam([0], bounds=[-10, 10])
        opt = GradientDescent(obj, param, 0.05, normalize_gradient=False)

        # Iterate once and check against manual gradient.
        opt.iterate()
        self.assertEqual(param.encode(), [0.2])

    def test_single_variable_quadratic(self):
        obj, param, optimum = problems.build_single_variable_quadratic()
        opt = GradientDescent(obj, param, 0.05)
        opt.max_iters = 500
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum)

    def test_single_variable_quartic(self):
        for instance in [0, 1]:
            obj, param, optimum = problems.build_single_variable_quartic(
                instance)
            opt = GradientDescent(obj, param, 0.05)
            opt.max_iters = 500
            opt.optimize()
            np.testing.assert_almost_equal(opt.param.encode(), optimum)

    def test_two_variable_quadratic(self):
        obj, param, optimum = problems.build_two_variable_quadratic()
        opt = GradientDescent(obj, param, 0.3)
        opt.max_iters = 100
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum)


class TestAdaptiveGradientDescent(unittest.TestCase):

    def test_single_variable_quadratic(self):
        for initial_step in [0.001, 0.01, 10]:
            obj, param, optimum = problems.build_single_variable_quadratic()
            opt = AdaptiveGradientDescent(obj, param, initial_step)
            opt.max_iters = 60
            opt.optimize()
            np.testing.assert_almost_equal(opt.param.to_vector(), optimum)

    def test_single_variable_quartic(self):
        for instance in [0, 1]:
            for initial_step in [0.001, 0.01, 10]:
                obj, param, optimum = problems.build_single_variable_quartic(
                    instance)
                opt = AdaptiveGradientDescent(obj, param, initial_step)
                opt.max_iters = 60
                opt.optimize()
                np.testing.assert_almost_equal(opt.param.to_vector(), optimum)

    def test_two_variable_quadratic(self):
        for initial_step in [0.001, 0.01, 10]:
            obj, param, optimum = problems.build_two_variable_quadratic()
            opt = AdaptiveGradientDescent(obj, param, initial_step)
            opt.max_iters = 60
            opt.optimize()
            np.testing.assert_almost_equal(opt.param.to_vector(), optimum)

    def test_rosenbrock_function(self):
        obj, param, optimum = problems.build_rosenbrock_function()
        opt = AdaptiveGradientDescent(obj, param, 0.05)
        opt.max_iters = 3000
        opt.optimize()
        np.testing.assert_almost_equal(
            opt.param.to_vector(), optimum, decimal=2)

    def test_stop_tolerance(self):
        # Check that optimization actually stops after hitting tolerance.
        objective = Variable(1)**2 + 3
        param = DirectParam([0.5], bounds=(0, 1))

        opt = AdaptiveGradientDescent(objective, param, 1)
        opt.max_iters = 10
        # We should hit stop tolerance in two steps.
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.to_vector(), 0)
        self.assertEqual(opt.iter, 2)


class TestAdagrad(unittest.TestCase):

    def test_single_variable_quadratic(self):
        obj, param, optimum = problems.build_single_variable_quadratic()
        opt = Adagrad(obj, param, 1)
        opt.max_iters = 40
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum)

    def test_single_variable_quartic(self):
        for instance in [0, 1]:
            obj, param, optimum = problems.build_single_variable_quartic(
                instance)
            opt = Adagrad(obj, param, 1)
            opt.max_iters = 70
            opt.optimize()
            np.testing.assert_almost_equal(opt.param.encode(), optimum)

    def test_two_variable_quadratic(self):
        obj, param, optimum = problems.build_two_variable_quadratic()
        opt = Adagrad(obj, param, 1)
        opt.max_iters = 70
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum)


class TestRmsProp(unittest.TestCase):

    def test_single_variable_quadratic(self):
        obj, param, optimum = problems.build_single_variable_quadratic()
        opt = RmsProp(obj, param, 0.1)
        opt.max_iters = 60
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum)

    def test_single_variable_quartic(self):
        for instance in [0, 1]:
            obj, param, optimum = problems.build_single_variable_quartic(
                instance)
            opt = RmsProp(obj, param, 0.1)
            opt.max_iters = 30
            opt.optimize()
            np.testing.assert_almost_equal(opt.param.encode(), optimum)

    def test_two_variable_quadratic(self):
        obj, param, optimum = problems.build_two_variable_quadratic()
        opt = RmsProp(obj, param, 0.1)
        opt.max_iters = 50
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum, decimal=4)


class TestAdam(unittest.TestCase):

    def test_single_variable_quadratic(self):
        obj, param, optimum = problems.build_single_variable_quadratic()
        opt = Adam(obj, param, 0.1)
        opt.max_iters = 100
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum, decimal=2)

    def test_single_variable_quartic(self):
        for instance in [0, 1]:
            obj, param, optimum = problems.build_single_variable_quartic(
                instance)
            opt = Adam(obj, param, 0.1)
            opt.max_iters = 100
            opt.optimize()
            np.testing.assert_almost_equal(
                opt.param.encode(), optimum, decimal=2)

    def test_two_variable_quadratic(self):
        obj, param, optimum = problems.build_two_variable_quadratic()
        opt = Adam(obj, param, 0.1)
        opt.max_iters = 100
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum, decimal=2)


class TestNag(unittest.TestCase):

    def test_single_variable_quadratic(self):
        obj, param, optimum = problems.build_single_variable_quadratic()
        opt = Nag(obj, param, 0.1)
        opt.max_iters = 60
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum, decimal=4)

    def test_single_variable_quartic(self):
        for instance in [0, 1]:
            obj, param, optimum = problems.build_single_variable_quartic(
                instance)
            opt = Nag(obj, param, 0.01)
            opt.max_iters = 50
            opt.optimize()
            np.testing.assert_almost_equal(
                opt.param.encode(), optimum, decimal=4)

    def test_two_variable_quadratic(self):
        obj, param, optimum = problems.build_two_variable_quadratic()
        opt = Nag(obj, param, 0.1)
        opt.max_iters = 60
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum, decimal=2)

    def test_rosenbrock_function(self):
        obj, param, optimum = problems.build_rosenbrock_function()
        opt = Nag(obj, param, 0.001)
        opt.max_iters = 1000
        opt.optimize()
        np.testing.assert_almost_equal(opt.param.encode(), optimum, decimal=2)
