""" Test objective module. """
import unittest
from typing import List

import numpy as np
import pytest

import spins.invdes.problem.objective as objective
from spins.invdes.problem.objective import (Constant, OptimizationProblem,
                                            Variable, ValueSlice)
from spins.invdes.parametrization import DirectParam


class TestOptimizationProblem(unittest.TestCase):
    """ Test optimization problem. """

    def test_problem(self):
        # Test with no constraints.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = ValueSlice(placeholder, 0) + 2
        opt = OptimizationProblem(obj)
        self.assertEqual(opt.calculate_objective_function(param), 3)
        np.testing.assert_array_equal(opt.calculate_gradient(param), [1, 0, 0])
        np.testing.assert_array_equal(
            opt.calculate_constraints(param), ([], []))
        np.testing.assert_array_equal(
            opt.calculate_constraint_gradients(param), ([], []))

        # Test with single equality constraint.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = ValueSlice(placeholder, 0) + 2
        cons_eq = [ValueSlice(placeholder, 1) - 1]
        opt = OptimizationProblem(obj, cons_eq=cons_eq)
        self.assertEqual(opt.calculate_objective_function(param), 3)
        np.testing.assert_array_equal(opt.calculate_gradient(param), [1, 0, 0])
        eq_cons, ineq_cons = opt.calculate_constraints(param)
        np.testing.assert_array_equal(eq_cons, [1])
        np.testing.assert_array_equal(ineq_cons, [])
        eq_grad, ineq_grad = opt.calculate_constraint_gradients(param)
        self.assertEqual(len(eq_grad), 1)
        np.testing.assert_array_equal(eq_grad[0], [0, 1, 0])
        self.assertEqual(len(ineq_grad), 0)

        # Test with single inequality constraint.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = ValueSlice(placeholder, 0) + 2
        cons_ineq = [ValueSlice(placeholder, 1) - 1]
        opt = OptimizationProblem(obj, cons_ineq=cons_ineq)
        self.assertEqual(opt.calculate_objective_function(param), 3)
        np.testing.assert_array_equal(opt.calculate_gradient(param), [1, 0, 0])
        eq_cons, ineq_cons = opt.calculate_constraints(param)
        np.testing.assert_array_equal(eq_cons, [])
        np.testing.assert_array_equal(ineq_cons, [1])
        eq_grad, ineq_grad = opt.calculate_constraint_gradients(param)
        self.assertEqual(len(eq_grad), 0)
        self.assertEqual(len(ineq_grad), 1)
        np.testing.assert_array_equal(ineq_grad[0], [0, 1, 0])

        # Test with multiple constraints.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = ValueSlice(placeholder, 0) + 2
        cons_eq = [
            ValueSlice(placeholder, 1) - 1,
            ValueSlice(placeholder, 0) + ValueSlice(placeholder, 1)
        ]
        cons_ineq = [
            ValueSlice(placeholder, 2) + 2,
            ValueSlice(placeholder, 2) * ValueSlice(placeholder, 1)
        ]
        opt = OptimizationProblem(obj, cons_eq=cons_eq, cons_ineq=cons_ineq)
        self.assertEqual(opt.calculate_objective_function(param), 3)
        np.testing.assert_array_equal(opt.calculate_gradient(param), [1, 0, 0])
        eq_cons, ineq_cons = opt.calculate_constraints(param)
        np.testing.assert_array_equal(eq_cons, [1, 3])
        np.testing.assert_array_equal(ineq_cons, [5, 6])
        eq_grad, ineq_grad = opt.calculate_constraint_gradients(param)
        self.assertEqual(len(eq_grad), 2)
        np.testing.assert_array_equal(eq_grad[0], [0, 1, 0])
        np.testing.assert_array_equal(eq_grad[1], [1, 1, 0])
        self.assertEqual(len(ineq_grad), 2)
        np.testing.assert_array_equal(ineq_grad[0], [0, 0, 1])
        np.testing.assert_array_equal(ineq_grad[1], [0, 3, 2])


class TestConstant(unittest.TestCase):
    """ Tests the constant objective. """

    def test_objective_function(self):
        """ Tests the objective function value. """
        obj = objective.Constant(1)
        self.assertEqual(obj.calculate_objective_function(None), 1)

        # Make sure that a copy of the constant is made.
        val = 2
        obj = objective.Constant(val)
        val = 3
        self.assertEqual(obj.calculate_objective_function(None), 2)

        # Check complex support.
        obj = Constant(3 + 2j)
        self.assertEqual(obj.calculate_objective_function(None), 3 + 2j)

    def test_gradient(self):
        """ Tests that gradient is always zero. """
        param = DirectParam([1, 2])
        obj = objective.Constant(3)
        self.assertEqual(list(obj.calculate_gradient(param)), [0, 0])

        obj = Constant(3 + 2j)
        self.assertEqual(list(obj.calculate_gradient(param)), [0, 0])

    def test_objective_function_matrix(self):
        """ Test objective function for matrix-valued scalar. """
        obj = Constant(np.array([1, 2]))
        np.testing.assert_array_equal(
            obj.calculate_objective_function(None), np.array([1, 2]))

        obj = Constant(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(
            obj.calculate_objective_function(None), np.array([[1, 2], [3, 4]]))

    def test_gradient_matrix(self):
        """ Test gradient function for matrix-valued scalar. """
        obj = Constant(np.array([1, 2]))
        param = DirectParam([1, 2, 3])
        np.testing.assert_array_equal(
            obj.calculate_gradient(param), np.array([[0, 0, 0], [0, 0, 0]]))

        obj = Constant(np.array([[1, 2], [3, 4], [5, 6]]))
        np.testing.assert_array_equal(
            obj.calculate_gradient(param),
            np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [0, 0, 0]]]))

    def test_string(self):
        self.assertEqual(str(objective.Constant(12)), '12')


class TestValueSlice(unittest.TestCase):
    """ Test ValueSlice. """

    def test_objective_function(self):
        """ Tests that returns correct scalar from parametrization. """
        var = objective.ValueSlice(Variable(4), 0)
        self.assertEqual(
            var.calculate_objective_function(DirectParam([1, 2, 4, 2])), 1)

        var = objective.ValueSlice(Variable(1), 0)
        self.assertEqual(var.calculate_objective_function(DirectParam([1])), 1)

        var = objective.ValueSlice(Variable(4), 2)
        self.assertEqual(
            var.calculate_objective_function(DirectParam([0, 1, 4, 2])), 4)

        var = objective.ValueSlice(Variable(3), 2)
        self.assertEqual(
            var.calculate_objective_function(DirectParam([0, 1, -3])), -3)

    def test_gradient(self):
        """ Tests that scalar returns correct derivatives. """
        var = objective.ValueSlice(Variable(1), 0)
        self.assertTrue(
            np.array_equal(var.calculate_gradient(DirectParam([0])), [1]))

        var = objective.ValueSlice(Variable(3), 2)
        np.testing.assert_array_equal(
            var.calculate_gradient(DirectParam([0, 1, 2])), [0, 0, 1])

        var = objective.ValueSlice(Variable(4), 2)
        self.assertTrue(
            np.array_equal(
                var.calculate_gradient(DirectParam([0, 1, 2, 3])),
                [0, 0, 1, 0]))

    def test_string(self):
        self.assertEqual(str(objective.ValueSlice(Variable(1), 1)), 'p[1]')


class TestParallelObjectiveCalculation(unittest.TestCase):
    """ Test functions used for parallelization. """

    def test_calculate_objective_parallel(self):
        """ Tests the parallelization of objectives. """
        # Test case with no objectives.
        self.assertEqual(objective.calculate_objective_parallel([], None), [])

        # Test case with a single objective.
        objs = [objective.Constant(1)]
        self.assertEqual(
            objective.calculate_objective_parallel(objs, None), [1])

        # Test case with two objectives.
        objs = [objective.Constant(1), objective.Constant(2)]
        self.assertEqual(
            objective.calculate_objective_parallel(objs, None), [1, 2])

    def test_calculate_gradient_parallel(self):
        """ Tests the parallelization of gradients. """
        # Test case with no objectives.
        self.assertEqual(objective.calculate_gradient_parallel([], None), [])

        # Test case with a single objective.
        param = DirectParam([1, 2])
        objs = [objective.ValueSlice(Variable(2), 0)]
        self.assertTrue(
            np.array_equal(
                objective.calculate_gradient_parallel(objs, param),
                [np.array([1, 0])]))

        # Test case with two objectives.
        param = DirectParam([1, 2])
        placeholder = Variable(2)
        objs = [
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ]
        self.assertTrue(
            np.array_equal(
                objective.calculate_gradient_parallel(objs, param),
                [np.array([1, 0]), np.array([0, 1])]))


class TestSum(unittest.TestCase):
    """ Tests the sum objective. """

    def test_objective_function(self):
        """ Tests the sum part of the objective function. """
        # Use no parametrization because work with constants that
        # do not care about parametrization.
        param = None

        # Test case with one objective.
        obj = objective.Sum([objective.Constant(5)])
        self.assertEqual(obj.calculate_objective_function(param), 5)

        # Test case with two objectives.
        obj = objective.Sum([objective.Constant(2), objective.Constant(4)])
        self.assertEqual(obj.calculate_objective_function(param), 6)

        # Test case with more than two objectives.
        obj = objective.Sum([
            objective.Constant(3),
            objective.Constant(4),
            objective.Constant(-10)
        ])
        self.assertEqual(obj.calculate_objective_function(param), -3)

    def test_gradient(self):
        """ Test the gradient calculation. """
        # Test case where the first gradient is zero.
        param = DirectParam([1, 2])
        obj = objective.Sum(
            [objective.Constant(2),
             objective.ValueSlice(Variable(2), 0)])
        self.assertTrue(np.array_equal(obj.calculate_gradient(param), [1, 0]))

        # Test case where the second gradient is zero.
        param = DirectParam([1, 2])
        obj = objective.Sum(
            [objective.ValueSlice(Variable(2), 0),
             objective.Constant(2)])
        self.assertTrue(np.array_equal(obj.calculate_gradient(param), [1, 0]))

        # Test case where there are two non-zero gradients.
        placeholder = Variable(2)
        param = DirectParam([1, 2])
        obj = objective.Sum([
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 0)
        ])
        self.assertTrue(np.array_equal(obj.calculate_gradient(param), [1, 1]))

    def test_matrix(self):
        """ Test sum calculation for matrix multiplication. """
        # TODO(logansu): Enable test.
        if False:
            # Sum 1x1 variable against 2x1 constant.
            param = DirectParam([1, 2])
            obj = objective.Sum([ValueSlice(1), Constant(np.array([1, 2]))])
            np.testing.assert_array_equal(
                obj.calculate_objective_function(param), [2, 4])
            np.testing.assert_array_equal(
                obj.calculate_gradient(param), [[0, 2], [0, 4]])

    def test_weighted_sum(self):
        param = DirectParam([1, 2, 3])

        x = objective.Variable(3)
        weights = [-1, 2, 0]
        obj = objective.Sum([x[0], x[1], x[2]], weights=weights)

        np.testing.assert_array_equal(
            obj.calculate_objective_function(param), 3)
        np.testing.assert_array_equal(obj.calculate_gradient(param), [-1, 2, 0])

    def test_string(self):
        obj = objective.Sum(
            [objective.ValueSlice(Variable(2), 0),
             objective.Constant(2)],
            parallelize=False)
        self.assertTrue(str(obj), '(p[0] + 2)')

        obj = objective.Sum(
            [objective.ValueSlice(Variable(2), 0),
             objective.Constant(2)],
            parallelize=True)
        self.assertEqual(str(obj), '||(p[0] + 2)')


class TestProduct(unittest.TestCase):
    """ Tests the product objective. """

    def test_objective_function(self):
        """ Tests the product part of the objective function. """
        # Use no parametrization because work with constants that
        # do not care about parametrization.
        param = None

        # Test case with one objective.
        obj = objective.Product([objective.Constant(5)])
        self.assertEqual(obj.calculate_objective_function(param), 5)

        # Test case with two objectives.
        obj = objective.Product([objective.Constant(2), objective.Constant(4)])
        self.assertEqual(obj.calculate_objective_function(param), 8)

        # Test case with more than two objectives.
        obj = objective.Product([
            objective.Constant(3),
            objective.Constant(4),
            objective.Constant(-10)
        ])
        self.assertEqual(obj.calculate_objective_function(param), -120)

    def test_gradient(self):
        """ Test the gradient calculation. """
        # Test single variable case.
        param = DirectParam([1, 2])
        obj = objective.Product([objective.ValueSlice(Variable(2), 1)])
        self.assertTrue(np.array_equal(obj.calculate_gradient(param), [0, 1]))

        # Test single variable and constant.
        param = DirectParam([-1, 2])
        obj = objective.Product(
            [objective.Constant(4),
             objective.ValueSlice(Variable(2), 1)])
        self.assertTrue(np.array_equal(obj.calculate_gradient(param), [0, 4]))

        # Test two variable case.
        param = DirectParam([3])
        placeholder = Variable(1)
        obj = objective.Product([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 0)
        ])
        self.assertEqual(obj.calculate_gradient(param), [6.0])

        param = DirectParam([-1, 2])
        placeholder = Variable(2)
        obj = objective.Product([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ])
        self.assertTrue(np.array_equal(obj.calculate_gradient(param), [2, -1]))

        # Test three variable case.
        param = DirectParam([-1, 2, 3])
        placeholder = Variable(3)
        obj = objective.Product([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertTrue(
            np.array_equal(obj.calculate_gradient(param), [6, -3, -2]))

    def test_matrix(self):
        """ Test product calculation for matrix multiplication. """
        # TODO(logansu): Enable test.
        # Multiply 1x1 variable against 2x1 constant.
        if False:
            param = DirectParam([1, 2])
            obj = objective.Product([ValueSlice(1), Constant(np.array([1, 2]))])
            np.testing.assert_array_equal(
                obj.calculate_objective_function(param), [2, 4])
            np.testing.assert_array_equal(
                obj.calculate_gradient(param), [[0, 2], [0, 4]])

    def test_string(self):
        obj = objective.Product(
            [objective.ValueSlice(Variable(1), 0),
             objective.Constant(2)])
        self.assertEqual(str(obj), '||(p[0] * 2)')


class TestPower(unittest.TestCase):
    """ Tests the power objective. """

    def test_objective_function(self):
        param = None

        # Test with positive integral power.
        obj = objective.Power(objective.Constant(3), 2)
        self.assertEqual(obj.calculate_objective_function(param), 9)

        # Test with positive fractional power.
        obj = objective.Power(objective.Constant(9), 0.5)
        self.assertAlmostEqual(obj.calculate_objective_function(param), 3)

        # Test with negative power.
        obj = objective.Power(objective.Constant(5), -1)
        self.assertAlmostEqual(obj.calculate_objective_function(param), 0.2)

        # Test with zero.
        obj = objective.Power(objective.Constant(5), 0)
        self.assertAlmostEqual(obj.calculate_objective_function(param), 1)

    def test_gradient(self):
        # Test with positive integral power.
        param = DirectParam([5, 7])
        obj = objective.Power(
            objective.Product(
                [objective.ValueSlice(Variable(2), 0),
                 objective.Constant(3)]), 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [90, 0])

        # Test with positive fractional power.
        param = DirectParam([12, 7])
        obj = objective.Power(
            objective.Product(
                [objective.ValueSlice(Variable(2), 0),
                 objective.Constant(3)]), 0.5)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0.25, 0])

        # Test with negative power.
        param = DirectParam([1 / 3, 7])
        obj = objective.Power(
            objective.Product(
                [objective.ValueSlice(Variable(2), 0),
                 objective.Constant(3)]), -1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [-3, 0])

        # Test with zero.
        param = DirectParam([12, 7])
        obj = objective.Power(
            objective.Product(
                [objective.ValueSlice(Variable(2), 0),
                 objective.Constant(3.)]), 0)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

    def test_string(self):
        obj = objective.Power(objective.ValueSlice(Variable(1), 1), 2)
        self.assertEqual(str(obj), 'p[1]**2')


class AbsTestFunction(objective.OptimizationFunction):
    """Objective to test abs value.

    This is needed because z (the structure) is real and not complex.
    """

    def __init__(self, obj, const):
        """Constructs the objective `f(x) = c * x`.

        Args:
            obj: The objective to wrap.
            const: Complex constant.
        """
        super().__init__(obj)
        self._const = const

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return self._const * input_vals[0]

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        # TODO(logansu): Fix gradient. See `objective.AbsoluteValue` for details
        # on why this happened.
        return [np.real(grad_val * self._const)]


class TestAbsoluteValue(unittest.TestCase):

    def test_objective_function(self):
        param = None

        obj = objective.AbsoluteValue(objective.Constant(3))
        self.assertEqual(obj.calculate_objective_function(param), 3)

        obj = objective.AbsoluteValue(objective.Constant(-5))
        self.assertEqual(obj.calculate_objective_function(param), 5)

        obj = objective.AbsoluteValue(objective.Constant(1 + 1j * 9))
        self.assertEqual(
            obj.calculate_objective_function(param), abs(1 + 1j * 9))

    def test_gradient(self):
        param = DirectParam([0, 3])
        obj = objective.AbsoluteValue(objective.ValueSlice(Variable(2), 1))
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 1])

        param = DirectParam([0, -5])
        obj = objective.AbsoluteValue(objective.ValueSlice(Variable(2), 1))
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, -1])

        param = DirectParam([1])
        obj = objective.AbsoluteValue(AbsTestFunction(Variable(1), 1 + 9j))
        np.testing.assert_allclose(obj.calculate_gradient(param), [abs(1 + 9j)])

    def test_chain_rule_complex(self):
        """Tests that chain rule works properly for complex functions."""
        param = DirectParam([3])
        obj = objective.AbsoluteValue(
            objective.Power(AbsTestFunction(Variable(1), 1 - 2j), 2))
        np.testing.assert_allclose(
            obj.calculate_gradient(param), [2 * abs(1 - 2j)**2 * 3])

    def test_string(self):
        obj = objective.AbsoluteValue(objective.ValueSlice(Variable(1), 0))
        self.assertEqual(str(obj), "abs(p[0])")


class TestIndicatorPlus(unittest.TestCase):

    def test_objective_function(self):
        param = None

        obj = objective.IndicatorPlus(objective.Constant(3), 2, 1)
        self.assertEqual(obj.calculate_objective_function(param), 1)

        obj = objective.IndicatorPlus(objective.Constant(2), 3, 1)
        self.assertEqual(obj.calculate_objective_function(param), 0)

        obj = objective.IndicatorPlus(objective.Constant(9), 5, 2)
        self.assertEqual(obj.calculate_objective_function(param), 16)

        obj = objective.IndicatorPlus(objective.Constant(5), 9, 2)
        self.assertEqual(obj.calculate_objective_function(param), 0)

        obj = objective.IndicatorPlus(objective.Constant(4), 2, 0)
        self.assertEqual(obj.calculate_objective_function(param), 1)

        obj = objective.IndicatorPlus(objective.Constant(4), 2, -1)
        self.assertEqual(obj.calculate_objective_function(param), 0.5)

    def test_gradient(self):
        param = DirectParam([3, 7])
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 0), 2, 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [1, 0])

        param = DirectParam([2, 7])
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 0), 3, 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([9, 7])
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 0), 5, 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [8, 0])

        param = DirectParam([5, 7])
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 0), 9, 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([4, 7])
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 0), 2, 0)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([4, 7])
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 0), 2, -1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [-0.25, 0])

    def test_string(self):
        obj = objective.IndicatorPlus(
            objective.ValueSlice(Variable(2), 1), 2, 3)
        self.assertEqual(str(obj), 'I_plus(p[1]-2)**3')


class TestIndicatorMin(unittest.TestCase):

    def test_objective_function(self):
        param = None

        obj = objective.IndicatorMin(objective.Constant(3), 2, 1)
        self.assertEqual(obj.calculate_objective_function(param), 0)

        obj = objective.IndicatorMin(objective.Constant(2), 3, 1)
        self.assertEqual(obj.calculate_objective_function(param), 1)

        obj = objective.IndicatorMin(objective.Constant(9), 5, 2)
        self.assertEqual(obj.calculate_objective_function(param), 0)

        obj = objective.IndicatorMin(objective.Constant(5), 9, 2)
        self.assertEqual(obj.calculate_objective_function(param), 16)

        obj = objective.IndicatorMin(objective.Constant(2), 4, 0)
        self.assertEqual(obj.calculate_objective_function(param), 1)

        obj = objective.IndicatorMin(objective.Constant(2), 4, -1)
        self.assertEqual(obj.calculate_objective_function(param), 0.5)

    def test_gradient(self):
        param = DirectParam([3, 7])
        obj = objective.IndicatorMin(objective.ValueSlice(Variable(2), 0), 2, 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([2, 7])
        obj = objective.IndicatorMin(objective.ValueSlice(Variable(2), 0), 3, 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [-1, 0])

        param = DirectParam([9, 7])
        obj = objective.IndicatorMin(objective.ValueSlice(Variable(2), 0), 5, 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([5, 7])
        obj = objective.IndicatorMin(objective.ValueSlice(Variable(2), 0), 9, 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [-8, 0])

        param = DirectParam([2, 7])
        obj = objective.IndicatorMin(objective.ValueSlice(Variable(2), 0), 4, 0)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([2, 7])
        obj = objective.IndicatorMin(
            objective.ValueSlice(Variable(2), 0), 4, -1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0.25, 0])

    def test_string(self):
        obj = objective.IndicatorMin(objective.ValueSlice(Variable(2), 1), 2, 3)
        self.assertEqual(str(obj), 'I_min(p[1]-2)**3')


class TestPowerComparison(unittest.TestCase):

    def test_objective_function(self):
        param = None

        obj = objective.PowerComparison(objective.Constant(3), [2, 4], 1)
        self.assertEqual(obj.calculate_objective_function(param), 0)

        obj = objective.PowerComparison(objective.Constant(0), [2, 4], 1)
        self.assertEqual(obj.calculate_objective_function(param), 2)

        obj = objective.PowerComparison(objective.Constant(5), [2, 4], 1)
        self.assertEqual(obj.calculate_objective_function(param), 1)

        obj = objective.PowerComparison(objective.Constant(3), [2, 4], 2)
        self.assertEqual(obj.calculate_objective_function(param), 0)

        obj = objective.PowerComparison(objective.Constant(0), [2, 4], 2)
        self.assertEqual(obj.calculate_objective_function(param), 4)

        obj = objective.PowerComparison(objective.Constant(9), [2, 4], 2)
        self.assertEqual(obj.calculate_objective_function(param), 25)

    def test_gradient(self):
        param = DirectParam([3, 7])
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 0), [2, 4], 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([0, 7])
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 0), [2, 4], 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [-1, 0])

        param = DirectParam([5, 7])
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 0), [2, 4], 1)
        np.testing.assert_allclose(obj.calculate_gradient(param), [1, 0])

        param = DirectParam([3, 7])
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 0), [2, 4], 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [0, 0])

        param = DirectParam([0, 7])
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 0), [2, 4], 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [-4, 0])

        param = DirectParam([9, 7])
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 0), [2, 4], 2)
        np.testing.assert_allclose(obj.calculate_gradient(param), [10, 0])

    def test_string(self):
        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 1), [2, 4], 2)
        self.assertEqual(str(obj), 'PowerComp(2-4, power = 2)')

        obj = objective.PowerComparison(
            objective.ValueSlice(Variable(2), 1), [3, 6], 3)
        self.assertEqual(str(obj), 'PowerComp(3-6, power = 3)')


class TestLogSumExp(unittest.TestCase):
    """ Tests log-sum-exp. """

    def test_objective_function(self):
        # Single variable.
        param = DirectParam([4])
        obj = objective.LogSumExp([objective.ValueSlice(Variable(1), 0)])
        self.assertAlmostEqual(obj.calculate_objective_function(param), 4)

        # Two variables.
        param = DirectParam([1, 2])
        placeholder = Variable(2)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ])
        self.assertAlmostEqual(
            obj.calculate_objective_function(param), 2.31326168)

        # Three variables.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertAlmostEqual(
            obj.calculate_objective_function(param), 3.407606)

        # Overflow test.
        param = DirectParam([100, 3000, 3000])
        placeholder = Variable(3)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertAlmostEqual(
            obj.calculate_objective_function(param), 3000.69314718)

        # Underflow test.
        param = DirectParam([-1000, -2000, -3000])
        placeholder = Variable(3)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertAlmostEqual(obj.calculate_objective_function(param), -1000)

    def test_gradient(self):
        # Single variable.
        param = DirectParam([4])
        obj = objective.LogSumExp([objective.ValueSlice(Variable(1), 0)])
        np.testing.assert_almost_equal(obj.calculate_gradient(param), [1])

        # Two variables.
        param = DirectParam([1, 2])
        placeholder = Variable(2)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ])
        np.testing.assert_almost_equal(
            obj.calculate_gradient(param), [0.26894142, 0.73105858])

        # Three variables.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        np.testing.assert_almost_equal(
            obj.calculate_gradient(param), [0.09003057, 0.24472847, 0.66524095])

        # Overflow test.
        param = DirectParam([100, 3000, 3000])
        placeholder = Variable(3)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        np.testing.assert_almost_equal(
            obj.calculate_gradient(param), [0, 0.5, 0.5])

        # Underflow test.
        param = DirectParam([-1000, -2000, -3000])
        placeholder = Variable(3)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        np.testing.assert_almost_equal(obj.calculate_gradient(param), [1, 0, 0])

    def test_string(self):
        placeholder = Variable(2)
        obj = objective.LogSumExp([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ])
        self.assertTrue(str(obj), 'LogSumExp(p[0], p[1])')


class TestSoftmaxAverage(unittest.TestCase):
    """ Tests softmax average. """

    def test_objective_function(self):
        # Single variable.
        param = DirectParam([4])
        obj = objective.SoftmaxAverage([objective.ValueSlice(Variable(1), 0)])
        self.assertAlmostEqual(obj.calculate_objective_function(param), 4)

        # Two variables.
        param = DirectParam([1, 2])
        placeholder = Variable(2)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ])
        self.assertAlmostEqual(
            obj.calculate_objective_function(param), 1.7310585786)

        # Three variables.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertAlmostEqual(
            obj.calculate_objective_function(param), 2.5752103826)

        # Overflow test.
        param = DirectParam([100, 3000, 3000])
        placeholder = Variable(3)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertAlmostEqual(obj.calculate_objective_function(param), 3000)

        # Underflow test.
        param = DirectParam([-1000, -2000, -3000])
        placeholder = Variable(3)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        self.assertAlmostEqual(obj.calculate_objective_function(param), -1000)

    def test_gradient(self):
        # Single variable.
        param = DirectParam([4])
        obj = objective.SoftmaxAverage([objective.ValueSlice(Variable(1), 0)])
        np.testing.assert_almost_equal(obj.calculate_gradient(param), [1])

        # Two variables.
        param = DirectParam([1, 2])
        placeholder = Variable(2)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1)
        ])
        np.testing.assert_almost_equal(
            obj.calculate_gradient(param), [0.0723295, 0.9276705])

        # Three variables.
        param = DirectParam([1, 2, 3])
        placeholder = Variable(3)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        np.testing.assert_almost_equal(
            obj.calculate_gradient(param), [-0.0517865, 0.1039581, 0.9478284])

        # Overflow test.
        param = DirectParam([100, 3000, 3000])
        plaholder = Variable(3)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        np.testing.assert_almost_equal(
            obj.calculate_gradient(param), [0, 0.5, 0.5])

        # Underflow test.
        param = DirectParam([-1000, -2000, -3000])
        placeholder = Variable(3)
        obj = objective.SoftmaxAverage([
            objective.ValueSlice(placeholder, 0),
            objective.ValueSlice(placeholder, 1),
            objective.ValueSlice(placeholder, 2)
        ])
        np.testing.assert_almost_equal(obj.calculate_gradient(param), [1, 0, 0])

    def test_string(self):
        obj = objective.LogSumExp([
            objective.ValueSlice(Variable(2), 0),
            objective.ValueSlice(Variable(2), 1)
        ])
        self.assertTrue(str(obj), 'LogSumExp(p[0], p[1])')


class TestOperatorComposition(unittest.TestCase):
    """ Test that objective composition using operators. """

    class DummyObjective(objective.OptimizationFunction):
        """ Dummy objective that is a concrete realization of OptimizationFunction.

        Note that it is a bad idea to use a particular objective (e.g. Constant)
        because that objective may decide to implement specialized operations.
        """

        def __init__(self, placeholder, value):
            super().__init__(placeholder)

            self.value = value

        def eval(self, input_vals):
            return self.value

        def grad(self, input_vals, grad_val):
            return self.value

        def __str__(self):
            return 'dummy(' + str(self.value) + ')'

    def test_sum(self):
        # Test that sum operation works as intended.
        placeholder = Variable(1)
        param = DirectParam([2])

        # Two objectives.
        obj1 = self.DummyObjective(placeholder, 2)
        obj2 = self.DummyObjective(placeholder, 3.1)
        sum_obj = obj1 + obj2
        self.assertAlmostEqual(sum_obj.calculate_objective_function(param), 5.1)
        self.assertEqual(str(sum_obj), '||(dummy(2) + dummy(3.1))')

        # Three objectives.
        obj1 = self.DummyObjective(placeholder, 2)
        obj2 = self.DummyObjective(placeholder, 3.1)
        obj3 = self.DummyObjective(placeholder, 4)
        sum_obj = obj1 + obj2 + obj3
        self.assertAlmostEqual(sum_obj.calculate_objective_function(param), 9.1)
        self.assertEqual(str(sum_obj), '||(dummy(2) + dummy(3.1) + dummy(4))')

        # Add an integer.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = obj1 + 4
        self.assertEqual(sum_obj.calculate_objective_function(param), 5)
        self.assertEqual(str(sum_obj), '(dummy(1) + 4)')

        # Add a float.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = obj1 + 4.1
        self.assertAlmostEqual(sum_obj.calculate_objective_function(param), 5.1)
        self.assertEqual(str(sum_obj), '(dummy(1) + 4.1)')

        # Reverse add an integer.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = 4 + obj1
        self.assertEqual(sum_obj.calculate_objective_function(param), 5)
        self.assertEqual(str(sum_obj), '(dummy(1) + 4)')

        # Add a constant.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = obj1 + objective.Constant(4)
        self.assertEqual(sum_obj.calculate_objective_function(param), 5)
        self.assertEqual(str(sum_obj), '(dummy(1) + 4)')

        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = (obj1 + obj1) + (4 + 1j)
        self.assertEqual(sum_obj.calculate_objective_function(param), 6 + 1j)
        self.assertEqual(str(sum_obj), '||(dummy(1) + dummy(1) + (4+1j))')

        # Adding two constants.
        obj1 = objective.Constant(4)
        obj2 = objective.Constant(5)
        sum_obj = obj1 + obj2
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), 9)

        obj1 = objective.Constant(4)
        sum_obj = obj1 + 5
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), 9)

        obj1 = objective.Constant(4)
        sum_obj = 5 + obj1
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), 9)

        obj1 = objective.Constant(4)
        sum_obj = (1 + 2j) + obj1
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), 5 + 2j)

        obj1 = objective.Constant(np.array([[1, 2], [3, 4]]))
        obj2 = np.array([[2, 0], [3, 0]])
        sum_obj = obj1 + obj2
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        np.testing.assert_array_equal(
            sum_obj.calculate_objective_function(param), [[3, 2], [6, 4]])

        obj1 = objective.Constant(np.array([[1, 2], [3, 4]]))
        obj2 = np.array([[2, 0], [3, 0]])
        sum_obj = obj1 + obj2 + obj1
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        np.testing.assert_array_equal(
            sum_obj.calculate_objective_function(param), [[4, 4], [9, 8]])

        # Objective summing fun.
        obj1 = self.DummyObjective(placeholder, 1)
        obj2 = self.DummyObjective(placeholder, 2)
        obj3 = self.DummyObjective(placeholder, 3)
        obj4 = self.DummyObjective(placeholder, 4)
        sum_obj = (obj1 + 6 + obj2 + (obj3 + obj4)) + 5
        self.assertEqual(sum_obj.calculate_objective_function(param), 21)
        self.assertEqual(
            str(sum_obj),
            '||(dummy(1) + 6 + dummy(2) + dummy(3) + dummy(4) + 5)')

    def test_sub(self):
        # Test subtraction.
        placeholder = Variable(1)
        param = DirectParam([2])

        # Two objectives.
        obj1 = self.DummyObjective(placeholder, 2)
        obj2 = self.DummyObjective(placeholder, 3.1)
        sum_obj = obj1 - obj2
        self.assertAlmostEqual(
            sum_obj.calculate_objective_function(param), -1.1)
        self.assertEqual(str(sum_obj), '||(dummy(2) + (dummy(3.1) * -1))')

        # Three objectives.
        obj1 = self.DummyObjective(placeholder, 2)
        obj2 = self.DummyObjective(placeholder, 3.1)
        obj3 = self.DummyObjective(placeholder, 4)
        sum_obj = obj1 - obj2 - obj3
        self.assertAlmostEqual(
            sum_obj.calculate_objective_function(param), -5.1)
        self.assertEqual(
            str(sum_obj),
            '||(dummy(2) + (dummy(3.1) * -1) ' + '+ (dummy(4) * -1))')

        # Subtract an integer.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = obj1 - 4
        self.assertEqual(sum_obj.calculate_objective_function(param), -3)
        self.assertEqual(str(sum_obj), '(dummy(1) + -4)')

        # Subtract a float.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = obj1 - 4.1
        self.assertAlmostEqual(
            sum_obj.calculate_objective_function(param), -3.1)
        self.assertEqual(str(sum_obj), '(dummy(1) + -4.1)')

        # Reverse subtract an integer.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = 4 - obj1
        self.assertEqual(sum_obj.calculate_objective_function(param), 3)
        self.assertEqual(str(sum_obj), '((dummy(1) * -1) + 4)')

        # Use negation to do subtraction.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = -obj1 + 4
        self.assertEqual(sum_obj.calculate_objective_function(param), 3)
        self.assertEqual(str(sum_obj), '((dummy(1) * -1) + 4)')

        # Subtract a constant.
        obj1 = self.DummyObjective(placeholder, 1)
        sum_obj = obj1 - objective.Constant(4)
        self.assertEqual(sum_obj.calculate_objective_function(param), -3)
        self.assertEqual(str(sum_obj), '(dummy(1) + -4)')

        # Subtracting two constants.
        obj1 = objective.Constant(4)
        obj2 = objective.Constant(5)
        sum_obj = obj1 - obj2
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), -1)

        obj1 = objective.Constant(4)
        sum_obj = obj1 - 5
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), -1)

        obj1 = objective.Constant(4)
        sum_obj = 5 - obj1
        self.assertTrue(isinstance(sum_obj, objective.Constant))
        self.assertEqual(sum_obj.calculate_objective_function(param), 1)

    def test_multiply(self):
        # Test that multiply operation works as intended.
        placeholder = Variable(1)
        param = DirectParam([2])

        # Two objectives.
        obj1 = self.DummyObjective(placeholder, 2)
        obj2 = self.DummyObjective(placeholder, 3.1)
        prod_obj = obj1 * obj2
        self.assertAlmostEqual(
            prod_obj.calculate_objective_function(param), 6.2)
        self.assertEqual(str(prod_obj), '||(dummy(2) * dummy(3.1))')

        # Three objectives.
        obj1 = self.DummyObjective(placeholder, 2)
        obj2 = self.DummyObjective(placeholder, 3)
        obj3 = self.DummyObjective(placeholder, 4)
        prod_obj = obj1 * obj2 * obj3
        self.assertEqual(prod_obj.calculate_objective_function(param), 24)
        self.assertEqual(str(prod_obj), '||(dummy(2) * dummy(3) * dummy(4))')

        # Multiply by an integer.
        obj1 = self.DummyObjective(placeholder, 1)
        prod_obj = obj1 * 4
        self.assertEqual(prod_obj.calculate_objective_function(param), 4)
        self.assertEqual(str(prod_obj), '(dummy(1) * 4)')

        # Multiply by a float.
        obj1 = self.DummyObjective(placeholder, 1)
        prod_obj = obj1 * 4.1
        self.assertAlmostEqual(
            prod_obj.calculate_objective_function(param), 4.1)
        self.assertEqual(str(prod_obj), '(dummy(1) * 4.1)')

        # Reverse multiply an integer.
        obj1 = self.DummyObjective(placeholder, 1)
        prod_obj = 4 * obj1
        self.assertEqual(prod_obj.calculate_objective_function(param), 4)
        self.assertEqual(str(prod_obj), '(dummy(1) * 4)')

        # Multiply by an constant.
        obj1 = self.DummyObjective(placeholder, 1)
        prod_obj = obj1 * objective.Constant(4)
        self.assertEqual(prod_obj.calculate_objective_function(param), 4)
        self.assertEqual(str(prod_obj), '(dummy(1) * 4)')

        obj1 = self.DummyObjective(placeholder, 1)
        prod_obj = (obj1 * obj1) * (3 + 2j)
        self.assertEqual(prod_obj.calculate_objective_function(param), 3 + 2j)
        self.assertEqual(str(prod_obj), '||(dummy(1) * dummy(1) * (3+2j))')

        # Multiply two constants.
        obj1 = objective.Constant(4)
        obj2 = objective.Constant(5)
        prod_obj = obj1 * obj2
        self.assertTrue(isinstance(prod_obj, objective.Constant))
        self.assertEqual(prod_obj.calculate_objective_function(param), 20)

        obj1 = objective.Constant(4)
        prod_obj = obj1 * 5
        self.assertTrue(isinstance(prod_obj, objective.Constant))
        self.assertEqual(prod_obj.calculate_objective_function(param), 20)

        obj1 = objective.Constant(4)
        prod_obj = 5 * obj1
        self.assertTrue(isinstance(prod_obj, objective.Constant))
        self.assertEqual(prod_obj.calculate_objective_function(param), 20)

        obj1 = objective.Constant(4)
        prod_obj = (1 + 2j) * obj1
        self.assertTrue(isinstance(prod_obj, objective.Constant))
        self.assertEqual(prod_obj.calculate_objective_function(param), 4 + 8j)

        obj1 = objective.Constant(np.array([[1, 2], [3, 4]]))
        obj2 = np.array([[2, 0], [3, 0]])
        prod_obj = obj1 * obj2
        self.assertTrue(isinstance(prod_obj, objective.Constant))
        np.testing.assert_array_equal(
            prod_obj.calculate_objective_function(param), [[2, 0], [9, 0]])

        # Objective product fun.
        obj1 = self.DummyObjective(placeholder, 1)
        obj2 = self.DummyObjective(placeholder, 2)
        obj3 = self.DummyObjective(placeholder, 3)
        obj4 = self.DummyObjective(placeholder, 4)
        prod_obj = (obj1 * 6 * obj2 * (obj3 * obj4)) * 5
        self.assertEqual(prod_obj.calculate_objective_function(param), 720)
        self.assertEqual(
            str(prod_obj),
            '||(dummy(1) * 6 * dummy(2) * dummy(3) * dummy(4) * 5)')

    def test_power(self):
        # Test that the pow operator works.
        placeholder = Variable(1)
        param = DirectParam([2])

        # Basic usage.
        obj = self.DummyObjective(placeholder, 2)**3
        self.assertAlmostEqual(obj.calculate_objective_function(param), 8)
        self.assertEqual(str(obj), 'dummy(2)**3')

        # Raising power by Constant is okay.
        obj = self.DummyObjective(placeholder, 2)**objective.Constant(3)
        self.assertAlmostEqual(obj.calculate_objective_function(param), 8)
        self.assertEqual(str(obj), 'dummy(2)**3')

        # Raising power of Constant should produce a Constant.
        obj = objective.Constant(2)**3
        self.assertTrue(isinstance(obj, objective.Constant))
        self.assertAlmostEqual(obj.calculate_objective_function(param), 8)

    def test_combined(self):
        # Test that combination of operators works fine.
        placeholder = Variable(1)
        param = DirectParam([2])

        obj1 = self.DummyObjective(placeholder, 1)
        obj2 = self.DummyObjective(placeholder, 2)
        obj3 = self.DummyObjective(placeholder, 3)
        obj4 = self.DummyObjective(placeholder, 4)
        obj = obj1 + 2 * obj2 - obj3 * (6 + obj4)
        self.assertEqual(obj.calculate_objective_function(param), -25)
        self.assertEqual(
            str(obj), '||(dummy(1) + (dummy(2) * 2) ' +
            '+ ||(dummy(3) * (dummy(4) + 6) * -1))')

        # Test that reusing objectives will not cause any issue.
        obj1 = self.DummyObjective(placeholder, 1)
        obj2 = self.DummyObjective(placeholder, 2)
        sum_obj = obj1 + obj2
        obj = (sum_obj + obj1) * sum_obj**2
        self.assertEqual(obj.calculate_objective_function(param), 36)
        self.assertEqual(
            str(obj),
            '||(||(dummy(1) + dummy(2) + dummy(1)) * ||(dummy(1) + dummy(2))**2)'
        )
