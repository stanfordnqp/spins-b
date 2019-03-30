import unittest

import numpy as np
import scipy.sparse as sparse

from spins.invdes.parametrization import DirectParam
from spins.invdes.problem import (OptimizationProblem, SlackOptimizationProblem,
                                  ValueSlice, Variable)
from spins.invdes.problem.slack_optimization_problem import SlackParam


class TestSlackParam(unittest.TestCase):
    # Test SlackParam.
    def test_direct_param(self):
        # Test by wrapping a DirectParam.

        # No slack variables.
        orig_param = DirectParam([0.1, 2, -3], bounds=(0, 1))
        param = SlackParam(orig_param, 0)
        self.assertEqual(param.get_structure().tolist(), [0.1, 2, -3])
        np.testing.assert_array_equal(
            np.array(param.calculate_gradient().todense()),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(param.get_param(), orig_param)
        param.project()
        np.testing.assert_array_equal(param.get_structure(), [0.1, 1, 0])
        self.assertEqual(param.get_bounds(), ((0, 0, 0), (1, 1, 1)))

        np.testing.assert_array_equal(param.to_vector(), [0.1, 1, 0])
        param.from_vector([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(param.to_vector(), [0.1, 0.2, 0.3])
        param.deserialize(param.serialize())
        np.testing.assert_array_equal(param.to_vector(), [0.1, 0.2, 0.3])

        # One slack variable.
        orig_param = DirectParam([0.1, 2, -3], bounds=(0, 1))
        param = SlackParam(orig_param, 1)
        self.assertEqual(param.get_structure().tolist(), [0.1, 2, -3])
        np.testing.assert_array_equal(
            np.array(param.calculate_gradient().todense()),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(param.get_param(), orig_param)
        param.project()
        np.testing.assert_array_equal(param.get_structure(), [0.1, 1, 0])
        self.assertEqual(param.get_bounds(), ((0, 0, 0, 0), (1, 1, 1, None)))

        np.testing.assert_array_equal(param.to_vector(), [0.1, 1, 0, 0])
        param.from_vector([0.1, 0.2, 0.3, 1])
        np.testing.assert_array_equal(param.to_vector(), [0.1, 0.2, 0.3, 1])
        param.deserialize(param.serialize())
        np.testing.assert_array_equal(param.to_vector(), [0.1, 0.2, 0.3, 1])

        self.assertEqual(param.get_slack_variable(0), 1)

        # Two slack variables.
        orig_param = DirectParam([0.1, 2, -3], bounds=(0, 1))
        param = SlackParam(orig_param, 2)
        self.assertEqual(param.get_structure().tolist(), [0.1, 2, -3])
        np.testing.assert_array_equal(
            np.array(param.calculate_gradient().todense()),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(param.get_param(), orig_param)
        param.project()
        np.testing.assert_array_equal(param.get_structure(), [0.1, 1, 0])
        self.assertEqual(param.get_bounds(), ((0, 0, 0, 0, 0),
                                              (1, 1, 1, None, None)))

        np.testing.assert_array_equal(param.to_vector(), [0.1, 1, 0, 0, 0])
        param.from_vector([0.1, 0.2, 0.3, 1, 2])
        np.testing.assert_array_equal(param.to_vector(), [0.1, 0.2, 0.3, 1, 2])
        param.deserialize(param.serialize())
        np.testing.assert_array_equal(param.to_vector(), [0.1, 0.2, 0.3, 1, 2])

        self.assertEqual(param.get_slack_variable(0), 1)
        self.assertEqual(param.get_slack_variable(1), 2)

        # Quick check for wrapping params with no bounds.
        param = SlackParam(DirectParam([1, 2, 3], bounds=None), 2)
        self.assertEqual(param.get_bounds(), ((None, None, None, 0, 0),
                                              (None, None, None, None, None)))


class TestSlackOptimizationProblem(unittest.TestCase):

    def test_sanity(self):
        # Quick sanity checks.
        x = Variable(3)
        opt = OptimizationProblem(
            x[0], cons_eq=(x[1],), cons_ineq=(x[2] + 2, x[1] - 1))
        slack_opt = SlackOptimizationProblem(opt)
        # Test that number of constraints is correct.
        self.assertEqual(len(slack_opt.get_inequality_constraints()), 0)
        self.assertEqual(len(slack_opt.get_equality_constraints()), 3)
        # Test that we can use SlackParam.
        slack_param = slack_opt.build_param(DirectParam([3, 1, 2]))
        self.assertEqual(slack_opt.calculate_objective_function(slack_param), 3)
        self.assertEqual(
            slack_opt.calculate_gradient(slack_param).tolist(), [1, 0, 0, 0, 0])
        eq_cons, ineq_cons = slack_opt.calculate_constraints(slack_param)
        self.assertEqual(eq_cons.tolist(), [1, 4, 0])
        self.assertEqual(ineq_cons.tolist(), [])
        eq_grad, ineq_grad = slack_opt.calculate_constraint_gradients(
            slack_param)
        self.assertEqual(len(eq_grad), 3)
        self.assertEqual(len(ineq_grad), 0)
        self.assertEqual(eq_grad[0].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(eq_grad[1].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(eq_grad[2].tolist(), [0, 1, 0, 0, 1])
