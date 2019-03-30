from typing import List

import numpy as np
import pytest

from spins.invdes import parametrization
from spins.invdes import problem

from spins.invdes.problem import graph_executor


def test_eval():
    param = parametrization.DirectParam((1, 2, 3, 4, 5))
    x = problem.Variable(5)
    obj = (x[0] * x[1] - x[2]**2) * (x[3] + 2 * x[4])

    assert graph_executor.eval_fun(obj, param) == -98
    np.testing.assert_array_equal(
        graph_executor.eval_grad(obj, param), [28, 14, -84, -7, -14])


def test_eval_multiple():
    param = parametrization.DirectParam((1, 2, 3, 4, 5))
    x = problem.Variable(5)
    obj_part1 = x[3] + 2 * x[4]
    obj_part2 = x[0] * x[1] - x[2]**2
    obj = obj_part1 * obj_part2

    assert graph_executor.eval_fun([obj, obj_part1, obj_part2],
                                   param) == [-98, 14, -7]


class HeavyIdentity(problem.OptimizationFunction):
    """Identity function that has heavy compute.

    This is used for testing purposes (i.e. used for code tests) because
    setting `_heavy_compute` directly can be unreliable given that operator
    overloading may choose to optimize the graph and get recognize that the
    operation isn't actually heavy. For example,
    ```python
    obj = Sum([a, b, c])
    obj._heavy_compute = True
    obj2 = obj + 2
    ```
    is actually equivalent to `obj2 = Sum([a, b, c, Constant(2)])` without
    the heavy flag set (since addition is recognized not be heavy).
    ```
    """

    def __init__(self, fun: problem.OptimizationFunction):
        super().__init__(fun, True)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return input_vals[0]

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        return [grad_val]


def test_eval_heavy_compute():
    param = parametrization.DirectParam((-1, 2, 1.2, 3, 5.1))
    x = problem.Variable(5)

    obj = HeavyIdentity(x[1] * x[0] * x[3])

    obj2 = HeavyIdentity(x[3] + x[1]**3)

    obj3 = obj * (x[0] + 2) + obj2 - x[4]

    np.testing.assert_array_almost_equal(
        graph_executor.eval_fun(obj3, param), -0.1)
    np.testing.assert_array_equal(
        graph_executor.eval_grad(obj3, param), [0, 9, 0, -1, -1])


def test_eval_heavy_compute_end_on_heavy():
    param = parametrization.DirectParam((-3, 2))
    x = problem.Variable(2)

    obj = HeavyIdentity(x[0] * x[1])

    np.testing.assert_array_almost_equal(
        graph_executor.eval_fun(obj, param), -6)
    np.testing.assert_array_equal(graph_executor.eval_grad(obj, param), [2, -3])


# Copy of the old power objective. This is used an old-style function.
class OldPower(problem.OptimizationFunction):
    """Represents an objective function raised to a constant power."""

    def __init__(self, obj, power):
        """Constructs the objective (obj)**power

        Args:
            obj: The objective to wrap.
            power: A real number power (cannot be a Constant).
        """
        self.power = power
        self.obj = obj

    def calculate_objective_function(self, param):
        return self.obj.calculate_objective_function(param)**self.power

    def calculate_gradient(self, param):
        obj_value = float(self.obj.calculate_objective_function(param))
        obj_grad = self.obj.calculate_gradient(param)
        return self.power * obj_value**(self.power - 1) * obj_grad


def test_eval_old_in_new():
    """Tests evals when a new-style function depends on old-style function."""
    param = parametrization.DirectParam((3, 2))
    x = problem.Variable(2)
    obj = OldPower(x[0], 2) + x[0]

    assert graph_executor.eval_fun(obj, param) == 12
    np.testing.assert_array_equal(graph_executor.eval_grad(obj, param), [7, 0])


def test_eval_new_in_old():
    """Tests evals when an old-style function depends on new-style function."""
    param = parametrization.DirectParam((3, 2))
    x = problem.Variable(2)
    obj = OldPower(x[0] + x[1], 2)

    assert graph_executor.eval_fun(obj, param) == 25
    np.testing.assert_array_equal(
        graph_executor.eval_grad(obj, param), [10, 10])


def test_eval_fun_no_in_node_works():
    """Tests when no input node is found.

    This can happen if the input node is fed only to the old-style function,
    which does not list its dependencies. Consequently, no input node can be
    found.
    """
    param = parametrization.DirectParam((2,))
    x = problem.Variable(1)
    obj = OldPower(x[0], 2) + OldPower(x[0], 3)

    assert graph_executor.eval_fun(obj, param) == 12
    np.testing.assert_array_equal(graph_executor.eval_grad(obj, param), [16])


def test_eval_fun_multiple_variables_raises_value_error():
    with pytest.raises(ValueError, match=r"Multiple Variable"):
        param = parametrization.DirectParam((1, 2))
        x = problem.Variable(2)
        y = problem.Variable(2)
        graph_executor.eval_fun(x + y, param)


def test_top_sort_affinity():
    graph = {
        "a": ["b", "c", "d"],
        "b": ["h"],
        "c": ["g"],
        "d": ["e"],
        "e": ["f"],
        "f": ["g"],
        "g": ["h"],
        "h": []
    }
    affinity_nodes = set(["b", "g", "f"])

    sorted_nodes = graph_executor._top_sort_affinity(graph, affinity_nodes)

    # The topological ordering must be a[cde][bf]gh where square brackets denote
    # that any combination is acceptable.
    assert sorted_nodes[0] == "a"
    assert set(sorted_nodes[1:4]) == set(["c", "d", "e"])
    assert set(sorted_nodes[4:6]) == set(["b", "f"])
    assert sorted_nodes[6] == "g"
    assert sorted_nodes[7] == "h"
