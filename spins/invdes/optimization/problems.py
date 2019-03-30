""" Collection of optimization problems for testing and benchmarking optimizers.

Each problem is a function of the form build_problem_name. The function returns
a tuple containing 1) the objective, 2) the initial parametrization, and 3) the
optimum value.

Example:
    obj, param, optimum = build_rosenbrock_function()
    optimizer = GradientDescent(obj, param)
    optimizer.max_iters = 100
    optimizer.optimize()
    # Calculate error with optimal value.
    error = np.linalg.norm(param.encode() - optimum)
"""
import numpy as np

from spins.invdes.problem.objective import (
    Constant, OptimizationFunction, OptimizationProblem, Product, Variable, Sum)
from spins.invdes.parametrization import DirectParam, Parametrization


def build_single_variable_quadratic():
    """ Implements f(x) = x^2 - 4x + 1 """
    # Setup objective.
    x_var = Variable(1)
    x_var_squared = Product([x_var, x_var])
    obj = Sum([x_var_squared, Product([Constant(-4), x_var]), Constant(1)])

    # Optimization with x0 = 0
    param = DirectParam(np.array([0]), bounds=[-10, 10])
    return (obj, param, [2])


def build_single_variable_quartic(instance: int = 0):
    """ Implements: f(x) = (x-2)(x-3)(x+1)^2

    This function as both a local optimum at x = -1 and a global optima at
    x = (13 + sqrt(57)) / 8.

    Arg:
        Instance: Either 0 or 1 indicating whether initial condition should be
            set to find the global optimum (0) or the local optimum (1).
    """
    # Test quartic optimization: f(x) = (x - 2)(x - 3)(x + 1)^2.
    x_var = Variable(1)
    obj = Product([
        Sum([x_var, Constant(-2)]),
        Sum([x_var, Constant(-3)]),
        Sum([x_var, Constant(1)]),
        Sum([x_var, Constant(1)])
    ])
    if instance == 0:
        # Optimization with x0 = 1 should land in the optima at
        # (13 + sqrt(57)) / 8.
        param = DirectParam(np.array([1]), bounds=[-10, 10])
        return (obj, param, (13 + np.sqrt(57)) / 8)
    elif instance == 1:
        # Optimization with x0 = 0 should land in local optimum at x = -1.
        param = DirectParam(np.array([0]), bounds=[-10, 10])
        return (obj, param, -1)


def build_two_variable_quadratic():
    """ Implements f(x) = x^2 + 2y^2 - 5y - 2xy """
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
    return (obj, param, [5 / 2, 5 / 2])


def build_rosenbrock_function(a: float = 1, b: float = 100):
    """ Implements f(x) = (a - x)^2 + b(y - x^2)^2.

    This function has a valley that is easy to get to but hard to find the
    global optimum.
    """
    var = Variable(2)
    x_var = var[0]
    y_var = var[1]

    x_minus_a = Sum([x_var, Constant(-a)])
    y_minus_x2 = Sum([y_var, Product([Constant(-1), x_var, x_var])])
    obj = Sum([
        Product([x_minus_a] * 2),
        Product([Constant(b), Product([y_minus_x2] * 2)])
    ])
    param = DirectParam(np.array([0, 0]), bounds=[-10, 10])
    return (obj, param, [a, a**2])


class PlaneObjective(OptimizationFunction):
    """
        f(x) = -x_0 - x_1 - ... - x_n
    """

    def calculate_gradient(self, param: Parametrization) -> np.array:
        vec = param.get_structure()
        return -np.ones_like(vec)

    def calculate_objective_function(self, param: Parametrization) -> np.array:
        vec = param.get_structure()
        return -np.ones_like(vec) @ vec


class QuadObjective(OptimizationFunction):
    """
        f(x) = x_0**2 + x_1**2 + ... + x_n**2
    """

    def __init__(self, r0: np.array):
        self.r0 = r0

    def calculate_gradient(self, param: Parametrization) -> np.array:
        vec = param.get_structure()
        return 2 * (vec - self.r0)

    def calculate_objective_function(self, param: Parametrization) -> np.array:
        vec = param.get_structure()
        return np.ones_like(vec) @ ((vec - self.r0)**2)


class SphereObjective(OptimizationFunction):
    """
        f(x) = radius**2 - sum_i (x_i - r0_i)**2
    """

    def __init__(self, radius, r0):
        self.radius = radius
        self.r0 = r0

    def calculate_gradient(self, param: Parametrization) -> np.array:
        vec = param.get_structure()
        return -2 * (vec - self.r0)

    def calculate_objective_function(self, param: Parametrization) -> np.array:
        vec = param.get_structure()
        return self.radius**2 - (np.ones_like(vec) @ (vec - self.r0)**2)


def build_constrained_linear_problem(instance: int = 0):
    obj = PlaneObjective()
    cons0 = -SphereObjective(radius=1.0, r0=np.array([0, 0]))
    cons1 = -SphereObjective(radius=1.0, r0=np.array([1, 0]))
    param = DirectParam(np.array([0.5, 0.5]), bounds=(-1, 1))
    if instance == 0:
        cons_eq = [cons0]
        cons_ineq = [cons1]
        res = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    elif instance == 1:
        cons_eq = [cons1]
        cons_ineq = [cons0]
        res = [0.5, np.sqrt(1 - 0.5**2)]
    opt = OptimizationProblem(obj, cons_eq, cons_ineq)
    return (opt, param, res)


def build_constrained_quadratic_problem(instance: int = 0):
    obj = QuadObjective(np.array([-0.5, 0]))
    cons0 = -SphereObjective(radius=1.0, r0=np.array([0, 0]))
    cons1 = -SphereObjective(radius=1.0, r0=np.array([1, 0]))
    param = DirectParam(np.array([0.5, 0.5]), bounds=(-1, 1))
    if instance == 0:
        cons_eq = [cons1]
        cons_ineq = [cons0]
        res = [0, 0]
    elif instance == 1:
        cons_eq = [cons1]
        cons_ineq = []
        res = [0, 0]
    elif instance == 2:
        cons_eq = []
        cons_ineq = [cons0]
        res = [-0.5, 0]
    opt = OptimizationProblem(obj, cons_eq, cons_ineq)
    return (opt, param, res)


def build_constrained_ellipsoidal_problem():
    # Implements f(x) = x^2 + 2y^2 - 5y - 2xy constrainted to x - y >= 1
    var = Variable(2)
    x_var = var[0]
    y_var = var[1]

    obj = x_var**2 + 2 * y_var**2 - 5 * y_var - 2 * x_var * y_var
    cons_ineq = [y_var - x_var + 1]
    opt = OptimizationProblem(obj, cons_ineq=cons_ineq)
    param = DirectParam(np.array([0, 0]), bounds=[-10, 10])
    return (opt, param, [7 / 2, 5 / 2])


def build_constrained_problem_list():
    # Construct a list of "standard" constrained problems.
    prob = []
    prob.append(build_constrained_ellipsoidal_problem())
    for i in range(2):
        prob.append(build_constrained_linear_problem(i))
    for i in range(3):
        prob.append(build_constrained_quadratic_problem(i))
    return prob
