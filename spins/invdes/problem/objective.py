"""Module for defining generic, commonly-used optimization functions.

Optimization functions are intended to be composable, i.e. multiple objective
functions can be easily combined to create another objective function.
For instance, the SumObjective is an objective that is the sum of individual
objectives.
"""

import abc
import concurrent.futures
import numbers
from typing import List, Union

import numpy as np

from spins.invdes.parametrization import Parametrization
# TODO(logansu): Eliminate circular import?
from spins.invdes.problem import graph_executor


class OptimizationFunction(metaclass=abc.ABCMeta):
    """Defines the interface for a differentiable function.

    Formally, the objective function is a function that maps a
    parametrization onto a real value. The objective function should be
    designed such that smaller values are more optimal than larger values.

    Each optimization function must specify its set of input functions on
    which it depends. This information is used to keep track of the
    computational graph, i.e. a graph of all the operations used to compute it.

    `calculate_objective_function` and `calculate_gradient` are used to evaluate
    the function value and the gradient with respect to the parametrization.

    Gradients of this function are computed via reverse-mode autodiff. In order
    to accomplish this, each optimization function needs to define a `eval`,
    which evaluates the functions given its inputs, and a `grad`, which
    evaluates the gradient of the output function with respect to the input
    function.
    """

    def __init__(self,
                 inputs: Union["OptimizationFunction", List[
                     "OptimizationFunction"]] = None,
                 heavy_compute: bool = False) -> None:
        """Initializes a new optimization function.

        Args:
            inputs: A list of inputs that this function depends on. This is used
                to determine the dependencies in the computational graph.
            heavy_compute: If True, marks the node as a heavy compute node.
                Heavy computes nodes will be parallelly executed if possible.
        """
        if inputs is None:
            # Must be old-style function. Ignore setup.
            # TODO(logansu): Add warning.
            return

        if isinstance(inputs, OptimizationFunction):
            inputs = [inputs]

        self._inputs = inputs
        self._heavy_compute = heavy_compute

    @property
    def heavy_compute(self) -> bool:
        """Returns the heavy_compute value, which is True if parallel processing
        is used.
        """
        return self._heavy_compute

    def calculate_gradient(self, param: Parametrization) -> np.ndarray:
        """Computes the gradient with respect to the parametrization.

        Args:
            param: Parametrization.

        Returns:
            The gradient of the objective function.
        """
        return graph_executor.eval_grad(self, param)

    def calculate_objective_function(self, param: Parametrization) -> float:
        """Computes the objective function value.

        Args:
            param: Parametrization.

        Returns:
            A single float representing objective function value.
        """
        return graph_executor.eval_fun(self, param)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Evaluates the function given the values of its inputs.

        Args:
            input_vals: A list of values corresponding to each of its input
                functions (that were passed into the constructor).

        Returns:
            The value of the function evaluated with inputs `inputs_vals`.
        """
        raise NotImplementedError()

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Calculates the gradient.

        Specifically, this method calculates the gradient of the output function
        with respect to this function. The output function is the function on
        which `calculate_gradient` was called by the user.

        Args:
            input_vals: A list of input values corresponding to each of its
                input functions (as specified during initialization).
            out_grads: The gradient of the output function with respect to
                the output function.

        Returns:
            A list of gradients, one corresponding to each input of this
            function. Each gradient should be equal to the gradient of the
            output function with respect to the input function.
        """
        raise NotImplementedError()

    def __add__(self, obj):
        # More efficient to have a + Sum([...]) become Sum([..., a]) than
        # a + Sum([...]) become Sum([a, Sum([....])).
        if isinstance(obj, Sum):
            return obj.__add__(self)
        if isinstance(obj, Constant):
            return Sum([self, obj], parallelize=False)
        if isinstance(obj, numbers.Number):
            return Sum([self, Constant(obj)], parallelize=False)
        if isinstance(obj, np.ndarray):
            return Sum([self, Constant(obj)], parallelize=False)
        if isinstance(obj, OptimizationFunction):
            return Sum([self, obj])
        raise TypeError("Attempting to add unknown object to objective")

    def __mul__(self, obj):
        if isinstance(obj, Product):
            return obj.__mul__(self)
        if isinstance(obj, Constant):
            return Product([self, obj], parallelize=False)
        if isinstance(obj, numbers.Number):
            return Product([self, Constant(obj)], parallelize=False)
        if isinstance(obj, np.ndarray):
            return Sum([self, Constant(obj)], parallelize=False)
        if isinstance(obj, OptimizationFunction):
            return Product([self, obj])
        raise TypeError("Attempting to add unknown object to objective")

    def __pow__(self, obj):
        if isinstance(obj, numbers.Real):
            return Power(self, obj)
        if isinstance(obj, Constant):
            return Power(self, obj.value)
        raise TypeError("Attempting to raise objective to non-constant power.")

    def __radd__(self, obj):
        return self.__add__(obj)

    def __rmul__(self, obj):
        return self.__mul__(obj)

    def __sub__(self, obj):
        return self + (-obj)

    def __rsub__(self, obj):
        return -self + obj

    def __neg__(self):
        return -1 * self


def calculate_objective_parallel(objs: List[OptimizationFunction],
                                 param: Parametrization) -> List[float]:
    """Calculates the objective functions in parallel.

    Parallelization is done with threads so that locking mechanisms will work.

    Args:
        objs: List of objective functions.
        param: The parametrization to use.

    Returns:
        List of objective function values in the order given by objs.
    """

    def calculate_objective(obj: OptimizationFunction) -> float:
        """ Gets subobjective function value. """
        return obj.calculate_objective_function(param)

    if not objs:
        return []
    elif len(objs) == 1:
        return [calculate_objective(objs[0])]

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(objs)) as executor:
        return list(executor.map(calculate_objective, objs))


def calculate_gradient_parallel(objs: List[OptimizationFunction],
                                param: Parametrization) -> List[float]:
    """Calculates the objective function gradients in parallel.

    Parallelization done with threads so that locking mechanisms will work.

    Args:
        objs: List of objective functions.
        param: The parametrization to use.

    Returns:
        List of objective function values in the order given by objs.
    """

    def calculate_gradient(obj: OptimizationFunction) -> float:
        """Gets subobjective gradient. """
        return obj.calculate_gradient(param)

    if not objs:
        return []
    elif len(objs) == 1:
        return [calculate_gradient(objs[0])]

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(objs)) as executor:
        return list(executor.map(calculate_gradient, objs))


class OptimizationProblem:
    """Defines an optimization problem.

    An optimization problem consists of an objective and a list of equality and
    inequality constraints.

    Note that OptimizationProblem implements the same interface as
    OptimizationFunction so OptimizationProblem can be used in all places that
    accept an OptimizationFunction.
    """

    def __init__(self,
                 obj: OptimizationFunction,
                 cons_eq: List[OptimizationFunction] = None,
                 cons_ineq: List[OptimizationFunction] = None):
        """Constructs an optimization problem.

        Args:
            obj: An OptimizationFunction.
            cons_eq: List of equality constraints. Each equality constraint
                is represented as an OptimizationFunction f where the
                constraint is for f = 0.
            cons_ineq: List of inequality constraints. Each inequality
                constraint is represented by an OptimizationFunction f where
                the constraint is for f <= 0.
        """
        self.obj = obj
        if cons_eq is None:
            cons_eq = []
        if cons_ineq is None:
            cons_ineq = []
        self.cons_eq = cons_eq
        self.cons_ineq = cons_ineq

    def calculate_objective_function(self, param):
        """Return the objective function value.
        """
        return self.obj.calculate_objective_function(param)

    def calculate_gradient(self, param):
        """Return the objective function's gradient.
        """
        return self.obj.calculate_gradient(param)

    def calculate_constraints(self, param):
        """Calculates the values of all the constraints.

        Args:
            param: Parametrization.

        Returns:
            A tuple (eq, ineq) where eq is the list of evaluated
            equality constraints and ineq is the list of evaluated
            inequality constraints.
        """
        cons = calculate_objective_parallel(self.cons_eq + self.cons_ineq,
                                            param)
        return (np.array(cons[:len(self.cons_eq)]),
                np.array(cons[len(self.cons_eq):]))

    def calculate_constraint_gradients(self, param):
        """Calculates the gradients of all the constraints.

        Args:
            param: Parametrization.

        Returns:
            A tuple (eq, ineq) where eq is the list of evaluated
            equality constraint gradients and ineq is the list of evaluated
            inequality constraint gradients.
        """
        # Calculates gradients of all the constraints.
        cons = calculate_gradient_parallel(self.cons_eq + self.cons_ineq, param)
        return cons[:len(self.cons_eq)], cons[len(self.cons_eq):]

    def get_objective(self):
        """Return the objective function.

        Returns:
            Objective function

        """
        return self.obj

    def get_equality_constraints(self):
        """Return the equality constraints.

        Returns:
            List with the equality constraints.

        """
        return self.cons_eq

    def get_inequality_constraints(self):
        """Return the inequality constraints.

        Returns:
            List with the inequality constraints.

        """
        return self.cons_ineq


class Variable(OptimizationFunction):
    """Represents a single variable.

    This is used as a placeholder for the computational graph. Note that
    we do not actually have to define `eval` because this node's value will
    always be predetermined by the parametrization (see `graph_executor`
    for details).
    """

    def __init__(self, size: int) -> None:
        """Creates a new variable that represents a vector.

        Args:
            size: The length of the vector.
        """
        super().__init__([])

        self.shape = size

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        return []

    def __getitem__(self, index):
        return ValueSlice(self, index)


class Constant(OptimizationFunction):
    """Represents a single constant.

    ConstantObjective is useful in aggregate objective where a constant
    is desired in place of a function.
    """

    def __init__(self, value) -> None:
        """Initializes a new constant.

        Args:
            value: The value of the constant. The value of the constant is
                   copied.
        """
        super().__init__([])

        self.value = value
        self.value_shape = np.array(self.value).shape

    def calculate_objective_function(self, param: Parametrization) -> float:
        """Returns the value of the constant. """
        return self.value

    def calculate_gradient(self, param: Parametrization) -> float:
        """Returns zero. """
        return np.zeros(self.value_shape + (len(param.to_vector()),))

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the value of the constant. """
        return self.value

    def grad(self, input_vals, grad_val):
        """Returns zero. """
        return [0]

    def __add__(self, obj):
        # Combine constants when possible.
        if isinstance(obj, Constant):
            return Constant(self.value + obj.value)
        if isinstance(obj, numbers.Number):
            return Constant(self.value + obj)
        if isinstance(obj, np.ndarray):
            return Constant(self.value + obj)
        return super().__add__(obj)

    def __mul__(self, obj):
        # Combine constants when possible.
        if isinstance(obj, Constant):
            return Constant(self.value * obj.value)
        if isinstance(obj, numbers.Number):
            return Constant(self.value * obj)
        if isinstance(obj, np.ndarray):
            return Constant(self.value * obj)
        return super().__mul__(obj)

    def __neg__(self):
        return Constant(-self.value)

    def __pow__(self, obj):
        return Constant(self.value**obj)

    def __str__(self):
        return str(self.value)


class Parameter(OptimizationFunction):
    """Represents a single parameter.

    A parameter is a node in the graph whose value can change and be set at
    any time.
    """

    def __init__(self, initial_value: np.ndarray) -> None:
        """Initializes a new Parameter.

        Args:
            initial_value: Parameter value upon initialization.
        """
        super().__init__([])

        self.value = initial_value
        self.value_shape = np.array(self.value).shape

    def set_parameter_value(self, value: np.ndarray) -> None:
        self.value = value

    def calculate_objective_function(self, param: Parametrization) -> float:
        """Returns the value of the parameter. """
        return self.value

    def calculate_gradient(self, param: Parametrization) -> float:
        """Returns zero. """
        return np.zeros(self.value_shape + (len(param.to_vector()),))

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the value of the constant. """
        return self.value

    def grad(self, input_vals, grad_val):
        """Returns zero. """
        return [0]

    def __str__(self):
        return str(self.value)


class ValueSlice(OptimizationFunction):
    """Represents a scalar variable.

    The scalar variable value will be taken from a particular index of the
    parametrization. For example, if the parametrization vector is [1,2,3]
    and ValueSlice has been defined over index 2. Then the value of
    ValueSlice is 3.
    """

    def __init__(self, variable: Variable, index: int) -> None:
        """Configures a scalar variable.

        Args:
            variable: The `Variable` to slice.
            index: The index of the parametrization vector corresponding to
                   the scalar variable.
        """
        super().__init__(variable)

        self._index = index
        self._shape = variable.shape

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return input_vals[0][self._index]

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        grad_res = np.zeros(self._shape)
        grad_res[self._index] = grad_val
        return [grad_res]

    def __str__(self):
        return "p[" + str(self._index) + "]"


class Sum(OptimizationFunction):
    """Represents a sum of objective functions. """

    def __init__(self,
                 objectives: List[OptimizationFunction],
                 weights: np.ndarray = None,
                 parallelize: bool = True) -> None:
        """Represents a sum of objective functions.

        Args:
            objectives: A list of objectives.
            weights: Vector of weight factors for every objective in the
                objectives list.
            parallelize: Parallelize calculations. Unused.
        """
        self.parallelize = parallelize

        if weights is not None:
            self.objectives = []
            for i, obj in enumerate(objectives):
                self.objectives.append(weights[i] * obj)
        else:
            self.objectives = objectives

        super().__init__(self.objectives)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return np.array(sum(input_vals))

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        return [grad_val] * len(input_vals)

    def __add__(self, obj):
        if isinstance(obj, Sum):
            parallelize = self.parallelize or obj.parallelize
            return Sum(
                self.objectives + obj.objectives, parallelize=parallelize)
        if isinstance(obj, OptimizationFunction):
            return Sum(self.objectives + [obj], parallelize=self.parallelize)
        if isinstance(obj, numbers.Number):
            return Sum(
                self.objectives + [Constant(obj)], parallelize=self.parallelize)
        if isinstance(obj, np.ndarray):
            return Sum(
                self.objectives + [Constant(obj)], parallelize=self.parallelize)
        raise TypeError("Attempting to add unknown type to objective function.")

    def __str__(self):
        string = "(" + " + ".join(str(obj) for obj in self.objectives) + ")"
        if self.parallelize:
            string = "||" + string
        return string


class Product(OptimizationFunction):
    """Represents a product of objective functions.

    For matrices, products refer to element-wise product.
    """

    def __init__(self,
                 objectives: List[OptimizationFunction],
                 parallelize: bool = True) -> None:
        """Represents a product of objective functions.

        Args:
            objectives: A list of objectives.
            parallelize: Parallelize calculations.
        """
        super().__init__(objectives)

        self.objectives = objectives
        self.parallelize = parallelize

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        total = 1.0
        for obj in input_vals:
            total *= obj
        return total

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        grads = []
        for i in range(len(input_vals)):
            # Calculate product of everything except ith objective value.
            prod_before = np.prod(input_vals[:i], axis=0)
            prod_after = np.prod(input_vals[i + 1:], axis=0)
            grads.append(prod_before * prod_after * grad_val)
        return grads

    def __mul__(self, obj):
        if isinstance(obj, Product):
            parallelize = self.parallelize or obj.parallelize
            return Product(self.objectives + obj.objectives, parallelize)
        if isinstance(obj, OptimizationFunction):
            return Product(self.objectives + [obj], self.parallelize)
        if isinstance(obj, numbers.Number):
            return Product(self.objectives + [Constant(obj)], self.parallelize)
        if isinstance(obj, np.ndarray):
            return Product(self.objectives + [Constant(obj)], self.parallelize)
        raise TypeError("Attempting to add unknown type to objective function.")

    def __str__(self):
        # Keep the outer parentheses because it indicates that the interior
        # is kept in one Product object.
        string = "({0})".format(" * ".join(str(obj) for obj in self.objectives))
        if self.parallelize:
            string = "||" + string
        return string


class Power(OptimizationFunction):
    """Represents an objective function raised to a constant power."""

    def __init__(self, obj, power):
        """Constructs the objective (obj)**power

        Args:
            obj: Input objective.
            power: A real number power (cannot be a Constant).
        """
        super().__init__(obj)

        self.power = power
        self.obj = obj

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return input_vals[0]**self.power

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        return [grad_val * self.power * input_vals[0]**(self.power - 1)]

    def __str__(self):
        return str(self.obj) + "**" + str(self.power)


# TODO(logansu): Fix gradients here. The fundamental issue is that for
# `f(z) = |z|`, df/dz differs by a factor of 2 depending on whether z is
# considered complex or real. Resolving this issue requires tracking the type
# of z.
#
# Temporarily, we have implemented absolute value function treating the function
# as purely real. In order for the gradients to be correct, any function that
# that maps the reals to the complex numbers must take the real part (instead of
# twice the real part) of `grad_val`. See `AbsTestFunction` in
# `test_objective.py` as an example.
class AbsoluteValue(OptimizationFunction):
    """Objective takes the absolute value of the input."""

    def __init__(self, objective):
        """Initializes absolute value function.

        Args:
            obj: The objective to wrap.
        """
        super().__init__(objective)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return np.abs(input_vals[0])

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        grad = np.conj(input_vals[0]) / abs(input_vals[0])
        return [grad_val * grad]

    def __str__(self):
        return "abs({})".format(self._inputs[0])


class IndicatorPlus(OptimizationFunction):
    """Objective that penalizes the input being larger than some value, alpha.

    The input and range is assumed to be real!!!
    """

    def __init__(self,
                 objective: OptimizationFunction,
                 alpha: float = 0,
                 power: float = 2):
        """Constructs the objective (obj>alpha)*(obj-alpha)**power.

        Args:
            obj: The objective to wrap.
            alpha: Threshold.
            power: Exponent.
        """
        super().__init__(objective)
        self.obj = objective
        self.alpha = alpha
        self.power = power

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return (input_vals[0] >
                self.alpha) * abs(input_vals[0] - self.alpha)**self.power

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        alpha_diff = input_vals[0] - self.alpha

        graph_alpha = (input_vals[0] > self.alpha) * (self.power) * abs(
            alpha_diff)**(self.power - 1) * np.sign(alpha_diff)
        return [grad_val * graph_alpha]

    def __str__(self):
        return "I_plus({0}-{1})**{2}".format(
            str(self.obj), self.alpha, self.power)


class IndicatorMin(OptimizationFunction):
    """Objective that penalizes the input being smaller than some value, beta.

    The input and range is assumed to be real!!!
    """

    def __init__(self,
                 objective: OptimizationFunction,
                 beta: float = 0,
                 power: float = 2):
        """Constructs the objective (obj<beta)*(beta - obj)**power.

        Args:
            obj: The objective to wrap.
            beta: Threshold.
            power: Exponent.
        """
        super().__init__(objective)
        self.obj = objective
        self.beta = beta
        self.power = power

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return (input_vals[0] <
                self.beta) * abs(self.beta - input_vals[0])**self.power

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        beta_diff = self.beta - input_vals[0]

        graph_beta = (input_vals[0] < self.beta) * (self.power) * abs(
            beta_diff)**(self.power - 1) * np.sign(beta_diff) * (-1)
        return [grad_val * graph_beta]

    def __str__(self):
        return "I_min({0}-{1})**{2}".format(self.obj, self.beta, self.power)


class PowerComparison(OptimizationFunction):
    """Objective that penalizes the input for not being in a certain interval.

    The input and range is assumed to be real!!!
    """

    def __init__(self,
                 objective: List[OptimizationFunction],
                 value_range: Union[List[float], np.ndarray],
                 power: float = 1):
        """Constructs the objective:
            (obj<beta)*(beta - obj)**power + (obj>alpha)*(obj-alpha)**power

        Args:
            obj: Input functions.
            value_range: Desired power interval.
            power: Exponent.
        """
        super().__init__(objective)
        self.alpha = value_range[1]
        self.beta = value_range[0]
        self._power = power

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        beta_term = (input_vals[0] <
                     self.beta) * abs(self.beta - input_vals[0])**self._power
        alpha_term = (input_vals[0] >
                      self.alpha) * abs(input_vals[0] - self.alpha)**self._power
        return alpha_term + beta_term

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        alpha_diff = input_vals[0] - self.alpha
        beta_diff = self.beta - input_vals[0]

        graph_alpha = (input_vals[0] > self.alpha) * (self._power) * abs(
            alpha_diff)**(self._power - 1) * np.sign(alpha_diff)
        graph_beta = (input_vals[0] < self.beta) * (self._power) * abs(
            beta_diff)**(self._power - 1) * np.sign(beta_diff) * (-1)
        return [grad_val * (graph_beta + graph_alpha)]

    def __str__(self) -> str:
        return "PowerComp({0}-{1}, power = {2})".format(self.beta, self.alpha,
                                                        self._power)


class LogSumExp(OptimizationFunction):
    """ Implements log-sum-exp function.

    LogSumExp is a smooth approximation to the max function.
    In order to mitigate underflow and overflow issues, the function is
    calculated by first factoring out the maximum x_i.
    """

    def __init__(self, objectives: List[OptimizationFunction]):
        """ Initializes LogSumExp.

        Args:
            objectives: List of objectives to take the max over.
        """
        super().__init__(objectives)
        self.objectives = objectives

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        max_val = np.max(input_vals)
        return max_val + np.log(np.sum(np.exp(np.array(input_vals) - max_val)))

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        max_val = np.max(input_vals)
        denom = np.sum(np.exp(np.array(input_vals) - max_val))
        grads = []
        for i in range(len(self.objectives)):
            grads.append(1 / denom * np.exp(input_vals[i] - max_val))
        return grad_val * np.array(grads)

    def __str__(self):
        return "LogSumExp({0})".format(", ".join(
            str(obj) for obj in self.objectives))


class SoftmaxAverage(OptimizationFunction):
    """Implements averaging scheme where the weights are determined by softmax.

    Specifically, if x is a n-dimensional vector and sigma(x) is the softmax
    function, then the inner product of x and sigma(x) is implemented. This is
    effectively another approximation of the max function.
    """

    def __init__(self, objectives: List[OptimizationFunction]):
        """Initializes SoftmaxAverge.

        Args:
            objectives: List of objectives to take the max over.
        """
        super().__init__(objectives)
        self.objectives = objectives

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        max_val = np.max(input_vals)
        weights = np.exp(input_vals - max_val)
        denom = np.sum(weights)
        return np.dot(input_vals, weights) / denom

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        max_val = np.max(input_vals)
        weights = np.exp(input_vals - max_val)
        num = np.dot(input_vals, weights)
        denom = np.sum(weights)
        grads = []
        for i in range(len(self.objectives)):
            grad_num = ((weights[i] + input_vals[i] * weights[i]) * denom -
                        num * weights[i])
            grad_denom = denom**2
            grads.append(grad_num / grad_denom)
        return grad_val * np.array(grads)

    def __str__(self):
        return "SoftmaxAvg({0})".format(", ".join(
            str(obj) for obj in self.objectives))


class RealPart(OptimizationFunction):
    """Retrieves the real part of the objective.

    This is used to convert a real-valued function that has a complex dtype
    into one with a real-valued dtype.
    """

    def __init__(self, fun: OptimizationFunction) -> None:
        super().__init__(fun)
        self._fun = fun

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return np.real(input_vals[0])

    def grad(self, input_vals: List[np.ndarray], grad_val: np.ndarray) -> List[np.ndarray]:
        return [grad_val]
