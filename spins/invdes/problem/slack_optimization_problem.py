""" Defines functionality for transforming inequalities into equalities.

Given an optimization problem

minimize     f(x)
subject to   g_i(x) = 0
             h_i(x) <= 0

the equivalent slack optimization problem is

minimize     f(x)
subject to   g_i(x) = 0
             h_i(x) + s_i = 0

The main components are:
SlackOptimizationProblem: Wraps the optimization problem into one with slack
                          variables.
SlackParam: Wraps the original parametrization by appending the slack variables.
SlackRemover: Wraps the original functions and strips off slack variables.
SlackVariable: Defines a single slack variable (a single s_i).
"""
import numpy as np

from spins.invdes.parametrization import Parametrization
from spins.invdes.problem import OptimizationFunction, OptimizationProblem


class SlackOptimizationProblem(OptimizationProblem):

    def __init__(self, opt):
        """ Defines an optimization problem with slack variables.

        Args:
            opt: OptimizationProblem
        """
        cons_ineq = opt.get_inequality_constraints()
        self.num_slack = len(cons_ineq)
        cons_eq = []
        # Perform shallow copy of equality constraints.
        for eq in opt.get_equality_constraints():
            cons_eq.append(SlackRemover(eq, self.num_slack))
        # Convert inequality constraints into equalities.
        for i, ineq in enumerate(cons_ineq):
            cons_eq.append(
                SlackRemover(ineq, self.num_slack) +
                SlackVariable(self.num_slack, i))
        super().__init__(
            SlackRemover(opt.get_objective(), self.num_slack), cons_eq=cons_eq)

    def build_param(self, param):
        return SlackParam(param, self.num_slack)


class SlackParam(Parametrization):
    """ Wraps normal parametrization and adds slack variables. """

    def __init__(self, param, num_slack):
        """
        Args:
            param: Parametrization
            num_slack: Number of slack parameters to add to the parametrization.
        """
        self.param = param
        self.slack_variables = np.zeros(num_slack)
        self.num_slack = num_slack

    def project(self):
        self.param.project()

    def get_structure(self):
        return self.param.get_structure()

    def calculate_gradient(self):
        return self.param.calculate_gradient()

    def get_bounds(self):
        param_bounds = self.param.get_bounds()
        if param_bounds is None:
            param_size = len(self.param.to_vector())
            param_bounds = [(None,) * param_size, (None,) * param_size]
        minBounds = param_bounds[0] + self.num_slack * (0,)
        maxBounds = param_bounds[1] + self.num_slack * (None,)
        return minBounds, maxBounds

    def get_param(self):
        return self.param

    def get_slack_variable(self, slack_number: int):
        return self.slack_variables[slack_number]

    def encode(self):
        return np.append(self.param.encode(), self.slack_variables)

    def decode(self, vector: np.ndarray) -> None:
        if self.num_slack > 0:
            self.param.decode(vector[:-self.num_slack])
            self.slack_variables = np.array(vector[-self.num_slack:])
        else:
            self.param.decode(vector)

    def serialize(self):
        return {
            'num_slack': self.num_slack,
            'slack': self.slack_variables.tolist(),
            'wrapped': self.param.serialize()
        }

    def deserialize(self, data):
        self.num_slack = data['num_slack']
        self.slack_variables = np.array(data['slack'])
        self.param.deserialize(data['wrapped'])


class SlackVariable(OptimizationFunction):
    """ Represents a single slack variable. """

    def __init__(self, num_slack, slack_ind):
        """ Defines a slack variable.

        Args:
            num_slack: Number of slack variables in total.
            slack_ind: Index of the slack variable.
        """
        self.num_slack = num_slack
        self.slack_ind = slack_ind

    def calculate_objective_function(self, param):
        return param.get_slack_variable(self.slack_ind)

    def calculate_gradient(self, param):
        gradient = np.zeros(len(param.to_vector()))
        gradient[-self.num_slack + self.slack_ind] = 1
        return gradient


class SlackRemover(OptimizationFunction):
    """ Wraps an objective to drop slack variables from parametrization.

    SlackParam appends additional slack variables at the end of the
    parametrization. SlackRemover strips the slack variables.
    """

    def __init__(self, objective, num_slack):
        """
        Args:
            objective: the optimization function
            num_slack: the total amount of slack variables
        """
        self.obj = objective
        self.num_slack = num_slack

    def calculate_gradient(self, slack_param):
        # Gradient of wrapped objective.
        gradient_x = self.obj.calculate_gradient(slack_param.get_param())
        # Append zeros for slack variables.
        gradient_s = np.zeros(self.num_slack)
        return np.append(gradient_x, gradient_s)

    def calculate_objective_function(self, slack_param):
        return self.obj.calculate_objective_function(slack_param.get_param())
