""" This module implements basic gradient-descent-based optimizaiton schemes."""

import copy
import logging

import numpy as np
import scipy.io

from spins.invdes.problem.objective import OptimizationFunction
from spins.invdes.parametrization import Parametrization

import sys

logger = logging.getLogger(__name__)


class GradientOptimizer:
    """ Represents a gradient-based optimizer. """

    def __init__(self):
        self._iters = 0
        self._max_iters = None
        self._param = None

    @property
    def iter(self):
        """ Number of iterations successfully executed. """
        return self._iters

    @property
    def max_iters(self):
        """ Maximum of number of iterations to optimize for. """
        return self._max_iters

    @max_iters.setter
    def max_iters(self, val):
        self._max_iters = val

    @property
    def param(self):
        """ The parametrization used. """
        return self._param

    @param.setter
    def param(self, val):
        self._param = val

    def iterate(self):
        """ Performs one iteration of optimization.

        This is called by optimize() to take a single step.
        A single step is defined by a sequence of operations that
        should be performed before a check on convergence.
        """
        raise NotImplementedException('iterate is not implemented.')

    def optimize(self, iters=None, callback=None):
        """ Runs the optimizer.

        The optimizer will run until a termination condition is met.
        The termination conditions can be one of the following:
        1) max_iters is reached.
        2) An additional iters has be run.

        Args:
            iters: Specifies the maximum number of additional iterations
                to run for. If None, no maximum is specified (but could be
                bounded by max_iters, for example).
            callback: Function called after each iteration.
        """
        # Initialize the stop iteration number.
        # Keep a list of possible stop iterations.
        # Then take the minimum of the possibilities.
        stop_iter_list = []
        if iters:
            stop_iter_list.append(self._iters + iters)
        if self._max_iters:
            stop_iter_list.append(self._max_iters)

        # Compute stopping iteration.
        stop_iter = None
        if stop_iter_list:
            stop_iter = np.min(stop_iter_list)

        while True:
            stop_opt = self.iterate()
            # Increase iteration count and break if done.
            self._iters += 1
            if callback:
                callback(self.param)
            # Check if we should break from iteration count.
            if stop_iter and self._iters >= stop_iter:
                break
            # Check if we should break from other factors.
            if stop_opt:
                break


class GradientDescent(GradientOptimizer):
    """ Vanilla gradient descent. """

    def __init__(self,
                 objective: OptimizationFunction,
                 parametrization: Parametrization,
                 learning_rate: float,
                 normalize_gradient: bool = False):
        """ Initializes gradient descent object.

        Args:
            objective: Objective function to use.
            parametrization: Parametrization.
            learning_rate: Gradient descent rate.
            normalize_gradient: If True, gradient is normalized to have unit
                length.
        """
        super().__init__()
        self.alpha = learning_rate
        self.objective = objective
        self.param = parametrization
        self.normalize_gradient = normalize_gradient
        # Normalizing gradient may result in lack in convergence.
        if self.normalize_gradient:
            logger.warning('Normalizing gradient: Convergence not guaranteed.')

    def iterate(self):
        """ Perform one iteration of gradient descent. """
        logger.debug('Iterating...')

        gradient = self.objective.calculate_gradient(self.param)
        gradient_norm = np.linalg.norm(gradient)

        if self.normalize_gradient:
            gradient /= gradient_norm

        self.param.decode(self.param.encode() - self.alpha * gradient)
        self.param.project()

        logger.debug(('Performed gradient descent step with step size {0} '
                      + 'and gradient norm: {1}').format(
                          self.alpha, gradient_norm))


class AdaptiveGradientDescent(GradientOptimizer):
    """ AdaptiveGradientDescent scales the gradient descent step size
    dynamically.

    For each successful step, the step size is increased, and vice versa.
    Note that AdaptiveGradientDescent will keep shrinking the step size
    until the objective function decreases.

    In pseudo-code:
    while True:
        new_param = param - alpha * gradient
        if obj(new_param) < obj(param):
            param = new_param
            alpha *= success_factor
            break
        else:
            alpha *= failure_factor
    """

    def __init__(self,
                 objective: OptimizationFunction,
                 parametrization: Parametrization,
                 learning_rate: float,
                 success_factor: float = 1.3,
                 failure_factor: float = 0.3,
                 stop_tolerance: float = 1e-8):
        """ Constructs an adaptive gradient descent optimizer.

        Args:
            objective: The objective to optimize.
            parametrization: Parametrization.
            learning_rate: Initial step size.
            success_factor: Factor by which step size should increase in the
                            event of a successful step.
            failure_factor: Factor by which step size should decrease
                in the event of a failed step.
            stop_tolerance: Stop optimization when gradient magnitude drops
                below this value.
        """
        super().__init__()
        self.alpha = learning_rate
        self.objective = objective
        self.param = parametrization
        self.success_factor = success_factor
        self.failure_factor = failure_factor
        self.objective_value = objective.calculate_objective_function(
            self.param)
        self.stop_tolerance = stop_tolerance

    def iterate(self):
        """ Performs a single step.

        The step size keeps shrinking until a step can be taken.
        """
        logger.debug('Iterating...')

        gradient = self.objective.calculate_gradient(self.param)
        gradient_norm = np.linalg.norm(gradient)
        logger.debug('Gradient norm is {0}'.format(gradient_norm))
        if gradient_norm < self.stop_tolerance:
            logger.debug('Hit gradient tolerance. Stopping optimization...')
            return True
        while True:
            # Save the old parametrization in case we want to reset it.
            old_vector = self.param.encode()
            # Calculate the new step.
            new_vector = old_vector - self.alpha * gradient
            self.param.decode(new_vector)
            self.param.project()

            new_objective_value = self.objective.calculate_objective_function(
                self.param)
            if new_objective_value <= self.objective_value:
                logger.debug('Successful step with step size {0}'.format(
                    self.alpha))
                self.objective_value = new_objective_value
                self.alpha *= self.success_factor
                logger.debug('New objective value: {0}'.format(
                    self.objective_value))
                break
            else:
                logger.debug('Step size {0} too big. Shrinking...'.format(
                    self.alpha))
                self.alpha *= self.failure_factor
                self.param.decode(old_vector)


class Adagrad(GradientOptimizer):
    """ Implements AdaGrad algorithm.

    Not to be confused with AdaptiveGradientDescent, the legacy implementation
    of gradient descent, this implements the standard AdaGrad algorithm commonly
    used for SGD.
    """

    def __init__(self, objective: OptimizationFunction,
                 parametrization: Parametrization, learning_rate: float):
        """ Initializes gradient descent object.

        Args:
            objective: Objective function to use.
            parametrization: Parametrization.
            learning_rate: Gradient descent rate.
        """
        super().__init__()
        self.alpha = learning_rate
        self.objective = objective
        self.param = parametrization
        self.historical_gradient = 0

    def iterate(self):
        """ Perform one iteration of gradient descent. """
        logger.debug('Iterating...')

        gradient = self.objective.calculate_gradient(self.param)
        gradient_norm = np.linalg.norm(gradient)

        eps = 1e-6  # Small epsilon to prevent division by zero.
        self.historical_gradient += gradient * gradient
        self.param.from_vector(self.param.to_vector() - self.alpha * gradient /
                               np.sqrt(self.historical_gradient + eps))

        logger.debug('Performed adagrad step with gradient norm: {0}'.format(
            gradient_norm))


class RmsProp(GradientOptimizer):
    """ Implements RMSProp algorithm. """

    def __init__(self,
                 objective: OptimizationFunction,
                 parametrization: Parametrization,
                 learning_rate: float = 0.01,
                 gamma: float = 0.9):
        """ Initializes gradient descent object.

        Args:
            objective: Objective function to use.
            parametrization: Parametrization.
            learning_rate: Gradient descent rate.
            gamma: Retention rate for gradient variance.
        """
        super().__init__()
        self.alpha = learning_rate
        self.objective = objective
        self.param = parametrization
        self.gamma = gamma
        self.grad_var = 0

    def iterate(self):
        """ Perform one iteration of gradient descent. """
        logger.debug('Iterating...')

        gradient = self.objective.calculate_gradient(self.param)
        gradient_norm = np.linalg.norm(gradient)

        eps = 1e-6  # Small epsilon to prevent division by zero.
        self.grad_var = (self.gamma * self.grad_var +
                         (1 - self.gamma) * (gradient * gradient))
        self.param.from_vector(self.param.to_vector() - self.alpha * gradient /
                               np.sqrt(self.grad_var + eps))

        logger.debug('Performed rmsprop step with gradient norm: {0}'.format(
            gradient_norm))


class Adam(GradientOptimizer):
    """ Implements ADAM algorithm. """

    def __init__(self,
                 objective: OptimizationFunction,
                 parametrization: Parametrization,
                 learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999):
        """ Initializes gradient descent object.

        Args:
            objective: Objective function to use.
            parametrization: Parametrization.
            learning_rate: Gradient descent rate.
            beta1: Decay rate for gradient.
            beta2: Decay rate for RMS gradient.
        """
        super().__init__()
        self.alpha = learning_rate
        self.objective = objective
        self.param = parametrization
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad = 0
        self.grad_var = 0

    def iterate(self):
        """ Perform one iteration of gradient descent. """
        logger.debug('Iterating...')

        gradient = self.objective.calculate_gradient(self.param)
        gradient_norm = np.linalg.norm(gradient)

        eps = 1e-8  # Small epsilon to prevent division by zero.
        self.grad = self.beta1 * self.grad + (1 - self.beta1) * gradient
        self.grad_var = (self.beta2 * self.grad_var +
                         (1 - self.beta2) * (gradient * gradient))
        # Unbias gradients.
        grad_corrected = self.grad / (1 - self.beta1**(self.iter + 1))
        grad_var_corrected = self.grad_var / (1 - self.beta2**(self.iter + 1))

        self.param.from_vector(self.param.to_vector() -
                               self.alpha * grad_corrected /
                               (np.sqrt(grad_var_corrected) + eps))

        logger.debug(
            'Performed adam step with gradient norm: {0}'.format(gradient_norm))


class Nag(GradientOptimizer):
    """ Implements Nesterov accelerated gradient algorithm. """

    def __init__(self,
                 objective: OptimizationFunction,
                 parametrization: Parametrization,
                 learning_rate: float = 0.01,
                 gamma: float = 0.9):
        """ Initializes gradient descent object.

        Args:
            objective: Objective function to use.
            parametrization: Parametrization.
            learning_rate: Gradient descent rate.
            gamma: Retention rate for gradient variance.
        """
        super().__init__()
        self.alpha = learning_rate
        self.objective = objective
        self.param = parametrization
        self.gamma = gamma
        self.grad = 0

    def iterate(self):
        """ Perform one iteration of gradient descent. """
        logger.debug('Iterating...')

        temp_param = copy.deepcopy(self.param)
        temp_param.from_vector(self.param.to_vector() - self.gamma * self.grad)
        gradient = self.objective.calculate_gradient(temp_param)
        gradient_norm = np.linalg.norm(gradient)

        self.grad = self.gamma * self.grad + self.alpha * gradient
        self.param.from_vector(self.param.to_vector() - self.grad)

        logger.debug(
            'Performed NAG step with gradient norm: {0}'.format(gradient_norm))
