""" Defines optimizers that handle constraints.

1. PenaltyOptimizer: Optimizes using a power penalty function.
2. AugmentedLagrangianOptimizer: Optimizes using an augmented Lagrangian.

Both optimizers transform the inequality constraints into equality constraints
with slack variables.
"""
import logging
import numpy as np

from spins.invdes.optimization.scipy_optimizer import ScipyOptimizer
from spins.invdes.problem import (OptimizationProblem, SlackOptimizationProblem,
                                  Sum)

logger = logging.getLogger(__name__)


class PenaltyOptimizer:
    """
    For the optimization problem

    minimize   f(x)
    subject to g_i(x) = 0
               h_i(x) <= 0

    the penalty optimizer optimizes
    f(x) + sum_i(g_i(x)**mu) + sum_i((h_i(x)+s_i)**mu)
    where s_i is the slack variable and mu starts from mu0
    and mu := mu**tau at the end of each cycle.
    """

    def __init__(self, optimizer=None, options={}):
        """ Optimizer that uses power penalty function.

        Args:
            optimizer: Optimizer to use.
            options:
                A dictionary with the following possible options:
                    penalty_tol: termination tolerance for complete lagrangian
                    mu0: initial mu, penalty factor
                    tau: exponent with which mu increase every iteration
                    num_cycles: Number of optimization cycles

        TODO(vcruysse): Make the tolerance change improve per iteration.
        """

        self.mu0 = options.get('mu0', 2)
        self.tau = options.get('tau', 1.5)
        self.pf = options.get('pf', 2)
        self.num_cycles = options.get('num_cycles', 10)
        self.cycle_num = 0
        self.maxiter = options.get('maxiter', 30)
        self.ftol = options.get('ftol', 1e-9)
        self.gtol = options.get('gtol', 1e-7)
        self.optimizer = optimizer

    def __call__(self, opt, param, callback=None, cycle_callback=None):
        """ Run optimization.

        Args:
            opt: OptimizationProblem.
            param: Initial parametrization.
            callback: Optional callback.
        Returns:
            Optimized parametrization.
        """
        opt = SlackOptimizationProblem(opt)
        slack_param = opt.build_param(param)
        mu = self.mu0

        for k in range(self.cycle_num, self.num_cycles):
            self.cycle_num = k
            self.mu = mu
            logger.info('Running penalty objective cycle {} with mu {}'.format(
                k, mu))

            # Build penalty objective.
            penalty_obj = opt.get_objective() + mu * Sum(
                [eq**self.pf for eq in opt.get_equality_constraints()])
            penalty_opt = OptimizationProblem(penalty_obj)

            ftol = max(1 / 100**(k + 1), self.ftol)

            if self.optimizer is None:
                optimizer = ScipyOptimizer(
                    method='L-BFGS-B',
                    options={
                        'ftol': ftol,
                        'gtol': self.gtol,
                        'maxiter': self.maxiter
                    })
            else:
                optimizer = self.optimizer

            def aug_callback(x):
                slack_param.from_vector(x)
                logger.info('Current penalty objective: {}'.format(
                    penalty_opt.calculate_objective_function(slack_param)))
                if callback:
                    callback(slack_param.get_param())

            slack_param = optimizer(
                penalty_opt, slack_param, callback=aug_callback)

            res = optimizer.get_results()

            mu = mu**self.tau
            if cycle_callback:
                cycle_callback(slack_param.get_param())
        return slack_param.get_param()


class AugmentedLagrangianOptimizer:

    def __init__(self, options={}):
        """
        Augmented Lagrangian optimization based on Chapter 17 from Nocedal

        Args:
            options: A dictionary of options:
                lagr_tol: termination tolerance for complete lagrangian
                const_tol: termination tolerance for constraints
                mu0: initial mu
                num_cycles: Number of Augmented Lagrangian optimization steps
        """
        # We force L-BFGS-B as the optimizer.
        maxiter = options.get('maxiter', 15)
        self.optimizer = ScipyOptimizer(
            method='L-BFGS-B', options={'maxiter': maxiter})
        self.lagrangian_tolerance_final = options.get('lagr_tol', 1e-7)
        self.constraint_tolerance_final = options.get('cons_tol', 1e-7)
        self.mu0 = options.get('mu0', 10)
        self.mu_exp = options.get('mu_exp', 2)  # 1.5)
        self.num_cycles = options.get('num_cycles', 10)

    def __call__(self, opt, param, callback=None, cycle_callback=None):
        """
        Optimize augmented Lagrangian using L-BFGS-b
        (Nocedal: p520 Algorithm 17.4 (Algorithm implemented in LANCELOT)
             Note however that optimization is done by L-BFGS-B, not by gradient
             projection method.)
        """
        if not isinstance(opt, OptimizationProblem):
            opt = OptimizationProblem(opt)
        opt = SlackOptimizationProblem(opt)
        slack_param = opt.build_param(param)

        # Prepare tolerances and Lagrange variables.
        mu = self.mu0
        lagr_var0 = np.ones(len(opt.get_equality_constraints()))
        lagr_var = lagr_var0.astype(float)
        lagrangian_tolerance = 1 / mu**self.mu_exp  # In Nocedal this is 1/mu.
        constraint_tolerance = 1 / mu**0.1

        # Run optimization.
        for k in range(self.num_cycles):
            logger.info('Running augmented Lagrangian cycle {}'.format(k))

            sgn = -1
            # Build Lagrangian.
            lagrangian_obj = (opt.get_objective() + sgn * Sum(
                opt.get_equality_constraints(),
                weights=lagr_var,
                parallelize=False) + 0.5 * mu * Sum(
                    [eq**2 for eq in opt.get_equality_constraints()],
                    parallelize=False))
            lagrangian = OptimizationProblem(lagrangian_obj)

            def aug_callback(x):
                slack_param.from_vector(x)
                logger.info('Current Augmented Lagrangian objective: {}'.format(
                    lagrangian.calculate_objective_function(slack_param)))

                if callback:
                    callback(slack_param.get_param())

            if self.optimizer.options is None:
                self.optimizer.options = {}
            self.optimizer.options['ftol'] = lagrangian_tolerance
            slack_param = self.optimizer(
                lagrangian, slack_param, callback=aug_callback)

            # Evaluate results and prepare parameters for the next iteration.
            constraints_val, ineq_cons = opt.calculate_constraints(slack_param)
            total_constraint_err = np.sum(np.abs(constraints_val))
            if total_constraint_err < constraint_tolerance:
                # Test if the lagrangian_tolerance reached.
                if (lagrangian_tolerance < self.lagrangian_tolerance_final and
                        total_constraint_err < self.constraint_tolerance_final):
                    break
                lagr_var += sgn * mu * constraints_val
                constraint_tolerance /= mu**0.9
                lagrangian_tolerance /= mu
            else:
                mu *= 100
                constraint_tolerance = 1 / mu**0.1
                lagrangian_tolerance = 1 / mu
            if cycle_callback:
                cycle_callback(slack_param.get_param())
        return slack_param.get_param()
