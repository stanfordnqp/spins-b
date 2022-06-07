from typing import Dict, List

import copy

import numpy as np
import scipy.optimize

from spins import goos
from spins.goos import flows
from spins.goos import graph_executor

import sys


class ScipyOptimizer(goos.Action):
    node_type = "goos.action.optimizer.scipy"

    def __init__(self,
                 objective: goos.Function,
                 method: str,
                 monitor_list: List[goos.ProblemGraphNode] = None,
                 constraints_eq: List[goos.Function] = None,
                 constraints_ineq: List[goos.Function] = None,
                 iteration: goos.Variable = None,
                 max_iters: int = None,
                 ftol: float = None,
                 gtol: float = None,
                 maxls: int = None) -> None:
        """Creates a new optimizer.

        Args:
            objective: Function to minimize.
            method: Optimization method to use (see `scipy.optimize.minimize`)
                for options.
            monitor_list: List of nodes to monitor.
            constraints_eq: List of inequality constraints to apply.
                A constraint `g(x)` is satisfied if `g(x) = 0`.
            constraints_ineq: List of inequality constraints to apply.
                A constraint `g(x)` is satisfied if `g(x) <= 0`.
            iteration: A variable to set that keeps track of the current
                iteration. Each iteration `iteration` will be incremented.
                Ignored if `None`.
            max_iters: Maximum number of iterations to run.
            ftol: Function tolerance convergence criterion.
            gtol: Gradient tolerance convergence criterion.
            maxls: Maximum number of line search steps per iteration. Default is 20.
        """
        if not constraints_ineq:
            constraints_ineq = []
        if not constraints_eq:
            constraints_eq = []

        super().__init__([objective] + constraints_ineq + constraints_eq)
        self._obj = objective
        self._cons_eq = constraints_eq
        self._cons_ineq = constraints_ineq
        self._method = method
        self._monitor_list = monitor_list
        self._iter = iteration
        self.counter = 0

        # Convert options to a dictionary of keyword arguments for
        # `scipy.optimize.minimize`.
        self._options = {"options": {}}
        if max_iters:
            self._options["options"].update({
                "maxiter": max_iters,
            })
        if ftol:
            self._options["options"].update({
                "ftol": ftol,
            })
        if gtol:
            self._options["options"].update({
                "gtol": gtol,
            })
        if maxls:
            self._options["options"].update({
                "maxls": maxls,
            })

    def run(self, plan: goos.OptimizationPlan, start_iter: int = 0):
        variables = plan.get_thawed_vars()

        var_shapes = []
        initial_val = []
        bounds = []
        for var in variables:
            value = plan.get_var_value(var)
            if value.shape:
                var_shapes.append(value.shape)
            else:
                var_shapes.append([1])
            initial_val.append(value.flatten())

            bound = plan.get_var_bounds(var)
            for lower, upper in zip(bound[0].flatten(), bound[1].flatten()):
                if lower == -np.inf:
                    lower = None
                if upper == np.inf:
                    upper = None
                bounds.append((lower, upper))

        override_map = {
            plan._node_map[var_name]: flows.NumericFlow(value)
            for var_name, value in plan._var_value.items()
        }

        # TODO(logansu): Currently we call optimize with every single variable
        # in the plan, but we can really reduce the number of elements by
        # focusing only the variables that are required to compute the objective
        # function.
        def unpack(x):
            cur_ind = 0
            values = []
            for shape in var_shapes:
                values.append(
                    np.reshape(x[cur_ind:cur_ind + np.prod(shape)], shape))
                cur_ind += np.prod(shape)
            return values

        def unpack_and_set(x):
            values = unpack(x)
            for var, value in zip(variables, values):
                #print(f"we set {value.shape} {np.min(value)} {np.max(value)}", file=sys.stderr)
                plan.set_var_value(var, value)

        def unpack_and_set_binarized(x, threshold):
            values = unpack(x)
            for var, value in zip(variables, values):
                #print(f"we set {value.shape} {np.min(value)} {np.max(value)}", file=sys.stderr)
                value = np.array(value > threshold, dtype=value.dtype)
                plan.set_var_value(var, value)

        def func(x):
            unpack_and_set(x)
            val = plan.eval_node(self._obj).array

            # Checking the case with discretization, 16% of iterations.
            self.counter = (self.counter + 1) % 6
            if self.counter == 1:
              if "evergr" in self._method:  # Nevergrad does not need discretization.
                vals = [val] * 8
              else:
                vals = []
                for u in range(1, 9):
                    unpack_and_set_binarized(x, u/9.)
                    vals += [plan.eval_node(self._obj).array]
              print(f"we get value {val} {vals}    ({len(x)})", file=sys.stderr)

            plan.logger.debug("Function evaluated: %f", val)
            unpack_and_set(x)
            return val

        def grad(x):
            unpack_and_set(x)
            grad_flows = plan.eval_grad(self._obj, variables)
            val = np.hstack([flow.array_grad.flatten() for flow in grad_flows])
            plan.logger.debug("Gradient evaluated, norm: %f",
                              np.linalg.norm(val))
            return val

        # To avoid scipy warning, only pass Jacobian to methods that need it.
        methods_that_need_jacobian = {
            "CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP", "dogleg", "trust-ncg"
        }
        jac = None
        if self._method in methods_that_need_jacobian:
            jac = grad

        # Handle constraints.
        methods_that_accept_constraints = {'SLSQP', 'COBYLA'}
        constraints = None
        if self._method in methods_that_accept_constraints:
            constraints = []
            for eq in self._cons_eq:

                def cons_fun(x):
                    unpack_and_set(x)
                    val = plan.eval_node(eq).array
                    plan.logger.debug("Eq. cons. function evaluated: %f", val)
                    return val

                def cons_jac(x):
                    unpack_and_set(x)
                    grad_flows = plan.eval_grad(eq, variables)
                    val = []
                    for flow, var_shape in zip(grad_flows, var_shapes):
                        # Flatten only the dimension corresponding to the
                        # variable.
                        arr = flow.array_grad
                        new_shape = arr.shape[:-len(var_shape)] + (
                            np.prod(var_shape),)
                        val.append(np.reshape(arr, new_shape))

                    val = np.hstack(val)
                    plan.logger.debug("Eq. cons. gradient evaluated, norm: %f",
                                      np.linalg.norm(val))
                    return val

                constraints.append({
                    "type": "eq",
                    "fun": cons_fun,
                    "jac": cons_jac
                })

            for ineq in self._cons_ineq:
                # Note the negative sign because of opposite convention
                # for inequalities (f >= 0 vs f <= 0).
                def cons_fun(x):
                    unpack_and_set(x)
                    val = plan.eval_node(ineq).array
                    plan.logger.debug("Ineq. cons. function evaluated: %f", val)
                    return -val

                def cons_jac(x):
                    unpack_and_set(x)
                    grad_flows = plan.eval_grad(ineq, variables)
                    val = []
                    for flow, var_shape in zip(grad_flows, var_shapes):
                        # Flatten only the dimension corresponding to the
                        # variable.
                        arr = flow.array_grad
                        new_shape = arr.shape[:-len(var_shape)] + (
                            np.prod(var_shape),)
                        val.append(np.reshape(arr, new_shape))

                    val = np.hstack(val)
                    plan.logger.debug(
                        "Ineq. cons. gradient evaluated, norm: %f",
                        np.linalg.norm(val))
                    return -val

                constraints.append({
                    "type": "ineq",
                    "fun": cons_fun,
                    "jac": cons_jac
                })
        elif (len(self._cons_ineq) > 0) or (len(self._cons_eq) > 0):
            plan.logger.warning(
                "Using optimizer that cannot handle constraints. Constraints "
                "ignored: %d" % (len(self._cons_ineq) + len(self._cons_eq)))

        # Keep track of iteration number.
        iter_num = start_iter

        def callback(x):
            # Update the variable values before evaluating monitors.
            values = unpack(x)
            for var, value in zip(variables, values):
                plan.set_var_value(var, value)

            # Update iteration number.
            nonlocal iter_num
            iter_num += 1
            if self._iter:
                plan.set_var_value(self._iter, iter_num)

            plan.write_event({
                "state": "optimizing",
                "iteration": iter_num
            }, self._monitor_list)

        # Adjust total number of iterations if we are resuming.
        options = copy.deepcopy(self._options)
        if "maxiter" in options:
            options["maxiter"] -= start_iter

        initial_val = np.hstack(initial_val)
        if self._method == "nevergrad":
            try:
                import nevergrad as ng
            except:
                assert False, "Please install nevergrad! << pip install nevergrad >>"
            arr = ng.p.TransitionChoice([0, 1], repetitions=len(initial_val))
            print(f"working in dimension {len(initial_val)}")
            value = ng.optimizers.registry["NGOpt"](arr, budget=60).minimize(func).value
            unpack_and_set(value)
        else:
            self._results = scipy.optimize.minimize(func,
                                                    initial_val,
                                                    method=self._method,
                                                    jac=jac,
                                                    callback=callback,
                                                    bounds=bounds,
                                                    constraints=constraints,
                                                    **options)
            obtained_data = self._results["x"]
            dec = [np.quantile(obtained_data, i/10.) for i in range(1,10)]
            print(f"lbfgsb obtains values in {min(obtained_data)} -- {max(obtained_data)}, with deciles {dec}, and length {len(obtained_data)}")
            unpack_and_set(self._results["x"])

    def resume(self, plan: goos.OptimizationPlan, event: Dict) -> None:
        self.run(plan, start_iter=event["iteration"])


def scipy_minimize(objective: goos.Function, *args, **kwargs) -> ScipyOptimizer:
    optimizer = ScipyOptimizer(objective, *args, **kwargs)
    goos.get_default_plan().add_action(optimizer)
    return optimizer


def scipy_maximize(objective: goos.Function, *args, **kwargs) -> ScipyOptimizer:
    return scipy_minimize(-objective, *args, **kwargs)
