from typing import Callable, Dict, Optional, Union

from spins.invdes import optimization as optim
from spins.invdes import parametrization
from spins.invdes import problem
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


class ScipyOptimizer:
    """Minimize a function with `ScipyOptimizer`."""

    # Define some constants for defining event state.
    EVENT_STATE_START = "start"
    EVENT_STATE_OPTIMIZING = "optimizing"
    EVENT_STATE_END = "end"

    EVENT_STATE = "state"
    EVENT_ITERATION = "iteration"

    def __init__(
            self,
            opt_prob: problem.OptimizationProblem,
            logger: workspace.Logger,
            optimizer: str = "L-BFGS",
            monitor_lists: Optional[optplan.ScipyOptimizerMonitorList] = None,
            optimization_options: Optional[Dict] = None) -> None:
        """Initializes transformation for running optimization with scipy.

        This transformation solves an optimization problem using a scipy
        optimizer.

        Args:
            opt_prob: Optimization problem to solve.
            logger: Logger to use.
            optimizer: Name of scipy optimizer to use, e.g. 'L-BFGS'. Should be
                one of those defined for `scipy.optimize`.
            monitor_lists: List of monitors to log.
            optimization_options: Dictionary specifying additional optimizations
                as defined in the scipy documentation. For example, to specify
                maximum iterations for 'L-BFGS', `optimization_options` should
                be set to `{"maxiter": 10}`.
        """
        self._opt_prob = opt_prob
        self.optimizer = optimizer
        if optimization_options is None:
            optimization_options = {}
        else:
            for key, value in optimization_options.items():
                if not value:
                    del optimization_options[key]
        self.optimization_options = optimization_options
        self.monitor_lists = monitor_lists
        if not self.monitor_lists:
            self.monitor_lists = optplan.ScipyOptimizerMonitorList()
        self.logger = logger

    def __call__(self,
                 param: parametrization.Parametrization,
                 event_data: Optional[Dict] = None) -> None:
        """Runs the optimization with the given parametriation.

        Args:
            param: Parametrization that will be optimized.
            event_data: Saved event data. This data is used to restore the
                optimization to given point.
        """
        state_completed = None  # Name of the stage last completed.
        iteration = 0

        # Restore the current state if necessary.
        if event_data:
            state_completed = event_data[ScipyOptimizer.EVENT_STATE]
            if ScipyOptimizer.EVENT_ITERATION in event_data:
                iteration = event_data[ScipyOptimizer.EVENT_ITERATION]

        # Handle the start state, i.e. when monitors are fist saved, if
        # the transformation has not been run before.
        if not state_completed:
            self._start(param)
            state_completed = ScipyOptimizer.EVENT_STATE_START

        # If only the start monitors have been saved or if we are still
        # optimizing, resume the optimization state.
        if (state_completed in (ScipyOptimizer.EVENT_STATE_START,
                                ScipyOptimizer.EVENT_STATE_OPTIMIZING)):
            self._iterate(param, iteration)
            state_completed = ScipyOptimizer.EVENT_STATE_OPTIMIZING

        if state_completed == ScipyOptimizer.EVENT_STATE_OPTIMIZING:
            self._end(param)
            state_completed = ScipyOptimizer.EVENT_STATE_END

    def _start(self, param: parametrization.Parametrization) -> None:
        """Start the optimization.

        This just logs the starting state.

        Args:
            param: Parametrization at the start.
        """
        # Evaluate start monitors.
        transformation_state = {
            ScipyOptimizer.EVENT_STATE: ScipyOptimizer.EVENT_STATE_START
        }
        self.logger.write(
            event=transformation_state,
            param=param,
            monitor_list=self.monitor_lists.start_monitors)

    def _iterate(self, param: parametrization.Parametrization,
                 iteration: int) -> None:
        """Run the optimization.

        This function actually runs the optimization.

        Args:
            param: Parametrization to use to run.
        """
        iter_num = iteration

        # Make a callback function for the optimization that logs the data.

        def _callback(_) -> None:
            nonlocal iter_num

            # Increase iteration count.
            iter_num += 1

            # Log the data.
            transformation_state = {
                ScipyOptimizer.EVENT_ITERATION: iter_num,
                ScipyOptimizer.EVENT_STATE:
                ScipyOptimizer.EVENT_STATE_OPTIMIZING
            }
            self.logger.write(
                event=transformation_state,
                param=param,
                monitor_list=self.monitor_lists.callback_monitors)

        # Run optimization.
        opt_options = self.optimization_options.to_native()
        # If restoring, decrease maximum number of iterations based on
        # number of iterations already performed.
        if "maxiter" in opt_options:
            opt_options["maxiter"] -= iteration

        scipy_optimizer = optim.ScipyOptimizer(
            method=self.optimizer, options=opt_options)

        if self.monitor_lists.callback_monitors:
            scipy_optimizer(opt=self._opt_prob, param=param, callback=_callback)
        else:
            scipy_optimizer(opt=self._opt_prob, param=param)

    def _end(self, param: parametrization.Parametrization) -> None:
        """Ends the optimization.

        This logs the final state.

        Args:
            param: Parametrization.
        """
        # Evaluate the end monitors.
        transformation_state = {
            ScipyOptimizer.EVENT_STATE: ScipyOptimizer.EVENT_STATE_END
        }
        self.logger.write(
            event=transformation_state,
            param=param,
            monitor_list=self.monitor_lists.end_monitors)


@optplan.register_transformation(optplan.ScipyOptimizerTransformation)
def create_scipy_optimizer(params: optplan.ScipyOptimizerTransformation,
                           work: workspace.Workspace) -> ScipyOptimizer:
    # Grab the constraints.
    eq_con_list = None
    if params.constraints_eq:
        eq_con_list = [work.get_object(con) for con in params.constraints_eq]

    ineq_con_list = None
    if params.constraints_ineq:
        ineq_con_list = [
            work.get_object(con) for con in params.constraints_ineq
        ]

    # Make optimization problem.
    opt_prob = problem.OptimizationProblem(
        obj=problem.RealPart(work.get_object(params.objective)),
        cons_eq=eq_con_list,
        cons_ineq=ineq_con_list)

    return ScipyOptimizer(
        opt_prob=opt_prob,
        optimizer=params.optimizer,
        monitor_lists=params.monitor_lists,
        logger=work.logger,
        optimization_options=params.optimization_options)


class PenaltyOptimizer:
    """Minimize a function with `PenaltyOptimizer`."""

    # Define some constants for defining event state.
    EVENT_STATE_START = "start"
    EVENT_STATE_OPTIMIZING = "optimizing"
    EVENT_STATE_END = "end"

    EVENT_STATE = "state"
    EVENT_ITERATION = "iteration"
    EVENT_CYCLE_NUM = "cycle_num"
    EVENT_MU = "mu"

    def __init__(
            self,
            opt_prob: problem.OptimizationProblem,
            logger: workspace.Logger,
            optimizer: str = "L-BFGS",
            monitor_lists: Optional[optplan.ScipyOptimizerMonitorList] = None,
            optimization_options: Optional[Dict] = None) -> None:
        """Initializes transformation for `PenaltyOptimizer`.

        This transformation solves an optimization problem using the
        `PenaltyOptimizer` which solves a constrained optimization problem
        by adding a L1 norm penalty function. For example, the problem

        minimize f(x)
        subject to h(x) = 0

        is solved by minimizing `f(x) + mu |h(x)|` for successively larger
        values of `mu`.

        Args:
            opt_prob: Optimization problem to solve.
            logger: Logger to use.
            optimizer: Name of scipy optimizer to use, e.g. 'L-BFGS'. Should be
                one of those defined for `scipy.optimize`.
            monitor_lists: List of monitors to log.
            optimization_options: Dictionary specifying additional optimizations
                as defined in the scipy documentation. For example, to specify
                maximum iterations for 'L-BFGS', `optimization_options` should
                be set to `{"maxiter": 10}`.
        """
        self._opt_prob = opt_prob
        self.optimizer = optimizer
        if optimization_options is None:
            optimization_options = {}
        else:
            for key, value in optimization_options.items():
                if not value:
                    del optimization_options[key]
        self.optimization_options = optimization_options
        self.monitor_lists = monitor_lists
        if not self.monitor_lists:
            self.monitor_lists = optplan.ScipyOptimizerMonitorList()
        self.logger = logger

    def __call__(self,
                 param: parametrization.Parametrization,
                 event_data: Optional[Dict] = None) -> None:
        """Runs the optimization with the given parametriation.

        Args:
            param: Parametrization that will be optimized.
            event_data: Saved event data.
        """
        state_completed = None

        if event_data:
            state_completed = event_data[PenaltyOptimizer.EVENT_STATE]

        if not state_completed:
            self._start(param)
            state_completed = PenaltyOptimizer.EVENT_STATE_START

        if (state_completed in (PenaltyOptimizer.EVENT_STATE_START,
                                PenaltyOptimizer.EVENT_STATE_OPTIMIZING)):
            self._iterate(param, event_data)
            state_completed = PenaltyOptimizer.EVENT_STATE_OPTIMIZING

        if state_completed == PenaltyOptimizer.EVENT_STATE_OPTIMIZING:
            self._end(param)
            state_completed = PenaltyOptimizer.EVENT_STATE_END

    def _start(self, param: parametrization.Parametrization) -> None:
        """Start the optimization.

        This just logs the starting state.

        Args:
            param: Parametrization at the start.
        """
        # Evaluate start monitors.
        transformation_state = {
            PenaltyOptimizer.EVENT_STATE: PenaltyOptimizer.EVENT_STATE_START
        }
        self.logger.write(
            event=transformation_state,
            param=param,
            monitor_list=self.monitor_lists.start_monitors)

    def _iterate(self,
                 param: parametrization.Parametrization,
                 event_data: Optional[Dict] = None) -> None:
        """Run the optimization.

        This function actually runs the optimization.

        Args:
            param: Parametrization to use to run.
            event_data: Dictionary of last optimization state.
        """
        if not event_data:
            event_data = {}

        iteration = event_data.get(PenaltyOptimizer.EVENT_ITERATION, 0)

        # Make a callback function for the optimization that logs the data.
        def _callback(_) -> None:
            nonlocal iteration

            # Increase iteration count.
            iteration += 1

            # Log the data.
            transformation_state = {
                PenaltyOptimizer.EVENT_ITERATION: iteration,
                PenaltyOptimizer.EVENT_STATE:
                PenaltyOptimizer.EVENT_STATE_OPTIMIZING,
                PenaltyOptimizer.EVENT_CYCLE_NUM: optimizer.cycle_num,
                PenaltyOptimizer.EVENT_MU: optimizer.mu,
            }
            self.logger.write(
                event=transformation_state,
                param=param,
                monitor_list=self.monitor_lists.callback_monitors)

        # TODO(logansu): Adjust number iterations based on iteration number.
        # TODO(vcruysse): Make this work for arbitrary optimizers besides
        # L-BFGS.
        optimizer = optim.PenaltyOptimizer(
            optimizer=None, options=self.optimization_options)
        optimizer.cycle_num = event_data.get(PenaltyOptimizer.EVENT_CYCLE_NUM,
                                             0)
        if PenaltyOptimizer.EVENT_MU in event_data:
            optimizer.mu0 = event_data[PenaltyOptimizer.EVENT_MU]

        if self.monitor_lists.callback_monitors:
            optimizer(opt=self._opt_prob, param=param, callback=_callback)
        else:
            optimizer(opt=self._opt_prob, param=param)

    def _end(self, param: parametrization.Parametrization) -> None:
        """Ends the optimization.

        This logs the final state.

        Args:
            param: Parametrization.
        """
        # Evaluate the end monitors.
        transformation_state = {"state": "end"}
        self.logger.write(
            event=transformation_state,
            param=param,
            monitor_list=self.monitor_lists.end_monitors)


@optplan.register_transformation(optplan.PenaltyTransformation)
def create_penalty_optimizer(params: optplan.ScipyOptimizerTransformation,
                             work: workspace.Workspace) -> ScipyOptimizer:
    # Grab the constraints.
    eq_con_list = None
    if params.constraints_eq:
        eq_con_list = [work.get_object(con) for con in params.constraints_eq]

    ineq_con_list = None
    if params.constraints_ineq:
        ineq_con_list = [
            work.get_object(con) for con in params.constraints_ineq
        ]

    # Make optimization problem.
    opt_prob = problem.OptimizationProblem(
        obj=problem.RealPart(work.get_object(params.objective)),
        cons_eq=eq_con_list,
        cons_ineq=ineq_con_list)

    return PenaltyOptimizer(
        opt_prob=opt_prob,
        optimizer=params.optimizer,
        monitor_lists=params.monitor_lists,
        logger=work.logger,
        optimization_options=params.optimization_options)


class CubicParamSigmoidStrength:
    """Change the sigmoid function used in `CubicParam`.

    `CubicParam` applies a sigmoid function after cubic interpolation
    to make the structure more discrete. This transformation changes the
    parameter in the sigmoid function, effectively changing how discrete the
    structure becomes. In the limit as the value tends to infinity, the sigmoid
    function becomes a step function (i.e. perfectly discrete structure).
    """

    def __init__(
            self,
            value: float = 4,
    ) -> None:
        """Initializes the transformation.

        Args:
          value: Value.
        """
        self._value = value

    def __call__(self,
                 param: parametrization.CubicParam,
                 event_data: Optional[Dict] = None) -> None:
        """Changes the sigmoid strength.

        Args:
            param: Parametrization to change.
        """
        param.set_k(self._value)


@optplan.register_transformation(optplan.CubicParamSigmoidStrength)
def create_change_k_value(
        params: optplan.CubicParamSigmoidStrength,
        unused_work: workspace.Workspace) -> CubicParamSigmoidStrength:
    return CubicParamSigmoidStrength(value=params.value)


@optplan.register_transformation(optplan.HermiteParamFixBorder)
def create_fix_border(params: optplan.HermiteParamFixBorder,
                      unused_work: Optional[workspace.Workspace] = None
                     ) -> Callable[[parametrization.HermiteParam], None]:
    """Creates transformation to fix the Hermite parametrization.

    In order to seamlessly transition from inside the design region to outside
    the design region with fabrication constraints, it is necessary to fix
    the values of the levelset function near the boundaries (i.e. do not allow
    them to optimize). Calling the created transformation sets the number of
    cells along the borders that will be fixed into place.
    """

    def fix_borders(param: parametrization.HermiteParam,
                    unused_event_data: Optional[Dict] = None) -> None:
        param.fix_borders(
            xmin_border=params.border_layers[0],
            xmax_border=params.border_layers[1],
            ymin_border=params.border_layers[2],
            ymax_border=params.border_layers[3])

    return fix_borders


class ContToDiscThresholding:
    """Fixes the border of the Hermite parametrization transformation.

    In order to seamlessly transition from inside the design region to outside
    the design region with fabrication constraints, it is necessary to fix
    the values of the levelset function near the boundaries (i.e. do not allow
    them to optimize). Calling this function sets the number of cells along
    the borders that will be fixed into place.
    """

    def __init__(self,
                 cont_param: parametrization.Parametrization,
                 threshold=float) -> None:
        """Initializes the transformation.

        Args:
            cont_param: Continuous parametrization of which the transformation's
                        parametrization will take the threshold.
        """
        self.cont_param = cont_param
        self.threshold = threshold

    def __call__(self,
                 param: parametrization.Parametrization,
                 event_data: Optional[Dict] = None) -> None:
        """Runs the optimization with the given parametriation.

        Args:
            param: Parametrization to change.
        """
        from spins.invdes.parametrization import levelset_parametrization
        if isinstance(param,
                      levelset_parametrization.BicubicLevelSet) and isinstance(
                          self.cont_param, parametrization.CubicParam):
            param.decode(self.cont_param.encode() - self.threshold)
        elif isinstance(
                param, levelset_parametrization.HermiteLevelSet) and isinstance(
                    self.cont_param, parametrization.HermiteParam):
            p = self.cont_param.encode()
            p[:len(p) // 4] -= self.threshold
            param.decode(p)
        elif isinstance(
                param, levelset_parametrization.HermiteLevelSet) and isinstance(
                    self.cont_param, parametrization.CubicParam):
            vec = self.cont_param.geometry_matrix @ self.cont_param.encode()
            p = param.reverse_geometry_matrix @ param.derivative_matrix @ vec
            p[:len(p) // 4] -= self.threshold
            param.decode(p)
        else:
            raise ValueError("Parameterization do not match.")


@optplan.register_transformation(optplan.ContToDiscThresholding)
def create_cont_to_disc_thresholding(
        params: optplan.ContToDiscThresholding,
        work: Optional[workspace.Workspace] = None
) -> Callable[[parametrization.HermiteParam], None]:
    """Creates transformation that thresholds a cubic parametrization and uses
    this as the levelset function for a Hermite parametrization.

    Args:
        params: Optplan parameters of conttodisc_threshold.
        work: Workspace of the transformation.
    """
    cont_param = work.get_object(params.continuous_parametrization)
    return ContToDiscThresholding(
        cont_param=cont_param, threshold=params.threshold)
