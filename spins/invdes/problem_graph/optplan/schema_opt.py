"""Defines schema for optimization-related nodes."""
from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils


class ScipyOptimizerOptions(schema_utils.Model):
    """Defines an optimizer carried out by `ScipyOptimizer`.

    Attributes:
        tol: (As explained in the scipy minimize documentation)
        maxcor: (As explained in the scipy minimize documentation)
        ftol: (As explained in the scipy minimize documentation)
        gtol: (As explained in the scipy minimize documentation)
        eps: (As explained in the scipy minimize documentation)
        maxfun: (As explained in the scipy minimize documentation)
        maxiter: (As explained in the scipy minimize documentation)
        maxls: (As explained in the scipy minimize documentation)
    """
    tol = types.FloatType()
    maxcor = types.IntType()
    ftol = types.FloatType()
    gtol = types.FloatType()
    eps = types.FloatType()
    maxfun = types.IntType()
    maxiter = types.IntType()
    maxls = types.IntType()


class PenaltyOptimizerOptions(schema_utils.Model):
    """Defines an optimizer carried out by `PenaltyOptimizer`.

    Attributes:
        mu0: initial mu, i.e. the weight factor for the penalty term.
        tau: exponent by which mu is increased.
        pf: exponent over the penalty function.
        num_cycles: number of suboptimization with an increased mu.
        ftol: (As explained in the scipy minimize documentation)
        gtol: (As explained in the scipy minimize documentation)
        maxiter: maximum iteration in one suboptimization.
    """
    mu0 = types.FloatType()
    tau = types.FloatType()
    pf = types.FloatType()
    num_cycles = types.IntType()
    maxiter = types.IntType()
    ftol = types.FloatType()
    gtol = types.FloatType()


class ScipyOptimizerMonitorList(schema_utils.Model):
    """Defines an optimizer carried out by `ScipyOptimizer`.

    Attributes:
        callback_monitors: monitors evaluated every iteration
        start_monitors: monitors evaluated at the transformation start
        end_monitors: monitors evaluated at the transformation end

    """
    callback_monitors = types.ListType(optplan.ReferenceType(optplan.Monitor))
    start_monitors = types.ListType(optplan.ReferenceType(optplan.Monitor))
    end_monitors = types.ListType(optplan.ReferenceType(optplan.Monitor))


@optplan.register_node_type(optplan.NodeMetaType.TRANSFORMATION)
class ScipyOptimizerTransformation(optplan.TransformationBase):
    """Defines an optimizer carried out by `ScipyOptimizer`.

    Attributes:
        type: Must be "scipy_optimizer".
        optimizer: Name of optimizer.
        objective: Name of objective function.
        constraints_eq: List of names of equality constraint functions.
        constraints_ineq: List of names of inequality constraint functions.
        monitor_lists: List of names of monitors to trigger at certain events.
        optimization_options: Options to use for the optimization.
    """
    type = schema_utils.polymorphic_model_type("scipy_optimizer")
    optimizer = types.StringType()
    objective = optplan.ReferenceType(optplan.Function)
    constraints_eq = types.ListType(optplan.ReferenceType(optplan.Function))
    constraints_ineq = types.ListType(optplan.ReferenceType(optplan.Function))
    monitor_lists = types.ModelType(ScipyOptimizerMonitorList)
    optimization_options = types.ModelType(ScipyOptimizerOptions)


@optplan.register_node_type(optplan.NodeMetaType.TRANSFORMATION)
class PenaltyTransformation(optplan.TransformationBase):
    """Defines an optimizer carried out by `PenaltyOptimizer`.

    Attributes:
        type: Must be "penalty_optimizer".
        optimizer: Name of optimizer.
        objective: Name of objective function.
        constraints_eq: List of names of equality constraint functions.
        constraints_ineq: List of names of inequality constraint functions.
        monitor_lists: List of names of monitors to trigger at certain events.
        optimization_options: Options to use for the optimization.
    """
    type = schema_utils.polymorphic_model_type("penalty_optimizer")
    optimizer = types.StringType()
    objective = optplan.ReferenceType(optplan.Function)
    constraints_eq = types.ListType(optplan.ReferenceType(optplan.Function))
    constraints_ineq = types.ListType(optplan.ReferenceType(optplan.Function))
    monitor_lists = types.ModelType(ScipyOptimizerMonitorList)
    optimization_options = types.ModelType(PenaltyOptimizerOptions)


@optplan.register_node_type(optplan.NodeMetaType.TRANSFORMATION)
class CubicParamSigmoidStrength(optplan.TransformationBase):
    """Changes the strength of the sigmoid function in `CubicParametrization`.

    `CubicParametrization` applies a sigmoid function after cubic interpolation
    to make the structure more discrete. This transformation changes the
    parameter in the sigmoid function, effectively changing how discrete the
    structure becomes. In the limit as the value tends to infinity, the sigmoid
    function becomes a step function (i.e. perfectly discrete structure).

    Attributes:
        value: Value for sigmoid function.
    """
    type = schema_utils.polymorphic_model_type("cubic_param.sigmoid_strength")
    value = types.FloatType(default=4)


@optplan.register_node_type(optplan.NodeMetaType.TRANSFORMATION)
class HermiteParamFixBorder(optplan.TransformationBase):
    """Defines parametrization to fix the Hermite parametrization border.

    In order to seamlessly transition from inside the design region to outside
    the design region with fabrication constraints, it is necessary to fix
    the values of the levelset function near the boundaries (i.e. do not allow
    them to optimize). Calling the created transformation sets the number of
    cells along the borders that will be fixed into place.

    Attributes:
        type: Must be "fix_borders".
        border_layer: List with the number of layer to fix at the edge of the
            design area. [#xmin, #xmax, #ymin, #ymax]
    """
    type = schema_utils.polymorphic_model_type("hermite_param.fix_borders")
    border_layers = types.ListType(types.IntType())


@optplan.register_node_type(optplan.NodeMetaType.TRANSFORMATION)
class GratingEdgeFitTransformation(optplan.TransformationBase):
    """Defines the discretization procedure for gratings.

    Specifically, this will convert any epsilon description into a
    `GratingEdgeParametrization`.

    Attributes:
        parametrization: Parametrization to match structure to.
        min_feature: Minimum feature size in terms of number of pixels. Can be
            fractional.
    """
    type = schema_utils.polymorphic_model_type(
        "grating_edge_fit_transformation")
    parametrization = optplan.ReferenceType(optplan.Parametrization)
    min_feature = types.FloatType()

@optplan.register_node_type(optplan.NodeMetaType.TRANSFORMATION)
class ContToDiscThresholding(optplan.TransformationBase):
    """Defines a transformation that takes a continuous parametrization and
    thresholds it at a value.

    Attributes:
        type: Must be "cont_to_disc_thresholding".
        value: Threshold value.

    Note that this requests requires the parametrization to have the same
    parametrization  vector size, e.g. cubic to bicubic or hermiteparam
    to hermitelevelset.
    """
    type = schema_utils.polymorphic_model_type("cont_to_disc_thresholding")
    continuous_parametrization = optplan.ReferenceType(optplan.Parametrization)
    threshold = types.FloatType()
