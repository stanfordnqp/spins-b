"""Defines schema for parametrizations."""
from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils


class Initializer(optplan.ProblemGraphNode):
    """Represents a initializer."""


@optplan.register_node_type()
class UniformInitializer(Initializer):
    """Initializes parametrization uniformly at random.

    The parametrization values are initialized element-wise.

    Attributes:
        min_val: Minimum value of distribution.
        max_val: Maximum value of distribution.
    """
    type = schema_utils.polymorphic_model_type("initializer.uniform_random")
    min_val = types.FloatType()
    max_val = types.FloatType()


@optplan.register_node_type()
class NormalInitializer(Initializer):
    """Initializes parametrization using normal distribution.

    The parametrization values are initialized element-wise by a normal
    distribution.

    Attributes:
        mean: Mean value of the normal distribution.
        std: Standard deviation value of the normal distribution.
    """
    type = schema_utils.polymorphic_model_type("initializer.normal")
    mean = types.FloatType()
    std = types.FloatType()


@optplan.register_node_type()
class PixelParametrization(optplan.Parametrization):
    """Defines a `DirectParam`.

    Attributes:
        type: Must be "parametrization.pixel".
        init_method: Initialization condition.
        simulation_space: Simulation space to use parametrization in.
    """
    type = schema_utils.polymorphic_model_type("parametrization.direct")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    init_method = optplan.ReferenceType(Initializer)


@optplan.register_node_type()
class GratingParametrization(optplan.Parametrization):
    """Defines a `GratingParam`.

    The grating is by default initialized to have no structure.

    Attributes:
        type: Must be "parametrization.grating_edge".
        simulation_space: Simulation space to use parametrization in.
        inverted: If `True`, defines a grating where each pair of edges
            correspond to a "hole". See
            `spins.invdes.parametrization.GratingParam` for details.
    """
    type = schema_utils.polymorphic_model_type("parametrization.grating")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    inverted = types.BooleanType(default=False)


@optplan.register_node_type()
class CompositeParametrization(optplan.Parametrization):
    """Defines a composite parametrization.

    Attributes:
        param_list: List of parametrizations in the composite. Note that the
            ordering of the parametrizations should be match the ordering
            of selection matrices in the simulation space.
    """
    type = schema_utils.polymorphic_model_type("parametrization.composite")
    param_list = types.ListType(optplan.ReferenceType(optplan.Parametrization))


@optplan.register_node_type()
class GratingFeatureConstraint(optplan.Function):
    """Defines a feature constraint on `GratingParametrization`.

    Args:
        simulation_space: Used to extract the number of pixels in the design
            region.
        min_feature_size: Minimum feature size in nm.
        boundary_constraint_scale: See `spins.invdes.problem.GratingConstraint`
            for details.
    """
    type = schema_utils.polymorphic_model_type(
        "function.grating_feature_constraint")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    min_feature_size = types.FloatType()
    boundary_constraint_scale = types.FloatType(default=2.0)


@optplan.register_node_type()
class CubicParametrization(optplan.Parametrization):
    """Defines `CubicParametrization`.

    Attributes:
        type: Must be "parametrization.cubic".
        simulation_space: Name of simulation space to reference to generate
            the coarse grid.
        undersample: How much the coarse grid undersamples the rough grid.
        reflection_symmetry: List of booleans corresponding whether the
            structure should be symmetric about the x- and y- axes.
        init_method: Specifications on how to initialize the parametrization.
    """
    type = schema_utils.polymorphic_model_type("parametrization.cubic")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    undersample = types.FloatType()
    init_method = optplan.ReferenceType(Initializer)
    reflection_symmetry = types.ListType(types.BooleanType())
    periods = types.ListType(types.IntType())


@optplan.register_node_type()
class HermiteLevelSetParametrization(optplan.Parametrization):
    """Defines `CubicParametrization`.

    Attributes:
        type: Must be "parametrization.hermite_levelset".
        simulation_space: Name of simulation space to reference to generate
            the coarse grid.
        undersample: How much the coarse grid undersamples the rough grid.
        reflection_symmetry: List of booleans corresponding whether the
            structure should be symmetric about the x- and y- axes.
        init_method: Specifications on how to initialize the parametrization.
    """
    type = schema_utils.polymorphic_model_type(
        "parametrization.hermite_levelset")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    undersample = types.FloatType()
    init_method = optplan.ReferenceType(Initializer)
    reflection_symmetry = types.ListType(types.BooleanType())
    periods = types.ListType(types.IntType())


@optplan.register_node_type()
class BicubicLevelSetParametrization(optplan.Parametrization):
    """Defines `BicubicLevelsetParametrization`.

    Attributes:
        simulation_space: Name of simulation space to reference to generate
            the coarse grid.
        undersample: How much the coarse grid undersamples the rough grid.
        reflection_symmetry: List of booleans corresponding whether the
            structure should be symmetric about the x- and y- axes.
        init_method: Specifications on how to initialize the parametrization.
    """
    type = schema_utils.polymorphic_model_type(
        "parametrization.bicubic_levelset")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    undersample = types.FloatType()
    init_method = optplan.ReferenceType(Initializer)
    reflection_symmetry = types.ListType(types.BooleanType())
    periods = types.ListType(types.IntType())


@optplan.register_node_type()
class DiscretePenalty(optplan.Function):
    """Defines discreteness biasing penalty function."""
    type = schema_utils.polymorphic_model_type("function.discrete_penalty")


@optplan.register_node_type()
class FabricationConstraint(optplan.Function):
    """Defines fabrication constraint penalty function.

    Attributes:
        type: Must be "function.fabrication_constraint"
        minimum_curvature_diameter: Smallest allowed curvature.
        minimum_gap: Smallest allowed gap.
        simulation_space: Simulation space where the fabrication constraint is evaluated.
        oversample: Oversample the fine grid by this value to evaluate the penalty.
        method:
                gap: only applies the gap constraint,
                curv: only apply the curvature constraint,
                gapAndCurve: apply the gap and curvature constraint.
        apply_weights: To meet the constraint you typically need to go for a slightly more
            stringent gap and curvature values. If true then weigth factor are applied that
            take care of this.(default: True)
    """
    type = schema_utils.polymorphic_model_type(
        "function.fabrication_constraint")
    minimum_curvature_diameter = types.FloatType()
    minimum_gap = types.FloatType()
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    method = types.StringType(choices=("gap", "curv", "gap_and_curve"))
    apply_factors = types.BooleanType()
