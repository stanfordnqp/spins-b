from typing import List, Union

import numpy as np
from scipy.ndimage import filters

from spins.invdes import parametrization
from spins.invdes import problem
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


@optplan.register_node(optplan.UniformInitializer)
class UniformDistribution:

    def __init__(self, params: optplan.UniformInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        return np.random.uniform(self._params.min_val, self._params.max_val,
                                 shape)


@optplan.register_node(optplan.NormalInitializer)
class NormalDistribution:

    def __init__(self, params: optplan.NormalInitializer,
                 work: workspace.Workspace) -> None:
        self._params = params

    def __call__(self, shape: List[int]) -> np.ndarray:
        return np.random.normal(self._params.mean, self._params.std, shape)


@optplan.register_node(optplan.PixelParametrization)
def create_pixel_param(
        params: optplan.PixelParametrization,
        work: workspace.Workspace) -> parametrization.DirectParam:
    design_dims = work.get_object(params.simulation_space).design_dims
    init_val = work.get_object(params.init_method)(design_dims)
    return parametrization.DirectParam(init_val.flatten(order="F"))


@optplan.register_node(optplan.GratingParametrization)
def create_grating_param(
        params: optplan.GratingParametrization,
        work: workspace.Workspace) -> parametrization.GratingParam:
    # Only one of the design areas is nonzero. Figure out which one.
    design_dims = work.get_object(params.simulation_space).design_dims
    if design_dims[0] > 1 and design_dims[1] > 1:
        raise ValueError("Grating parametrization should have 1D design "
                         "area, got {}".format(design_dims))

    grating_len = np.max(design_dims)
    return parametrization.GratingParam([],
                                        num_pixels=grating_len,
                                        inverted=params.inverted)


@optplan.register_node(optplan.GratingFeatureConstraint)
def create_grating_feature_constraint(
        params: optplan.GratingFeatureConstraint,
        work: workspace.Workspace) -> problem.GratingFeatureConstraint:
    dx = work.get_object(params.simulation_space).dx
    return problem.GratingFeatureConstraint(
        params.min_feature_size / dx,
        boundary_constraint_scale=params.boundary_constraint_scale)


@optplan.register_node(optplan.CubicParametrization)
@optplan.register_node(optplan.BicubicLevelSetParametrization)
@optplan.register_node(optplan.HermiteLevelSetParametrization)
def create_cubic_or_hermite_levelset(
        params: Union[optplan.CubicParametrization,
                      optplan.HermiteLevelSetParametrization],
        work: workspace.Workspace) -> parametrization.CubicParam:
    design_dims = work.get_object(params.simulation_space).design_dims

    # Calculate periodicity of the parametrization.
    periods = params.periods
    if periods is None:
        periods = np.array([0, 0])
    periodicity = [p != 0 for p in periods]

    # Calculate reflection symmetry.
    reflection_symmetry = params.reflection_symmetry
    if reflection_symmetry is None:
        reflection_symmetry = np.array([0, 0])

    # Make fine grid.
    undersample = params.undersample
    fine_x = np.arange(-design_dims[0] / 2, design_dims[0] / 2)
    fine_y = np.arange(-design_dims[1] / 2, design_dims[1] / 2)
    # Center the grid.
    fine_x -= (fine_x[-1] + fine_x[0]) / 2
    fine_y -= (fine_y[-1] + fine_y[0]) / 2

    # Create the coarse grid.
    if periodicity[0]:
        n_x = int(np.round((fine_x[-1] - fine_x[0]) / undersample) + 1)
        coarse_x = np.linspace(fine_x[0], fine_x[-1] + 1, n_x)
    else:
        coarse_x = np.arange(-design_dims[0] / 2 - undersample,
                             design_dims[0] / 2 + undersample, undersample)
    coarse_x -= (coarse_x[-1] +
                 coarse_x[0]) / 2  # this is necessary to have correct symmetry

    if periodicity[1]:
        n_y = int(np.round((fine_y[-1] - fine_y[0]) / undersample) + 1)
        coarse_y = np.linspace(fine_y[0], fine_y[-1] + 1, n_y)
    else:
        coarse_y = np.arange(-design_dims[1] / 2 - undersample,
                             design_dims[1] / 2 + undersample, undersample)
    coarse_y -= (coarse_y[-1] +
                 coarse_y[0]) / 2  # this is necessary to have correct symmetry

    init_val = work.get_object(
        params.init_method)([len(coarse_x), len(coarse_y)])
    init_val = filters.gaussian_filter(init_val, 1).flatten(order='F')

    # Make parametrization.
    if params.type == "parametrization.hermite_levelset":
        from spins.invdes.parametrization import levelset_parametrization
        param_class = levelset_parametrization.HermiteLevelSet
    elif params.type == "parametrization.bicubic_levelset":
        from spins.invdes.parametrization import levelset_parametrization
        param_class = levelset_parametrization.BicubicLevelSet
    elif params.type == "parametrization.cubic":
        param_class = parametrization.CubicParam
    else:
        raise ValueError("Unexpected parametrization type, got {}".format(
            params.type))

    return param_class(initial_value=init_val,
                       coarse_x=coarse_x,
                       coarse_y=coarse_y,
                       fine_x=fine_x,
                       fine_y=fine_y,
                       symmetry=reflection_symmetry,
                       periodicity=periodicity,
                       periods=periods)


class DiscretePenaltyFun(problem.OptimizationFunction):
    """
    Discrteness penalty function.
    This optimization function is for a term to the objective function of the form
    z*(1-z) where z is a parameterization vector ranging from 0 to 1.
    The role of this term is to bias intermediate values of the parameterization
    towards discrete values 0 or 1.
    """

    def __init__(self, fun: problem.OptimizationFunction) -> None:
        super().__init__(fun)

    def eval(self, inputs: List[np.ndarray]) -> float:
        return np.sum(inputs[0] * (1 - inputs[0]))

    def grad(self, inputs: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        penalty = 1 - 2 * inputs[0]
        return [grad_val * penalty]


@optplan.register_node(optplan.DiscretePenalty)
def create_discrete_penalty(param: optplan.DiscretePenalty,
                            work: workspace.Workspace) -> DiscretePenaltyFun:
    return DiscretePenaltyFun(fun=work.get_object(workspace.VARIABLE_NODE))


class FabricationPenalty(problem.OptimizationFunction):
    """
    Fabrication Penalty objective
    This optimization function evaluates the fabrication penalty of the parametrization
    for a certain fabrication size limit.
    """

    def __init__(self,
                 fcon_gap: float,
                 fcon_curv: float,
                 fabcon_method: int = 2,
                 apply_factors: bool = True):
        '''
        Arg:
            fcon_gap: the smallestallowed gap size.
            fcon_curv: the smallest allowed curvarure diameter.
            fabcon_method:
                0: only applies the gap constraint,
                1: applies the gap and curvature constraint by evaluating the curvature
                    constraint on the border (only available with BicubicLevelSet)
                2: applies the gap and curvature constraint (curvature is evaluated
                    everywhere) (only available with HermiteLevelSet)
            apply_factors: boolean that indiates whether or not you scale up the fcon
                values.
        '''

        self.d_gap = np.pi / fcon_gap
        self.d_curv = np.pi / fcon_curv
        self.method = fabcon_method

        self.d_gap_factor = 1
        self.d_curv_factor = 1
        if apply_factors:
            self.d_gap_factor = 1.2**-1
            self.d_curv_factor = 1.1**-1

    def calculate_objective_function(self, param) -> np.ndarray:
        if self.method == 0:
            penalty = param.calculate_gap_penalty(self.d_gap_factor *
                                                  self.d_gap)
        elif self.method == 1:
            penalty = param.calculate_curv_penalty(self.d_curv_factor *
                                                   self.d_curv)
        elif self.method == 2:
            curv = param.calculate_curv_penalty(self.d_curv_factor *
                                                self.d_curv)
            gap = param.calculate_gap_penalty(self.d_gap_factor * self.d_gap)
            penalty = curv + gap
        else:
            raise ValueError("Fabcon method is invalid.")
        return penalty

    def calculate_gradient(self, param) -> List[np.ndarray]:
        if self.method == 0:
            gradient = param.calculate_gap_penalty_gradient(self.d_gap_factor *
                                                            self.d_gap)
        elif self.method == 1:
            gradient = param.calculate_curv_penalty_gradient(
                self.d_curv_factor * self.d_curv)
        elif self.method == 2:
            curv = param.calculate_curv_penalty_gradient(self.d_curv_factor *
                                                         self.d_curv)
            gap = param.calculate_gap_penalty_gradient(self.d_gap_factor *
                                                       self.d_gap)
            gradient = curv + gap
        else:
            raise ValueError("Fabcon method is invalid.")
        return gradient

    def __str__(self):
        return 'FabCon(' + str(self.d_gap) + ')'


@optplan.register_node(optplan.FabricationConstraint)
def create_fabrication_constraint(
        params: optplan.FabricationConstraint,
        work: workspace.Workspace) -> FabricationPenalty:
    dx = work.get_object(params.simulation_space).dx
    minimum_curvature_diameter = params.minimum_curvature_diameter / (dx / 2)
    minimum_gap = params.minimum_gap / (dx / 2)
    methods = {"gap": 0, "curv": 1, "gap_and_curve": 2}

    return FabricationPenalty(fcon_gap=minimum_gap,
                              fcon_curv=minimum_curvature_diameter,
                              fabcon_method=methods[params.method],
                              apply_factors=params.apply_factors)
