"""This module aids with backwards-compatibility with SPINS-INVDES."""
from typing import Callable, List, Tuple, Optional, Union

import numpy as np
import numpy.random as rand

from spins import goos
from spins.goos_sim import maxwell
from spins.invdes import parametrization
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


class OldStyleParametrization(goos.Shape, goos.Function):
    """Wraps a `spins.invdes.parametrization.Parametrization`.

    This node essentially makes `spins.invdes.parametrization.Parametrization`
    a function instead of a node that saves state; the actual values are stored
    in another variable.
    """
    node_type = "goos.compat.spins0_2_0.shape.param"

    def __init__(
            self,
            var: goos.Variable,
            param: optplan.Parametrization,
            pos: goos.Function,
            material: goos.material.Material,
            material2: goos.material.Material,
            pixel_size: np.ndarray,
            extents: np.ndarray,
    ) -> None:
        super().__init__([var, pos])
        self._var = var

        self._mat = goos.material.get_material(material)
        self._mat2 = goos.material.get_material(material2)
        self._pixel_size = pixel_size
        self._extents = extents

        self._param_schema = param
        self._param = self.create_param()

        self._value_shape = goos.PixelatedContShapeFlow.get_shape(
            extents, pixel_size)

    def create_param(self) -> parametrization.Parametrization:
        return create_old_param(self._param_schema, self._extents,
                                self._pixel_size)[0]

    def eval_const_flags(
        self, inputs: List[goos.NumericFlow.ConstFlags]
    ) -> goos.PixelatedContShapeFlow.ConstFlags:
        return goos.PixelatedContShapeFlow.ConstFlags(pos=inputs[1].array,
                                                      rot=True,
                                                      array=inputs[0].array)

    def eval(self,
             input_vals: List[goos.NumericFlow]) -> goos.PixelatedContShapeFlow:
        self._param.from_vector(input_vals[0].array.flatten(order="F"))
        values = np.reshape(self._param.get_structure(),
                            self._value_shape,
                            order="F")
        return goos.PixelatedContShapeFlow(pos=input_vals[1].array,
                                           rot=np.zeros(3),
                                           array=values,
                                           material=self._mat,
                                           material2=self._mat2,
                                           pixel_size=self._pixel_size,
                                           extents=self._extents)

    def grad(
        self, input_vals: List[goos.NumericFlow],
        grad_val: goos.PixelatedContShapeFlow.Grad
    ) -> List[goos.NumericFlow.Grad]:
        self._param.from_vector(input_vals[0].array.flatten(order="F"))
        grad = np.array(
            grad_val.array_grad.flatten(order="F")
            @ self._param.calculate_gradient())
        grad = np.reshape(grad, input_vals[0].array.shape, order="F")
        return [
            goos.NumericFlow.Grad(grad),
            goos.NumericFlow.Grad(grad_val.pos_grad)
        ]


def compat_param(param: optplan.Parametrization,
                 initializer: Callable,
                 extents: List[float],
                 pixel_size: List[float],
                 var_name: Optional[str] = None,
                 **kwargs) -> Tuple[goos.Variable, OldStyleParametrization]:
    """Creates a parametrization based on `optplan.Parametrization`.

    Note that some arguments of `optplan.Parametrization` may be ignored.
    Some arguments may need to be omitted (e.g. `optplan.SimulationSpace`).

    Args:
        param: Old style parametrization.
        initializer: A callable that accepts a shape and returns an array with
            that shape. This is used to initialize the parametrization variable.
        extents: Extents of the shape.
        pixel_size: Size of each pixel in the shape.
        var_name: Name to give variable.
        **kwargs: Additional arguments to pass to `OldStyleParametrization`.

    Returns:
        A tuple `(var, eps)` where `var` is the variable controlling the
        parametrization and `eps` is the shape node.
    """
    _, var_shape, lower_bound, upper_bound = create_old_param(
        param, extents, pixel_size)
    var = goos.Variable(initializer(var_shape),
                        name=var_name,
                        lower_bounds=lower_bound,
                        upper_bounds=upper_bound)
    return var, OldStyleParametrization(var=var,
                                        param=param,
                                        extents=extents,
                                        pixel_size=pixel_size,
                                        **kwargs)


def get_param_shape(param: optplan.Parametrization, extents: List[float],
                    pixel_size: List[float]) -> List[int]:
    """Computes the shape of a variable for a given parametrization."""
    return create_old_param(param, extents, pixel_size)[1]


def create_old_param(
    param: optplan.Parametrization, extents: np.ndarray, pixel_size: np.ndarray
) -> Tuple[parametrization.Parametrization, Tuple[int]]:
    if type(param) == optplan.CubicParametrization:
        return create_cubic_param(param, extents, pixel_size)
    elif type(param) == optplan.HermiteLevelSetParametrization:
        return create_hermite_levelset_param(param, extents, pixel_size)
    elif type(param) == optplan.BicubicLevelSetParametrization:
        return create_bicubic_levelset_param(param, extents, pixel_size)
    else:
        raise ValueError("Cannot create parametrization for type {}".format(
            type(param)))


def create_spins_param(params: Union[optplan.CubicParametrization,
                                     optplan.BicubicLevelSetParametrization,
                                     optplan.HermiteLevelSetParametrization],
                       extents: np.ndarray, pixel_size: np.ndarray) -> None:
    if pixel_size[0] != pixel_size[1]:
        raise ValueError("Pixel size must be square in xy-plane.")
    if pixel_size[2] != extents[2]:
        raise ValueError("Pixel must cover entire z-extents.")

    design_dims = goos.PixelatedContShapeFlow.get_shape(extents, pixel_size)

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
        n_x = int(
            np.round((fine_x[-1] + 1 - fine_x[0]) /
                     (periods[0] * undersample)) * periods[0] + 1)
        coarse_x = np.linspace(fine_x[0], fine_x[-1] + 1, n_x)
    else:
        coarse_x = np.arange(-design_dims[0] / 2 - undersample,
                             design_dims[0] / 2 + undersample, undersample)
    coarse_x -= (coarse_x[-1] +
                 coarse_x[0]) / 2  # this is necessary to have correct symmetry

    if periodicity[1]:
        n_y = int(
            np.round((fine_y[-1] + 1 - fine_y[0]) /
                     (periods[1] * undersample)) * periods[1] + 1)
        coarse_y = np.linspace(fine_y[0], fine_y[-1] + 1, n_y)
    else:
        coarse_y = np.arange(-design_dims[1] / 2 - undersample,
                             design_dims[1] / 2 + undersample, undersample)
    coarse_y -= (coarse_y[-1] +
                 coarse_y[0]) / 2  # this is necessary to have correct symmetry

    var_shape = np.array([len(coarse_x), len(coarse_y)])
    # adapt var_shape for periodicity
    var_shape = ((var_shape - np.array(periodicity)) /
                 np.array([max(p, 1) for p in periods])).astype(int)
    # adapt var_shape for symmetry
    for i, s in enumerate(reflection_symmetry):
        if s:
            var_shape[i] = int(np.ceil((var_shape[i] + periodicity[i]) / 2))

    init_val = 0.4 + 0.2 * rand.random(tuple(var_shape)).flatten(order="F")

    # adapt var_shape for periodicity
    for i, s in enumerate(reflection_symmetry):
        if s:
            init_val[i] = int(np.ceil((init_val[i] + periodicity[i]) / 2))

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
                       periods=periods), var_shape, 0, 1


def create_cubic_param(*args, **kwargs):
    param, var_shape, lowerbound, upperbound = create_spins_param(
        *args, **kwargs)
    param.set_k(0)
    return param, param.encode().shape, lowerbound, upperbound


def create_bicubic_levelset_param(*args, **kwargs):
    param, var_shape, _, _ = create_spins_param(*args, **kwargs)
    return param, param.encode().shape, -np.inf, np.inf


def create_hermite_levelset_param(*args, **kwargs):
    param, var_shape, _, _ = create_spins_param(*args, **kwargs)
    # TODO(vcruysse): Use a more sensible variable shape.
    return param, param.encode().shape, -np.inf, np.inf
