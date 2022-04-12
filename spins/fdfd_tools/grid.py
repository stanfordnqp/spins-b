"""
Functions for creating stretched coordinate PMLs.
"""
import copy
import itertools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spins import fdfd_tools

s_function_type = Callable[[float], float]


def prepare_s_function(ln_R: float = -16, m: float = 4) -> s_function_type:
    """Create an s_function to pass to the SCPML functions.

    This is used when you would like to customize the PML parameters.

    Args:
        ln_R: Natural logarithm of the desired reflectance.
        m: Polynomial order for the PML (imaginary part increases as
            `distance**m`).

    Returns:
        An s_function, which takes an ndarray (distances) and returns an ndarray
        (complex part of the cell width; needs to be divided by
         `sqrt(epilon_effective) * real(omega)` before use).
    """

    def s_factor(distance: np.ndarray) -> np.ndarray:
        s_max = (m + 1) * ln_R / 2  # / 2 because we assume periodic boundaries
        return s_max * (distance**m)

    return s_factor


def uniform_grid_scpml(
        shape: np.ndarray or List[int],
        thicknesses: np.ndarray or List[int],
        omega: float,
        epsilon_effective: float = 1.0,
        s_function: s_function_type = None,
) -> fdfd_tools.GridSpacing:
    """
    Create dx arrays for a uniform grid with a cell width of 1 and a pml.

    If you want something more fine-grained, check out stretch_with_scpml(...).

    :param shape: Shape of the grid, including the PMLs (which are 2*thicknesses thick)
    :param thicknesses: [th_x, th_y, th_z] Thickness of the PML in each direction.
        Both polarities are added.
        Each th_ of pml is applied twice, once on each edge of the grid along the given axis.
        th_* may be zero, in which case no pml is added.
    :param omega: Angular frequency for the simulation
    :param epsilon_effective: Effective epsilon of the PML. Match this to the material
        at the edge of your grid.
        Default 1.
    :param s_function: s_function created by prepare_s_function(...), allowing
        customization of pml parameters.
        Default uses prepare_s_function() with no parameters.
    :return: Complex cell widths (dx_lists)
    """
    if s_function is None:
        s_function = prepare_s_function()

    # Normalized distance to nearest boundary
    def l(u, n, t):
        return ((t - u).clip(0) + (u - (n - t)).clip(0)) / t

    dx_a = [np.array(np.inf)] * 3
    dx_b = [np.array(np.inf)] * 3

    # divide by this to adjust for epsilon_effective and omega
    s_correction = np.sqrt(epsilon_effective) * np.real(omega)

    for k, th in enumerate(thicknesses):
        s = shape[k]
        if th > 0:
            sr = np.arange(s)
            dx_a[k] = 1 + 1j * s_function(l(sr, s, th)) / s_correction
            dx_b[k] = 1 + 1j * s_function(l(sr + 0.5, s, th)) / s_correction
        else:
            dx_a[k] = np.ones((s,))
            dx_b[k] = np.ones((s,))
    return [dx_a, dx_b]


def stretch_with_scpml(
        dxes: fdfd_tools.GridSpacing,
        axis: int,
        polarity: int,
        omega: float,
        epsilon_effective: float = 1.0,
        thickness: int = 10,
        s_function: s_function_type = None,
) -> fdfd_tools.GridSpacing:
    """
        Stretch dxes to contain a stretched-coordinate PML (SCPML) in one direction along one axis.

        :param dxes: dx_tuple with coordinates to stretch
        :param axis: axis to stretch (0=x, 1=y, 2=z)
        :param polarity: direction to stretch (-1 for -ve, +1 for +ve)
        :param omega: Angular frequency for the simulation
        :param epsilon_effective: Effective epsilon of the PML. Match this to the material at the
            edge of your grid. Default 1.
        :param thickness: number of cells to use for pml (default 10)
        :param s_function: s_function created by prepare_s_function(...), allowing customization
            of pml parameters. Default uses prepare_s_function() with no parameters.
        :return: Complex cell widths
    """
    if s_function is None:
        s_function = prepare_s_function()

    dx_ai = dxes[0][axis].astype(complex)
    dx_bi = dxes[1][axis].astype(complex)

    pos = np.hstack((0, dx_ai.cumsum()))
    pos_a = (pos[:-1] + pos[1:]) / 2
    pos_b = pos[:-1]

    # divide by this to adjust for epsilon_effective and omega
    s_correction = np.sqrt(epsilon_effective) * np.real(omega)

    if polarity > 0:
        # front pml
        bound = pos[thickness]
        d = bound - pos[0]

        def l_d(x):
            return (bound - x) / (bound - pos[0])

        slc = slice(thickness)

    else:
        # back pml
        bound = pos[-thickness - 1]
        d = pos[-1] - bound

        def l_d(x):
            return (x - bound) / (pos[-1] - bound)

        if thickness == 0:
            slc = slice(None)
        else:
            slc = slice(-thickness, None)

    dx_ai[slc] *= 1 + 1j * s_function(l_d(pos_a[slc])) / d / s_correction
    dx_bi[slc] *= 1 + 1j * s_function(l_d(pos_b[slc])) / d / s_correction

    dxes[0][axis] = dx_ai
    dxes[1][axis] = dx_bi

    return dxes


def generate_periodic_dx(pos: List[np.ndarray]) -> fdfd_tools.GridSpacing:
    """
    Given a list of 3 ndarrays cell centers, creates the cell width parameters for a periodic grid.

    :param pos: List of 3 ndarrays of cell centers
    :return: (dx_a, dx_b) cell widths (no pml)
    """
    if len(pos) != 3:
        raise Exception('Must have len(pos) == 3')

    dx_a = [np.array(np.inf)] * 3
    dx_b = [np.array(np.inf)] * 3

    for i, p_orig in enumerate(pos):
        p = np.array(p_orig, dtype=float)
        if p.size != 1:
            p_shifted = np.hstack((p[1:], p[-1] + (p[1] - p[0])))
            dx_a[i] = np.diff(p)
            dx_b[i] = np.diff((p + p_shifted) / 2)
    return dx_a, dx_b


def make_nonuniform_grid(SimBorders: List[int],
                         dx_default: List[int],
                         Boxes: List[dict],
                         grad_Mesh=0.05,
                         step=1.0) -> (np.array, np.array, np.array):
    '''
    make_nonuniform_grid makes x, y, z vector for a non-uniform grid. In
    addition to the simulation boundaries and the default dx, you can add boxes
    where you want a finer mesh. The mesh will change gradually by grad_Mesh.
    (a grad_Mesh larger then 1 does not make any sense)

    input:
        - SimBorders: the boundaries of your simulation in the form
                        [xmin xmax ymin ymax zmin zmax]
        - dx_default: the largest mesh allowed (a 3 element np.array)
        - Boxes: List of dicts that define finer mesh boxes
            These have 'pos' (a 3 element np.array), 'size' (a 3
            element np.array) and 'meshsize' the meshsize
        - The grad_Mesh (by default 0.05)
        - step: the minimum mesh size is first calculated on a fine grid in
            the x,y and z direction. Step is the mesh size of this vector. It
            should be significantly smaller than the mesh size of the boxes

    output:
        - xs: mesh spacing along the x direction
        - ys: mesh spacing along the y direction
        - zs: mesh spacing along the z direction

    (Dries Vercruysse)
    '''

    # make x, y, z vectors with a step  with a step pitch and the dx, dy, dz vectors specifying

    # the default mesh size
    NX = int((np.ceil(SimBorders[1]) - np.floor(SimBorders[0])) / step)
    NY = int((np.ceil(SimBorders[3]) - np.floor(SimBorders[2])) / step)
    NZ = int((np.ceil(SimBorders[5]) - np.floor(SimBorders[4])) / step)
    x = np.linspace(np.floor(SimBorders[0]), np.ceil(SimBorders[1]), NX + 1)
    y = np.linspace(np.floor(SimBorders[2]), np.ceil(SimBorders[3]), NY + 1)
    z = np.linspace(np.floor(SimBorders[4]), np.ceil(SimBorders[5]), NZ + 1)
    dx = dx_default[0] * np.ones((1, NX + 1))
    dy = dx_default[0] * np.ones((1, NY + 1))
    dz = dx_default[0] * np.ones((1, NZ + 1))

    # define a function that makes a dx vector with DX in between x0 and xn and
    # that increases outside of the [x0, xn] with grad_mesh
    def MeshBox(x, x0, xn, DX, grad_Mesh):
        dx = DX * np.ones_like(x)
        dx[x < x0] += grad_Mesh * (x0 - x[x < x0]) + DX
        dx[x > xn] += grad_Mesh * (x[x > xn] - xn) + DX
        return np.expand_dims(dx, axis=0)

    # for every box element make the dx, dy, dz vector with MeshBox and append
    # it to the existing dx, dy and dz vector
    for box in Boxes:
        x0 = box['pos'][0] - box['size'][0] / 2
        xn = box['pos'][0] + box['size'][0] / 2
        y0 = box['pos'][1] - box['size'][1] / 2
        yn = box['pos'][1] + box['size'][1] / 2
        z0 = box['pos'][2] - box['size'][2] / 2
        zn = box['pos'][2] + box['size'][2] / 2
        dx = np.append(
            dx, MeshBox(x, x0, xn, box['meshsize'][0], grad_Mesh), axis=0)
        dy = np.append(
            dy, MeshBox(y, y0, yn, box['meshsize'][1], grad_Mesh), axis=0)
        dz = np.append(
            dz, MeshBox(z, z0, zn, box['meshsize'][2], grad_Mesh), axis=0)

    # take the minimum of all the dx vectors
    dxv = np.amin(dx, axis=0)
    dyv = np.amin(dy, axis=0)
    dzv = np.amin(dz, axis=0)

    # make the mesh: start at the simulation border and take step with the mesh
    # size give at that point
    xs = [SimBorders[0]]
    while xs[-1] < SimBorders[1]:
        xs = xs + [xs[-1] + dxv[int((xs[-1] - x[0]) // step)]]
    xs = (xs / np.max(xs) * SimBorders[1])
    ys = [SimBorders[2]]
    while ys[-1] < SimBorders[3]:
        ys = ys + [ys[-1] + dyv[int((ys[-1] - y[0]) // step)]]
    ys = (ys / np.max(ys) * SimBorders[3])
    zs = [SimBorders[4]]
    while zs[-1] < SimBorders[5]:
        zs = zs + [zs[-1] + dzv[int((zs[-1] - z[0]) // step)]]
    zs = (zs / np.max(zs) * SimBorders[5])

    # return results
    return xs, ys, zs


def apply_scpml(dxes: fdfd_tools.GridSpacing,
                pml_layers: Optional[Union[int, fdfd_tools.PmlLayers]],
                omega: float) -> fdfd_tools.GridSpacing:
    """Applies PMLs to the grid spacing.

    This function implements SC-PMLs by modifying the grid spacing based on
    the PML layers.

    Args:
        dxes: Grid spacing to modify.
        pml_layers: Indicates number of PML layers to apply on each side. If
            `None`, no PMLs are applied. If this is a scalar, the same number of
            PML layers are applied to each side.
        omega: Frequency of PML operation.

    Returns:
        A new grid spacing with SC-PML applied.
    """
    # Make a copy of `dxes`. We write this out so that we have an array of
    # array of numpy arrays.
    dxes = [
        [np.array(dxes[grid_num][i]) for i in range(3)] for grid_num in range(2)
    ]

    if not pml_layers:
        return dxes

    if isinstance(pml_layers, int):
        pml_layers = [pml_layers] * 6

    for pml, (axis, polarity) in zip(pml_layers,
                                     itertools.product(range(3), [1, -1])):
        if pml > 0:
            dxes = stretch_with_scpml(
                dxes, omega=omega, axis=axis, polarity=polarity, thickness=pml)

    return dxes
