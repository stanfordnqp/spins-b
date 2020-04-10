"""Code to make free space sources."""
import copy
from typing import Callable, List

import numpy as np

from spins import fdfd_tools
from spins.fdfd_solvers import waveguide_mode
from spins.fdfd_tools import vec, unvec
from spins.fdfd_tools import operators, functional
from spins.gridlock import Grid

COS = np.cos
SIN = np.sin
PI = np.pi


def rotation_matrix(vec: np.ndarray, angle: float) -> np.ndarray:
    """Matrix rotates around the vector.
    """
    R = np.array([[
        COS(angle) + vec[0]**2 * (1 - COS(angle)),
        vec[0] * vec[1] * (1 - COS(angle)) - vec[2] * SIN(angle),
        vec[0] * vec[2] * (1 - COS(angle)) + vec[1] * SIN(angle)
    ],
                  [
                      vec[0] * vec[1] * (1 - COS(angle)) + vec[2] * SIN(angle),
                      COS(angle) + vec[1]**2 * (1 - COS(angle)),
                      vec[1] * vec[2] * (1 - COS(angle)) - vec[0] * SIN(angle)
                  ],
                  [
                      vec[0] * vec[2] * (1 - COS(angle)) - vec[1] * SIN(angle),
                      vec[1] * vec[2] * (1 - COS(angle)) + vec[0] * SIN(angle),
                      COS(angle) + vec[2]**2 * (1 - COS(angle))
                  ]])

    return R


def gaussian_beam_z_axis_x_pol(x_grid, y_grid, z_grid, w0, center, R, omega,
                               polarity, eps_val) -> complex:
    """Scalar gaussian beam.
    """
    x = R[0, 0] * (x_grid - center[0]) + R[0, 1] * (
        y_grid - center[1]) + R[0, 2] * (z_grid - center[2])
    y = R[1, 0] * (x_grid - center[0]) + R[1, 1] * (
        y_grid - center[1]) + R[1, 2] * (z_grid - center[2])
    z = R[2, 0] * (x_grid - center[0]) + R[2, 1] * (
        y_grid - center[1]) + R[2, 2] * (z_grid - center[2])

    wlen = 2.0 * np.pi / (omega * np.sqrt(eps_val))
    k = 2.0 * np.pi / wlen
    z_r = np.pi * w0**2 / wlen  # raleigh length
    w_z = w0 * (1 + (z / z_r)**2)**0.5  # beam waist as a function of z
    inv_R_z = np.zeros_like(z_grid)
    inv_R_z[z != 0] = np.power(z[z != 0] * (1 + (z_r / z[z != 0])**2), -1)
    gouy_z = np.arctan(z / z_r)  # gouy phase
    r2 = x**2 + y**2

    imp = np.sqrt(1 / eps_val)

    #normalize power to 1
    return np.sqrt(imp)*2/(np.sqrt(np.pi)*(w0)) \
        *(w0/w_z)*np.exp(-r2/w_z**2)*np.exp(-1j*polarity
        *(k*z+k*inv_R_z*(r2/2)-gouy_z))


def plane_wave_z_axis_x_pol(x_grid, y_grid, z_grid, R, omega, polarity,
                            eps_val) -> complex:
    """Scalar plane wave.
    """
    z = R[2, 0] * x_grid + R[2, 1] * y_grid + R[2, 2] * z_grid

    wlen = 2.0 * np.pi / (omega * np.sqrt(eps_val))
    k = 2.0 * np.pi / wlen
    # normalize to power 1 per um**2
    return np.exp(-polarity * 1j * k * z)


def scalar2rotated_vector_fields(eps_grid: Grid,
                                 scalar_field_function: Callable,
                                 mu: List[np.ndarray],
                                 omega: float,
                                 axis: int,
                                 slices: List[slice],
                                 theta: float = 0,
                                 psi: float = 0,
                                 polarization_angle: float = 0,
                                 polarity: int = 1,
                                 power: float = 1,
                                 full_fields: bool = False):
    """
    Given a function that evaluates the scalar field that propagates in the z-direction,
    this function will make the vector field taking into account three rotations.

    Args:
        eps_grid: gridlock.grid with the permittivity distribution.
        mu: Permeability distribution.
        omega: The frequency of the mode.
        axis: Direction of propagation.
        slices: Source slice which define the position of the source in the grid.
        theta: Rotation around the default E-component.
        psi: Rotation around the source plane normal.
        polarization_angle: Rotation around the propagation direction.
        polarity: 1 if forward propagating. -1 if backward propagating.
        power: power
        full_fields: True gives you the field in the entire simulation space.
            False gives only the fields in the slices, the rest of the simulation space
            will be zero.

    Returns:
        Results: dict with the wavevector, the e-field and the h-field.

    """
    if mu is None:
        mu = [np.ones(eps_grid.shape)] * 3
    dxes = [eps_grid.dxyz, eps_grid.autoshifted_dxyz()]
    epsilon = eps_grid.grids
    xyz_shift = [eps_grid.shifted_xyz(which_shifts=i) for i in range(3)]

    # Define rotation to set z as propagation direction.
    order = np.roll(range(3), 2 - axis)
    reverse_order = np.roll(range(3), axis - 2)

    #Make grid points
    xyz = xyz_shift[order[0]]
    x_Ex, y_Ex, z_Ex = np.meshgrid(xyz[order[0]],
                                   xyz[order[1]],
                                   xyz[order[2]],
                                   indexing='ij')
    xyz = xyz_shift[order[1]]
    x_Ey, y_Ey, z_Ey = np.meshgrid(xyz[order[0]],
                                   xyz[order[1]],
                                   xyz[order[2]],
                                   indexing='ij')
    xyz = xyz_shift[order[2]]
    x_Ez, y_Ez, z_Ez = np.meshgrid(xyz[order[0]],
                                   xyz[order[1]],
                                   xyz[order[2]],
                                   indexing='ij')

    #Make total rotation matrix
    k = np.array([0, 0, polarity])
    e0 = np.array([1, 0, 0])
    R_theta = rotation_matrix(e0, theta)
    R_psi = rotation_matrix(k, psi)
    k = R_psi @ R_theta @ k
    R_pol = rotation_matrix(k, polarization_angle)
    R = R_pol @ R_psi @ R_theta
    e0_rot = R @ e0

    #Make wavevector
    ref_index = np.sqrt(np.real(np.average(epsilon[0][tuple(slices)])))
    bloch_vector = (k * omega * ref_index)

    #Evaluate fields.
    ex_mag = scalar_field_function(x_Ex, y_Ex, z_Ex,
                                   np.linalg.inv(R)) * e0_rot[0]
    ey_mag = scalar_field_function(x_Ey, y_Ey, z_Ey,
                                   np.linalg.inv(R)) * e0_rot[1]
    ez_mag = scalar_field_function(x_Ez, y_Ez, z_Ez,
                                   np.linalg.inv(R)) * e0_rot[2]
    E_fields = [ex_mag, ey_mag, ez_mag]

    #Make H fields.
    dxes = [[dx[i] for i in order] for dx in dxes]
    e_vec = vec(E_fields)
    h_vec = operators.e2h(omega=omega,
                          dxes=dxes,
                          mu=vec([mu[i].transpose(order) for i in order]),
                          bloch_vec=bloch_vector) @ e_vec
    H_fields = unvec(h_vec, E_fields[0].shape)

    #Normalize fields.
    #Roll back
    E = [None] * 3
    H = [None] * 3
    wavevector = np.zeros(3)
    for a, o in enumerate(reverse_order):
        E[a] = np.zeros_like(epsilon[0], dtype=complex)
        H[a] = np.zeros_like(epsilon[0], dtype=complex)

        if full_fields:
            E[a] = np.sqrt(power) * E_fields[o].transpose(reverse_order)
            H[a] = np.sqrt(power) * H_fields[o].transpose(reverse_order)
        else:
            E[a][tuple(slices)] = np.sqrt(power) * E_fields[o][tuple(
                [slices[i] for i in order])].transpose(reverse_order)
            H[a][tuple(slices)] = np.sqrt(power) * H_fields[o][tuple(
                [slices[i] for i in order])].transpose(reverse_order)

        wavevector[a] = bloch_vector[o]

    results = {
        'wavevector': wavevector,
        'H': H,
        'E': E,
    }

    return results


def build_plane_wave_source(eps_grid: Grid,
                            omega: float,
                            axis: int,
                            slices=List[slice],
                            mu: List[np.ndarray] = None,
                            theta: float = 0,
                            psi: float = 0,
                            polarization_angle: float = 0,
                            polarity: int = 1,
                            border: int or List[int] = 0,
                            power: float = 1):
    """Builds a plane wave source.

    By default, the plane wave propagates along polarity of the given axis and
    is linearly polarized along the x-direction if axis is z, y-direction if x and
    z direction if y. `theta` rotates the propagation direction around the E-field,
    then 'psi' rotates source plane normal, and the polarization_angle rotates around
    the propagation direction.

    Args:
        eps_grid: gridlock.grid with the permittivity distribution.
        omega: The frequency of the mode.
        axis: Direction of propagation.
        slices: Source slice which define the position of the source in the grid.
        mu: Permeability distribution.
        theta: Rotation around the default E-component.
        psi: Rotation around the source plane normal.
        polarization_angle: Rotation around the propagation direction.
        polarity: 1 if forward propagating. -1 if backward propagating.
        border: border in grid points where the intensity of the planewave decreased to 0.
            For example: [10, 10, 10] assuming radiation in the z direction will put 10
                grid border on both sides in the x and y direction and ignore the z.
                (If the border is larger then the simulation it will be ignored aswell.)
        power: power transmitted through the slice.

    Returns:
        Current source J.
    """
    # Perpare arguments.
    shape = eps_grid.shape

    # Make scalar fields.
    eps_val = np.real(np.average(eps_grid.grids[0][tuple(slices)]))
    scalar_field_function = lambda x, y, z, R: plane_wave_z_axis_x_pol(
        x, y, z, R, omega, polarity, eps_val)

    # Get vector fields.
    fields = scalar2rotated_vector_fields(
        eps_grid=eps_grid,
        scalar_field_function=scalar_field_function,
        mu=mu,
        omega=omega,
        axis=axis,
        slices=slices,
        theta=theta,
        psi=psi,
        polarization_angle=polarization_angle,
        polarity=polarity,
        power=power,
        full_fields=True)

    def make_mask(shape: List[int], slices: List, axis: int,
                  border: List[int]) -> np.ndarray:
        """Builds a mask with a border. On the border the intensity goes from 1 to 0.

        Args:
            shape: Shape of the simulation.
            slices: List of slices where the plane wave is.
            axis: Direction of the plane wave.
            border: number of gridpoint that make up the border.

        Returns:
            mask
        """
        size = [s.stop - s.start for s in slices]
        coords = []
        for i in range(3):
            if (size[i] - 1) > 2 * border[i] and border[i] > 0:
                d = 1 / np.array(border[i])
                x = np.hstack([
                    np.arange(1, 0, -d),
                    np.zeros(size[i] - 2 * border[i]),
                    np.arange(d, 1 + d, d)
                ])
            else:
                x = np.zeros(size[i])
            coords.append(x)
        coords[axis] = np.zeros(shape[axis])
        ind = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        co = (ind[0]**2 + ind[1]**2 + ind[2]**2)**0.5
        co[co > 1] = 1
        mask_slice = 1 + 2 * co**3 - 3 * co**2
        mask = np.zeros(shape)
        slices_mask = copy.deepcopy(slices)
        slices_mask[axis] = slice(None, None)
        mask[tuple(slices_mask)] = mask_slice
        return mask

    # Prepare border and the mask for be fields.
    if isinstance(border, int):
        border = 3 * [border]
    border[axis] = 0  # You can not have a border in the axis direction
    size = [s.stop - s.start for s in slices]
    for i, (b, s) in enumerate(zip(border, size)):
        if 2 * b < s:
            border[i] = b
        elif s == 1:
            border[i] = 0
        else:
            raise ValueError("Two times the border is larger then the size" +
                             " of the plane wave in axis {}.".format(i))

    border = [b if 2 * b < s else 0 for b, s in zip(border, size)]
    mask = make_mask(shape, slices, axis, border)

    # Normalize fields
    dxes = [eps_grid.dxyz, eps_grid.autoshifted_dxyz()]
    slices_P = copy.deepcopy(slices)
    slices_P = [
        slice(sl.start + b, sl.stop - b) for sl, b in zip(slices_P, border)
    ]
    P = waveguide_mode.compute_transmission_chew(E=fields['E'],
                                                 H=fields['H'],
                                                 axis=axis,
                                                 omega=omega,
                                                 dxes=dxes,
                                                 slices=slices_P)
    fields['E'] /= np.sqrt(abs(P))
    fields['H'] /= np.sqrt(abs(P))
    fields['E'] = [mask * e for e in fields['E']]
    fields['H'] = [mask * h for h in fields['H']]

    # Calculate the source.
    dxes = [eps_grid.dxyz, eps_grid.autoshifted_dxyz()]

    # make current
    field_slices = [slice(None, None)] * 3
    J_slices = slices
    if polarity == -1:
        ind = slices[axis].start
        field_slices[axis] = slice(None, ind)
        J_slices[axis] = slice(ind - 1, ind + 1)
    else:
        ind = slices[axis].stop - 1
        field_slices[axis] = slice(ind, None)
        J_slices[axis] = slice(ind - 1, ind + 1)
    E = np.zeros_like(fields['E'])
    for i in range(3):
        E[i][tuple(field_slices)] = fields['E'][i][tuple(field_slices)]

    full = operators.e_full(omega,
                            dxes,
                            vec(eps_grid.grids),
                            bloch_vec=fields['wavevector'])
    J_vec = 1 / (-1j * omega) * full @ vec(E)
    J_temp = unvec(J_vec, E[0].shape)
    J = np.zeros_like(J_temp)
    for i in range(3):
        J[i][tuple(J_slices)] = J_temp[i][tuple(J_slices)]

    return J, fields['wavevector']


def build_gaussian_source(eps_grid: Grid,
                          omega: float,
                          w0: float,
                          center: np.ndarray,
                          axis: int,
                          slices=List[slice],
                          mu: List[np.ndarray] = None,
                          theta: float = 0,
                          psi: float = 0,
                          polarization_angle: float = 0,
                          polarity: int = 1,
                          power: float = 1):
    """Builds a gaussian beam source.

    By default, the gaussian beam propagates along polarity of the given axis and
    is linearly polarized along the x-direction if axis is z, y-direction if x and
    z direction if y. `theta` rotates the propagation direction around the E-field,
    then 'psi' rotates source plane normal, and the polarization_angle rotates around
    the propagation direction.

    Args:
        eps_grid: gridlock.grid with the permittivity distribution.
        omega: The frequency of the mode.
        axis: Direction of propagation.
        slices: Source slice which define the position of the source in the grid.
        mu: Permeability distribution.
        theta: Rotation around the default E-component.
        psi: Rotation around the source plane normal.
        polarization_angle: Rotation around the propagation direction.
        polarity: 1 if forward propagating. -1 if backward propagating.
        power: Power is the gaussian beam.

    Returns:
        Current source J.
    """
    # Make scalar fields.
    eps_val = np.real(np.average(eps_grid.grids[0][tuple(slices)]))
    scalar_field_function = lambda x, y, z, R: gaussian_beam_z_axis_x_pol(
        x, y, z, w0, center, R, omega, polarity, eps_val)

    # Get vector fields.
    fields = scalar2rotated_vector_fields(
        eps_grid=eps_grid,
        scalar_field_function=scalar_field_function,
        mu=mu,
        omega=omega,
        axis=axis,
        slices=slices,
        theta=theta,
        psi=psi,
        polarization_angle=polarization_angle,
        polarity=polarity,
        power=power,
        full_fields=True)

    # Calculate the source.
    dxes = [eps_grid.dxyz, eps_grid.autoshifted_dxyz()]

    # make current
    field_slices = [slice(None, None)] * 3
    J_slices = slices
    if polarity == -1:
        ind = slices[axis].start
        field_slices[axis] = slice(None, ind)
        J_slices[axis] = slice(ind - 1, ind + 1)
    else:
        ind = slices[axis].stop - 1
        field_slices[axis] = slice(ind, None)
        J_slices[axis] = slice(ind - 1, ind + 1)
    E = np.zeros_like(fields['E'])
    for i in range(3):
        E[i][tuple(field_slices)] = fields['E'][i][tuple(field_slices)]

    full = operators.e_full(omega,
                            dxes,
                            vec(eps_grid.grids),
                            bloch_vec=fields['wavevector'])
    J_vec = 1 / (-1j * omega) * full @ vec(E)
    J_temp = unvec(J_vec, E[0].shape)
    J = np.zeros_like(J_temp)
    for i in range(3):
        J[i][tuple(J_slices)] = J_temp[i][tuple(J_slices)]

    return J, fields['wavevector']


def normalize_source_by_sim(
        omega: float,
        source: List[np.ndarray],
        eps: List[np.ndarray],
        dxes: List[np.ndarray],
        pml_layers: fdfd_tools.PmlLayers,
        solver: Callable,
        power: float,
        bloch_vector: List[float] = [0, 0, 0],
) -> List[np.ndarray]:
    """Normalizes a source by running a simulation.

    The simulation is run with uniform media (index is determined by averaging
    over all locations where the source is present), and the power emitted
    by the source is computed. Based on this, `source` is renormalized to
    emit `power` instead.

    WARNING: Only works for uniform meshes in all three directions.

    Args:
        omega: Angular frequency.
        source: Simulation source to normalize.
        eps: Permittivity distribution.
        dxes: List of grid spacings.
        pml_layers: Number of PML layers to apply on boundary.
        solver: Solver to use to run normalization simulation.
        power: Power that normalized source should emit.
        bloch_vector: Bloch vector to apply on simulation.

    Returns:
        Source that emits `power` in uniform media.
    """
    # Compute the average permittivity of the background by performing an
    # average over the permittivity distribution weighted by the magnitude of
    # the source (this is just convenience to avoid needing to set a threshold
    # for numerical errors in the source).
    source_abs = np.abs(fdfd_tools.vec(source))
    eps_avg = np.sum(source_abs * fdfd_tools.vec(eps)) / np.sum(source_abs)

    eps_uniform = [np.ones(eps[i].shape) * eps_avg for i in range(3)]
    electric_fields = solver.solve(
        omega=omega,
        dxes=dxes,
        epsilon=fdfd_tools.vec(eps_uniform),
        pml_layers=pml_layers,
        J=fdfd_tools.vec(source),
        bloch_vec=bloch_vector,
    )
    J_dot_E = -np.real(np.conj(fdfd_tools.vec(source)) * electric_fields)
    # TODO(logansu): Make this work for nonuniform meshes.
    dx = np.real(dxes[0][0][0])
    cur_power = 0.5 * np.sum(J_dot_E[:]) * dx**3

    J_normalized = copy.deepcopy(source)
    for i in range(3):
        J_normalized[i] *= np.sqrt(power / cur_power)
    return J_normalized
