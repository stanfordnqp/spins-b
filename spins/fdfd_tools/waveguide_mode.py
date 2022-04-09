from typing import Dict, List, Union, Optional
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg

from . import vec, unvec, dx_lists_t, vfield_t, field_t
from . import operators, waveguide, functional

from spins.gridlock import Direction


def solve_waveguide_mode_2d(mode_number: int,
                            omega: complex,
                            dxes: dx_lists_t,
                            epsilon: vfield_t,
                            mu: vfield_t = None,
                            wavenumber_correction: bool = True
                            ) -> Dict[str, Union[complex, field_t]]:
    """
    Given a 2d region, attempts to solve for the eigenmode with the specified mode number.

    :param mode_number: Number of the mode, 0-indexed
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :param wavenumber_correction: Whether to correct the wavenumber to
        account for numerical dispersion (default True)
    :return: {'E': List[np.ndarray], 'H': List[np.ndarray], 'wavenumber': complex}
    """

    '''
    Solve for the largest-magnitude eigenvalue of the real operator
     by using power iteration.
    '''
    # check if eps has a imaginary part
    if (np.imag(epsilon)!=0).any():
        warnings.warn('Epsilon in 2D mode solver has an imaginary part'
            )

    dxes_real = [[np.real(dx) for dx in dxi] for dxi in dxes]

    A_r = waveguide.operator(np.real(omega), dxes_real, np.real(epsilon), np.real(mu))

    # Use power iteration for 20 steps to estimate the dominant eigenvector
    v = np.random.rand(A_r.shape[0])
    for _ in range(20):
        v = A_r @ v
        v /= np.linalg.norm(v)

    lm_eigval = v @ A_r @ v

    '''
    Shift by the absolute value of the largest eigenvalue, then find a few of the
     largest (shifted) eigenvalues. The shift ensures that we find the largest
     _positive_ eigenvalues, since any negative eigenvalues will be shifted to the range
     0 >= neg_eigval + abs(lm_eigval) > abs(lm_eigval)
    '''
    shifted_A_r = A_r + abs(lm_eigval) * sparse.eye(A_r.shape[0])
    eigvals, eigvecs = spalg.eigs(shifted_A_r, which='LM', k=mode_number + 3, ncv=50)

    # Pick the eigenvalue we want from the few we found
    k = eigvals.argsort()[-(mode_number+1)]
    v = eigvecs[:, k]

    '''
    Now solve for the eigenvector of the full operator, using the real operator's
     eigenvector as an initial guess for Rayleigh quotient iteration.
    '''
    A = waveguide.operator(omega, dxes, epsilon, mu)

    eigval = None
    for _ in range(40):
        eigval = v @ A @ v
        if np.linalg.norm(A @ v - eigval * v) < 1e-13:
            break
        w = spalg.spsolve(A - eigval * sparse.eye(A.shape[0]), v)
        v = w / np.linalg.norm(w)

    # Calculate the wave-vector (force the real part to be positive)
    wavenumber = np.sqrt(eigval)
    wavenumber *= np.sign(np.real(wavenumber))

    e, h = waveguide.normalized_fields(v, wavenumber, omega, dxes, epsilon, mu)

    '''
    Perform correction on wavenumber to account for numerical dispersion.

     See Numerical Dispersion in Taflove's FDTD book.
     This correction term reduces the error in emitted power, but additional
      error is introduced into the E_err and H_err terms. This effect becomes
      more pronounced as beta increases.
    '''
    if wavenumber_correction:
        wavenumber -= 2 * np.sin(np.real(wavenumber / 2)) - np.real(wavenumber)

    shape = [d.size for d in dxes[0]]
    fields = {
        'wavenumber': wavenumber,
        'E': unvec(e, shape),
        'H': unvec(h, shape),
    }

    return fields


def solve_waveguide_mode(mode_number: int,
                         omega: complex,
                         dxes: dx_lists_t,
                         axis: int,
                         polarity: int,
                         slices: List[slice],
                         epsilon: field_t,
                         mu: field_t = None,
                         wavenumber_correction: bool = True
                         ) -> Dict[str, Union[complex, np.ndarray]]:
    """
    Given a 3D grid, selects a slice from the grid and attempts to
     solve for an eigenmode propagating through that slice.

    :param mode_number: Number of the mode, 0-indexed
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :param wavenumber_correction: Whether to correct the wavenumber to
        account for numerical dispersion (default True)
    :return: {'E': List[np.ndarray], 'H': List[np.ndarray], 'wavenumber': complex}
    """
    if mu is None:
        mu = [np.ones_like(epsilon[0])] * 3

    '''
    Solve the 2D problem in the specified plane
    '''
    # Define rotation to set z as propagation direction
    order = np.roll(range(3), 2 - axis)
    reverse_order = np.roll(range(3), axis - 2)

    # Reduce to 2D and solve the 2D problem
    args_2d = {
        'dxes': [[dx[i][slices[i]] for i in order[:2]] for dx in dxes],
        'epsilon': vec([epsilon[i][slices].transpose(order) for i in order]),
        'mu': vec([mu[i][slices].transpose(order) for i in order]),
        'wavenumber_correction': wavenumber_correction,
    }
    fields_2d = solve_waveguide_mode_2d(mode_number, omega=omega, **args_2d)

    '''
    Apply corrections and expand to 3D
    '''
    # Scale based on dx in propagation direction
    dxab_forward = np.array([dx[order[2]][slices[order[2]]] for dx in dxes])

    # Adjust for propagation direction
    if polarity < 0:
        for i in range(3):
            fields_2d['E'][i] = np.conj(fields_2d['E'][i])
            fields_2d['H'][i] = -np.conj(fields_2d['H'][i])

    # Apply phase shift to H-field
    d_prop = 0.5 * sum(dxab_forward)
    for a in range(3):
        fields_2d['H'][a] *= np.exp(-polarity * 1j * 0.5 * fields_2d['wavenumber'] * d_prop)

    # Expand E, H to full epsilon space we were given
    E = [None]*3
    H = [None]*3
    for a, o in enumerate(reverse_order):
        E[a] = np.zeros_like(epsilon[0], dtype=complex)
        H[a] = np.zeros_like(epsilon[0], dtype=complex)

        E[a][slices] = fields_2d['E'][o][:, :, None].transpose(reverse_order)
        H[a][slices] = fields_2d['H'][o][:, :, None].transpose(reverse_order)

    results = {
        'wavenumber': fields_2d['wavenumber'],
        'H': H,
        'E': E,
    }

    return results


def compute_source(E: field_t,
                   H: field_t,
                   wavenumber: complex,
                   omega: complex,
                   dxes: dx_lists_t,
                   axis: int,
                   polarity: int,
                   slices: List[slice],
                   mu: field_t = None,
                   ) -> field_t:
    """
    Given an eigenmode obtained by solve_waveguide_mode, returns the current source distribution
    necessary to position a unidirectional source at the slice location.

    :param E: E-field of the mode
    :param H: H-field of the mode (advanced by half of a Yee cell from E)
    :param wavenumber: Wavenumber of the mode
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: J distribution for the unidirectional source
    """
    if mu is None:
        mu = [1] * 3

    J = [None]*3
    M = [None]*3

    src_order = np.roll(range(3), -axis)
    exp_iphi = np.exp(1j * polarity * wavenumber * dxes[1][int(axis)][slices[int(axis)]])
    J[src_order[0]] = np.zeros_like(E[0])
    J[src_order[1]] = +exp_iphi * H[src_order[2]] * polarity
    J[src_order[2]] = -exp_iphi * H[src_order[1]] * polarity

    M[src_order[0]] = np.zeros_like(E[0])
    M[src_order[1]] = +np.roll(E[src_order[2]], -1, axis=axis) * polarity
    M[src_order[2]] = -np.roll(E[src_order[1]], -1, axis=axis) * polarity

    A1f = functional.curl_h(dxes)

    Jm_iw = A1f([M[k] / mu[k] for k in range(3)])
    for k in range(3):
        J[k] += Jm_iw[k] / (-1j * omega)

    return J / dxes[1][int(axis)][slices[int(axis)]]


def compute_overlap_e(E: field_t,
                      H: field_t,
                      wavenumber: complex,
                      omega: complex,
                      dxes: dx_lists_t,
                      axis: int,
                      polarity: int,
                      slices: List[slice],
                      mu: field_t = None,
                      ) -> field_t:
    """
    Given an eigenmode obtained by solve_waveguide_mode, calculates overlap_e for the
    mode orthogonality relation Integrate(((E x H_mode) + (E_mode x H)) dot dn)
    [assumes reflection symmetry].

    overlap_e makes use of the e2h operator to collapse the above expression into
     (vec(E) @ vec(overlap_e)), allowing for simple calculation of the mode overlap.

    :param E: E-field of the mode
    :param H: H-field of the mode (advanced by half of a Yee cell from E)
    :param wavenumber: Wavenumber of the mode
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: Propagation axis (0=x, 1=y, 2=z)
    :param polarity: Propagation direction (+1 for +ve, -1 for -ve)
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: overlap_e for calculating the mode overlap
    """
    cross_plane = [slice(None)] * 3
    cross_plane[axis] = slices[axis]

    # Determine phase factors for parallel slices
    a_shape = np.roll([-1, 1, 1], axis)
    a_E = np.real(dxes[0][axis]).cumsum()
    a_H = np.real(dxes[1][axis]).cumsum()
    iphi = -polarity * 1j * wavenumber
    phase_E = np.exp(iphi * (a_E - a_E[slices[axis]])).reshape(a_shape)
    phase_H = np.exp(iphi * (a_H - a_H[slices[axis]])).reshape(a_shape)

    # Expand our slice to the entire grid using the calculated phase factors
    Ee = [None]*3
    He = [None]*3
    for k in range(3):
        Ee[k] = phase_E * E[k][tuple(cross_plane)]
        He[k] = phase_H * H[k][tuple(cross_plane)]


    # Write out the operator product for the mode orthogonality integral
    domain = np.zeros_like(E[0], dtype=int)
    domain[slices] = 1

    npts = E[0].size
    dn = np.zeros(npts * 3, dtype=int)
    dn[0:npts] = 1
    dn = np.roll(dn, npts * axis)

    e2h = operators.e2h(omega, dxes, mu)
    ds = sparse.diags(vec([domain]*3))
    h_cross_ = operators.poynting_h_cross(vec(He), dxes)
    e_cross_ = operators.poynting_e_cross(vec(Ee), dxes)

    overlap_e = dn @ ds @ (-h_cross_ + e_cross_ @ e2h)

    # Normalize
    norm_factor = np.abs(overlap_e @ vec(Ee))
    overlap_e /= norm_factor

    return unvec(overlap_e, E[0].shape)


def build_waveguide_source(
    omega: complex, dxes: List[np.ndarray], eps: List[np.ndarray],
    mu: Optional[List[np.ndarray]], axis: Direction, waveguide_slice,
    polarity: int, mode_num: int, power: float) -> List[np.ndarray]:
    """ Builds a TFSF source for translationally-invariant waveguide.

    Args:
        omega: The frequency of the mode.
        dxes: List of cell widths.
        eps: Permittivity distribution.
        mu: Permeability distribution.
        axis: Direction of propagation.
        waveguide_slice: Coordinates denoting where to get a 2D slice
            of the waveguide.
        polarity: 1 if forward propagating. -1 if backward propagating.
        mode_num: The mode to inject.
        power: Power emitted by the source.
    Returns:
        Current source J.
    """
    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis.value,
        'slices': [slice(i, f+1) for i, f in zip(*waveguide_slice)],
        'polarity': polarity,
        'mu': mu
    }
    wgmode_result = solve_waveguide_mode(
        mode_number = mode_num,
        epsilon = eps,
        **sim_params)
    J = compute_source(**wgmode_result, **sim_params)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)
    return J
