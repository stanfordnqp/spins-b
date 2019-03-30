import copy
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg
from typing import Dict, List

from spins.fdfd_tools import vec, unvec, dx_lists_t, vfield_t, field_t
from spins.fdfd_tools import operators, waveguide, functional
from . import phc_mode_solver as phc

from spins.gridlock import Direction


def solve_waveguide_mode_2d(
        mode_number: int,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfield_t,
        mu: vfield_t = None,
        wavenumber_correction: bool = False) -> Dict[str, complex or field_t]:
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
    if (np.imag(epsilon) != 0).any():
        warnings.warn('Epsilon in 2D mode solver has an imaginary part')

    dxes_real = [[np.real(dx) for dx in dxi] for dxi in dxes]

    A_r = waveguide.operator(
        np.real(omega), dxes_real, np.real(epsilon), np.real(mu))

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
    eigvals, eigvecs = spalg.eigs(
        shifted_A_r, which='LM', k=mode_number + 3, ncv=50)

    # Pick the eigenvalue we want from the few we found
    k = eigvals.argsort()[-(mode_number + 1)]
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
        # TODO(logansu): This was the original code but written out more
        # clearly. Clearly, `dx` is not always unity but this function
        # currently does not have access to the `dx` in the propagating
        # direction.
        dx = 1
        wavenumber = np.sin(np.real(wavenumber * dx / 2)) / (dx / 2) + np.imag(wavenumber)

    shape = [d.size for d in dxes[0]]
    fields = {
        'wavenumber': wavenumber,
        'E': unvec(e, shape),
        'H': unvec(h, shape),
    }

    return fields


def solve_waveguide_mode_3d(
        mode_number: int,
        omega_appx: float,
        wavenumber: float,
        dxes: dx_lists_t,
        axis: int,
        axis_rot: int,
        angle: float,
        polarity: int,
        slices: List[slice],
        eps: field_t,
        mu: field_t = None,
        expand_fields: bool = False,
        additional_mode_cal: int = 1) -> Dict[str, complex or np.ndarray]:
    """
    Given a 3d region, attempts to solve for the eigenmode with the specified mode number.

    :param mode_number: Number of the mode, 0-indexed
    :param omega_appx: approximated angular frequency of the simulation
    :param wavenumber: wavenumber of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param axis: propagation axis, i.e. x, y or z
    :param angle: angle with the propagation axis
    :param polarity: propagation direction (1 or -1)
    :param slices: slices that describe the region for the mode solve
    :param eps: permittivity
    :param mu: magnetic permeability (default 1 everywhere)
    :param expand_field: whether the calculated fields are duplicated over the entire simulation
        according to the periodicity and angle (default=false)
    :additional_mode_cal: how many mode are calculated
    :return: {'wavenumber': List[np.ndarray], 'H': List[np.ndarray], 'E':List[np.array]}, omega
    """
    # set propagation direction to the z-axis
    order = np.roll(range(3), 2 - axis)
    reverse_order = np.roll(range(3), axis - 2)

    source_slices = [slices[i] for i in order]
    if polarity == 1:
        source_slices[2] = slice(source_slices[2].stop - 1,
                                 source_slices[2].stop)
    else:
        source_slices[2] = slice(source_slices[2].start,
                                 source_slices[2].start + 1)
    eig_slices = [slices[i] for i in order]

    eps_sim = [eps[i].transpose(order) for i in order]
    dxes_sim = [[dx[i] for i in order] for dx in dxes]

    eps_eig = [eps[i][tuple(slices)].transpose(order) for i in order]
    if mu:
        mu_eig = [mu[i][tuple(slices)].transpose(order) for i in order]
    else:
        mu_eig = np.ones_like(eps_eig)
    dxes_eig = [[dx[i][slices[i]] for i in order] for dx in dxes]
    shp_eig = eps_eig[0].shape

    # calculate eigenmode
    shift_orthogonal = np.zeros((3, 3))
    shift_orthogonal[2, int(not reverse_order[axis_rot])] = -np.round(
        shp_eig[2] * np.tan(angle))
    bloch_vec = polarity * np.array(
        [np.sin(angle), np.sin(angle),
         np.cos(angle)]) * wavenumber
    bloch_vec[reverse_order[axis_rot]] = 0
    mode_arg = {
        'omega_appx': omega_appx,
        'bloch_vec': bloch_vec,
        'dxes': dxes_eig,
        'epsilon': eps_eig,
        'op_type': 'hfield',
        'set_init_cond': False,
        'num_modes': mode_number + 1 + additional_mode_cal,
        'shift_orthogonal': shift_orthogonal
    }
    eig_list, mode_list = phc.mode_solver(**mode_arg)
    index_sort = np.argsort(np.real(eig_list))

    h_vec = mode_list[index_sort[mode_number]]
    e_vec = operators.h2e(
        omega=eig_list[index_sort[mode_number]],
        dxes=dxes_eig,
        eps=vec(eps_eig),
        bloch_vec=bloch_vec,
        shift_orthogonal=shift_orthogonal) @ h_vec
    h_unvec = unvec(h_vec, shp_eig)
    e_unvec = unvec(e_vec, shp_eig)

    # expand fields
    axis_angle = int(not reverse_order[axis_rot])
    h_fields = np.zeros_like(eps_sim)
    e_fields = np.zeros_like(eps_sim)
    i_start = slices[axis].start
    i = i_start
    while i < e_fields[0].shape[2]:
        sl = [source_slices[0], source_slices[1], slice(i, i + 1)]
        for a in range(3):
            e_fields[a][sl] = operators.append_bloch_shift(
                A=e_unvec[a],
                axis=2,
                ind_axis=i - i_start,
                dx=dxes_eig[0],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift_orthogonal[2])
            h_fields[a][sl] = operators.append_bloch_shift(
                A=h_unvec[a],
                axis=2,
                ind_axis=i - i_start,
                dx=dxes_eig[1],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift_orthogonal[2])
        i += 1
    i = slices[axis].start - 1
    while i >= 0:
        sl = [source_slices[0], source_slices[1], slice(i, i + 1)]
        for a in range(3):
            e_fields[a][sl] = operators.append_bloch_shift(
                A=e_unvec[a],
                axis=2,
                ind_axis=i - i_start,
                dx=dxes_eig[0],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift_orthogonal[2])
            h_fields[a][sl] = operators.append_bloch_shift(
                A=h_unvec[a],
                axis=2,
                ind_axis=i - i_start,
                dx=dxes_eig[1],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift_orthogonal[2])
        i -= 1

    omega = eig_list[index_sort[mode_number]]

    # normalize field by the transmission through the slice
    transmission = compute_transmission_chew(
        E=e_fields,
        H=h_fields,
        axis=2,
        omega=omega,
        dxes=dxes_sim,
        slices=source_slices,
    )
    e = e_fields / np.sqrt(transmission)
    h = h_fields / np.sqrt(transmission)

    # select slice
    if not expand_fields:
        h_fields = np.zeros_like(eps_sim)
        h_fields[[slice(0, 3)] + source_slices] = h[[slice(0, 3)] +
                                                    source_slices]
        e_fields = np.zeros_like(eps_sim)
        e_fields[[slice(0, 3)] + source_slices] = e[[slice(0, 3)] +
                                                    source_slices]
    else:
        e_fields = e
        h_fields = h

    # reverse the order
    e_fields = [e_fields[i].transpose(reverse_order) for i in reverse_order]
    h_fields = [h_fields[i].transpose(reverse_order) for i in reverse_order]

    return {'wavenumber': bloch_vec[2], 'H': h_fields, 'E': e_fields}, omega


def solve_waveguide_mode(
        mode_number: int,
        omega: complex,
        dxes: dx_lists_t,
        axis: int,
        polarity: int,
        slices: List[slice],
        epsilon: field_t,
        mu: field_t = None,
        wavenumber_correction: bool = True) -> Dict[str, complex or np.ndarray]:
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
        'epsilon':
        vec([epsilon[i][tuple(slices)].transpose(order) for i in order]),
        'mu':
        vec([mu[i][tuple(slices)].transpose(order) for i in order]),
        'wavenumber_correction':
        wavenumber_correction,
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
        fields_2d['H'][a] *= np.exp(
            -polarity * 1j * 0.5 * fields_2d['wavenumber'] * d_prop)

    # Expand E, H to full epsilon space we were given
    E = [None] * 3
    H = [None] * 3
    for a, o in enumerate(reverse_order):
        E[a] = np.zeros_like(epsilon[0], dtype=complex)
        H[a] = np.zeros_like(epsilon[0], dtype=complex)

        E[a][tuple(slices)] = fields_2d['E'][o][:, :, None].transpose(
            reverse_order)
        H[a][tuple(slices)] = fields_2d['H'][o][:, :, None].transpose(
            reverse_order)

    results = {
        'wavenumber': fields_2d['wavenumber'],
        'H': H,
        'E': E,
    }

    return results


def compute_source(
        E: field_t,
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

    J = [None] * 3
    M = [None] * 3

    src_order = np.roll(range(3), -axis)
    exp_iphi = np.exp(
        1j * polarity * wavenumber * dxes[1][int(axis)][slices[int(axis)]])
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


def compute_source_angle(
        E: field_t,
        H: field_t,
        wavevector: List[float],
        omega: complex,
        eps: List[np.array],
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
    if polarity == 1:
        Mslice = copy.deepcopy(slices)
        Jslice = copy.deepcopy(slices)
        Jslice[axis] = slice(Jslice[axis].start + 1, Jslice[axis].stop + 1)
    elif polarity == -1:
        Mslice = copy.deepcopy(slices)
        Mslice[axis] = slice(Mslice[axis].start - 1, Mslice[axis].stop - 1)
        Jslice = copy.deepcopy(slices)

    if mu is None:
        mu = np.ones_like(eps)

    curl_h = lambda x: unvec(operators.curl_h(dxes=dxes, bloch_vec=wavevector)@vec(x),
                             x[0].shape)
    curl_e = lambda x: unvec(operators.curl_e(dxes=dxes, bloch_vec=wavevector)@vec(x),
                             x[0].shape)

    M_temp = -np.array(curl_e(E)) - 1j * omega * np.array(mu) * np.array(H)
    J_temp = np.array(curl_h(H)) - 1j * omega * np.array(eps) * np.array(E)
    M = np.zeros_like(M_temp)
    J = np.zeros_like(J_temp)
    J[[slice(0, 3)] + Jslice] = J_temp[[slice(0, 3)] + Jslice]
    M[[slice(0, 3)] + Mslice] = M_temp[[slice(0, 3)] + Mslice]
    J[axis] = np.zeros_like(J[axis])
    M[axis] = np.zeros_like(M[axis])

    Js = np.array(n_cross(H))
    Ms = -np.array(n_cross(E))

    Jm = np.array(curl_h(M / mu))

    J = J - Jm / (-1j * omega)

    return J


def compute_overlap_e(
        E: field_t,
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
    Ee = [None] * 3
    He = [None] * 3
    for k in range(3):
        Ee[k] = phase_E * E[k][tuple(cross_plane)]
        He[k] = phase_H * H[k][tuple(cross_plane)]

    # Write out the operator product for the mode orthogonality integral
    domain = np.zeros_like(E[0], dtype=int)
    domain[tuple(slices)] = 1

    npts = E[0].size
    dn = np.zeros(npts * 3, dtype=int)
    dn[0:npts] = 1
    dn = np.roll(dn, npts * axis)

    e2h = operators.e2h(omega, dxes, mu)
    ds = sparse.diags(vec([domain] * 3))
    h_cross_ = operators.poynting_h_cross(vec(He), dxes)
    e_cross_ = operators.poynting_e_cross(vec(Ee), dxes)

    overlap_e = dn @ ds @ (-h_cross_ + e_cross_ @ e2h)

    # Normalize
    norm_factor = np.abs(overlap_e @ vec(Ee))
    overlap_e /= norm_factor

    return unvec(overlap_e, E[0].shape)


def compute_transmission(
        E: field_t,
        H: field_t,
        axis: int,
        omega: complex,
        dxes: dx_lists_t,
        slices: List[slice],
        bloch_vec: np.ndarray = np.zeros(3),
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
    :axis: propagation axis
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: overlap_e for calculating the mode overlap
    """
    if (slices[axis].stop - slices[axis].start) != 1:
        raise ValueError('The slice in the axis direction is not 1 wide.')
    # Write out the operator product for the mode orthogonality integral
    domain = np.zeros_like(E)
    domain[tuple([slice(0, 3)] + slices)] = 1

    dn = np.zeros_like(E)
    dn[axis] = np.ones_like(E[axis])
    dn_vec = vec(dn)

    e2h = operators.e2h(omega, dxes, mu, bloch_vec=bloch_vec)
    ds = sparse.diags(vec(domain))
    h_cross_ = operators.poynting_h_cross(vec(H), dxes)
    e_cross_ = operators.poynting_e_cross(vec(E), dxes)

    overlap_e = dn_vec @ ds @ (-h_cross_ + e_cross_ @ e2h)

    return np.abs(overlap_e @ vec(E))


def compute_transmission_chew(
        E: field_t,
        H: field_t,
        axis: int,
        omega: complex,
        dxes: dx_lists_t,
        slices: List[slice],
        mu: field_t = None,
        bloch_vec: np.ndarray = np.zeros(3)
    ) -> field_t:
    """
    :param E: E-field of the mode
    :param H: H-field of the mode (advanced by half of a Yee cell from E)
    :axis: propagation axis
    :omega: Angular frequency of the simulation
    :dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :mu: Magnetic permeability (default 1 everywhere)
    :return: overlap_e for calculating the mode overlap
    """
    if (slices[axis].stop - slices[axis].start) != 1:
        raise ValueError('The slice in the axis direction is not 1 wide.')
    # Write out the operator product for the mode orthogonality integral
    domain = np.zeros_like(E)
    domain[tuple([slice(0, 3)] + slices)] = 1

    dn = np.zeros_like(E)
    dn[axis] = np.ones_like(E[axis])
    dn_vec = vec(dn)

    ds = sparse.diags(vec(domain))
    e_cross_ = operators.poynting_chew_e_cross(vec(E), dxes)

    overlap_e = dn_vec @ ds @ e_cross_

    return np.abs(overlap_e @ np.conj(vec(H)))


def compute_overlap_e_angled(
        E: field_t,
        H: field_t,
        axis: int,
        omega: complex,
        dxes: dx_lists_t,
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
    :axis: propagation axis
    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param slices: epsilon[tuple(slices)] is used to select the portion of the grid to use
        as the waveguide cross-section. slices[axis] should select only one
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: overlap_e for calculating the mode overlap
    """
    if (slices[axis].stop - slices[axis].start) != 1:
        raise ValueError('The slice in the axis direction is not 1 wide.')
    # Write out the operator product for the mode orthogonality integral
    domain = np.zeros_like(E)
    domain[[slice(0, 3)] + slices] = 1

    dn = np.zeros_like(E)
    dn[axis] = np.ones_like(E[axis])
    dn_vec = vec(dn)

    e2h = operators.e2h(omega, dxes, mu)
    ds = sparse.diags(vec(domain))
    h_cross_ = operators.poynting_h_cross(np.conj(vec(H)), dxes)
    e_cross_ = operators.poynting_e_cross(np.conj(vec(E)), dxes)

    overlap_e = dn_vec @ ds @ (-h_cross_ + e_cross_ @ e2h)

    # Normalize
    norm_factor = np.abs(overlap_e @ vec(E))
    overlap_e /= norm_factor

    return unvec(overlap_e, E[0].shape)


def build_waveguide_source(omega: complex, dxes: List[np.ndarray],
                           eps: List[np.ndarray], mu: List[np.ndarray] or None,
                           axis: Direction, waveguide_slice, polarity: int,
                           mode_num: int, power: float, get_wavenumber: bool=False
                          ) -> List[np.ndarray]:
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
        get_wavenumber: Boolean indicating if the wavenumber is added to the
            output.
    Returns:
        Current source J.
    """

    if type(waveguide_slice[0]) is not slice:
        slices = [slice(i, f + 1) for i, f in zip(*waveguide_slice)]
    else:
        slices = waveguide_slice

    if type(axis) is Direction:
        axis = axis.value

    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis,
        'slices': slices,
        'polarity': polarity,
        'mu': mu
    }
    wgmode_result = solve_waveguide_mode(
        mode_number=mode_num, epsilon=eps, **sim_params)

    # Normalize the phase of the fields
    E2 = (abs(wgmode_result['E'][0])**2 + abs(wgmode_result['E'][1])**2 + abs(
        wgmode_result['E'][2])**2)**0.5
    max_ind = np.argmax(E2)
    max_field = [wgmode_result['E'][i].flatten()[max_ind] for i in range(3)]
    max_comp = np.argmax(np.abs(max_field))
    phase = max_field[max_comp] / abs(max_field[max_comp])
    wgmode_result['E'] /= phase
    wgmode_result['H'] /= phase

    J = compute_source(**wgmode_result, **sim_params)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)

    if get_wavenumber:
        return J, wgmode_result['wavenumber']
    return J


def build_waveguide_source_angle(omega: complex,
                                 dxes: List[np.ndarray],
                                 eps: List[np.ndarray],
                                 mu: List[np.ndarray] or None,
                                 axis: Direction,
                                 waveguide_slice,
                                 polarity: int,
                                 mode_num: int,
                                 power: float,
                                 axis_rot: int = None,
                                 angle: float = None) -> List[np.ndarray]:
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
        axis_rot: axis around which the waveguide is rotated
            !!! if this is defined the waveguide mode will be calculated with
                the 3D solver. In this case the slices define a periodic cell.
                The dimension in the direction of the axis determines the angle.
    Returns:
        Current source J.

    """
    if type(waveguide_slice[0]) is not slice:
        slices = [slice(i, f + 1) for i, f in zip(*waveguide_slice)]
    else:
        slices = waveguide_slice

    if type(axis) is Direction:
        axis = axis.value

    if not axis_rot:
        sim_params = {
            'omega': omega,
            'dxes': dxes,
            'axis': axis,
            'slices': slices,
            'polarity': polarity,
            'mu': mu
        }
        wgmode_result = solve_waveguide_mode(
            mode_number=mode_num, epsilon=eps, **sim_params)
        sim_params.pop('dxes')
        sim_params.pop('polarity')

        # Normalize the phase of the fields
        E2 = (abs(wgmode_result['E'][0])**2 + abs(wgmode_result['E'][1])**2 +
              abs(wgmode_result['E'][2])**2)**0.5
        max_ind = np.argmax(E2)
        max_field = [wgmode_result['E'][i].flatten()[max_ind] for i in range(3)]
        max_comp = np.argmax(np.abs(max_field))
        phase = max_field[max_comp] / abs(max_field[max_comp])
        wgmode_result['E'] /= phase
        wgmode_result['H'] /= phase

    else:
        J_slices = copy.deepcopy(slices)
        if polarity == 1:
            J_slices[axis] = slice(J_slices[axis].stop - 1, J_slices[axis].stop)
        else:
            J_slices[axis] = slice(J_slices[axis].start,
                                   J_slices[axis].start + 1)
        # get an estimate of the wavenumber
        dxes_squeeze = copy.deepcopy(dxes)
        dxes_squeeze[0][3 - axis - axis_rot] *= np.cos(angle)
        dxes_squeeze[1][3 - axis - axis_rot] *= np.cos(angle)
        sim_params = {
            'omega': omega,
            'axis': axis,
            'slices': J_slices,
            'mu': mu
        }
        wgmode_result = solve_waveguide_mode(
            mode_number=mode_num,
            dxes=dxes_squeeze,
            epsilon=eps,
            polarity=polarity,
            **sim_params)
        wavenumber = wgmode_result['wavenumber']

        #calculate the 3D mode profile
        wgmode_result, omega_3d = solve_waveguide_mode_3d(
            mode_number=mode_num,
            dxes=dxes,
            omega_appx=omega,
            wavenumber=wavenumber,
            polarity=polarity,
            axis=axis,
            axis_rot=axis_rot,
            angle=angle,
            slices=slices,
            eps=eps,
            expand_fields=False)
        if np.abs(omega_3d - omega) / omega > 0.02:
            raise ValueError(
                'The 3D mode calculation did not match the 2D calculation')

    J = compute_source_angle(**wgmode_result, **sim_params,  dxes=dxes,
                             eps=eps, polarity=polarity)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)
    return J


def build_photonic_crystal_waveguide_source(
        omega_appx: complex, wavenumber: complex, dxes: List[np.ndarray],
        eps: List[np.ndarray], mu: List[np.ndarray] or None, axis: Direction,
        waveguide_slice, polarity: int, mode_num: int,
        power: float) -> List[np.ndarray]:
    """ Builds a TFSF source for translationally-invariant waveguide.

    Args:
        omega_appx: initial guess of frequency of the mode.
        wavenumber: wavenumber
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
    if type(waveguide_slice[0]) is not slice:
        slices = [slice(i, f + 1) for i, f in zip(*waveguide_slice)]
    else:
        slices = waveguide_slice

    if type(axis) is Direction:
        axis = axis.value

    J_slices = copy.deepcopy(slices)
    if polarity == 1:
        J_slices[axis] = slice(J_slices[axis].stop - 1, J_slices[axis].stop)
    else:
        J_slices[axis] = slice(J_slices[axis].start, J_slices[axis].start + 1)

    #calculate the 3D mode profile
    wgmode_result, omega_3d = solve_waveguide_mode_3d(
        mode_number=mode_num,
        dxes=dxes,
        omega_appx=omega_appx,
        wavenumber=wavenumber,
        polarity=polarity,
        axis=axis,
        axis_rot=(axis + 1) % 3,
        angle=0,
        slices=slices,
        eps=eps,
        additional_mode_cal=0,
        expand_fields=False)
    if np.abs(omega_3d - omega_appx) / omega_appx > 0.05:
        raise ValueError(
            'The 3D mode eigenvalue differs strongly from the approximated omega'
        )

    sim_params = {'omega': omega_3d, 'axis': axis, 'slices': J_slices, 'mu': mu}
    J = compute_source_angle(**wgmode_result, **sim_params,  dxes=dxes,
                             eps=eps, polarity=polarity)

    # Increase/decrease J to emit desired power.
    for k in range(len(J)):
        J[k] *= np.sqrt(power)
    return J, omega_3d


def build_overlap(omega: complex, dxes: List[np.ndarray], eps: List[np.ndarray],
                  mu: List[np.ndarray] or None, axis: Direction or int,
                  waveguide_slice, polarity: int, mode_num: int,
                  power: float) -> List[np.ndarray]:
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
        C.

    """
    if type(waveguide_slice[0]) is not slice:
        slices = [slice(i, f + 1) for i, f in zip(*waveguide_slice)]
    else:
        slices = waveguide_slice

    if type(axis) is Direction:
        axis = axis.value

    if np.abs(polarity) != 1:
        raise ValueError("Polarity should be +/- 1, got {}".format(polarity))

    sim_params = {
        'omega': omega,
        'dxes': dxes,
        'axis': axis,
        'slices': slices,
        'polarity': polarity,
        'mu': mu
    }
    wgmode_result = solve_waveguide_mode(
        mode_number=mode_num, epsilon=eps, **sim_params)

    # Prepare slices
    source_slices = copy.deepcopy(slices)
    if polarity == 1:
        source_slices[axis] = slice(source_slices[axis].stop - 1,
                                    source_slices[axis].stop)
    else:
        source_slices[axis] = slice(source_slices[axis].start,
                                    source_slices[axis].start + 1)

    # Normalize the phase of the fields
    E2 = (abs(wgmode_result['E'][0])**2 + abs(wgmode_result['E'][1])**2 + abs(
        wgmode_result['E'][2])**2)**0.5
    max_ind = np.argmax(E2)
    max_field = [wgmode_result['E'][i].flatten()[max_ind] for i in range(3)]
    max_comp = np.argmax(np.abs(max_field))
    phase = max_field[max_comp] / abs(max_field[max_comp])

    # Calculate the overlap
    arg_overlap = {
        'E': wgmode_result['E'] / phase,
        'H': wgmode_result['H'] / phase,
        'wavenumber': wgmode_result['wavenumber'],
        'axis': axis,
        'omega': omega,
        'polarity': polarity,
        'dxes': dxes,
        'slices': source_slices,
        'mu': vec(mu)
    }
    C = compute_overlap_e(**arg_overlap)

    # Increase/decrease C to emit desired power.
    for k in range(len(C)):
        C[k] *= np.sqrt(power)
    return C


def build_overlap_angle(omega: complex,
                        dxes: List[np.ndarray],
                        eps: List[np.ndarray],
                        mu: List[np.ndarray] or None,
                        axis: Direction or int,
                        waveguide_slice,
                        polarity: int,
                        mode_num: int,
                        power: float,
                        axis_rot: int = None,
                        angle: float = None) -> List[np.ndarray]:
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
        axis_rot: axis around which the waveguide is rotated
            !!! if this is defined the waveguide mode will be calculated with
                the 3D solver. In this case the slices define a periodic cell.
                The dimension in the direction of the axis determines the angle.
        angle: rotation angle
    Returns:
        Overlap vector C.

    """
    if type(waveguide_slice[0]) is not slice:
        slices = [slice(i, f + 1) for i, f in zip(*waveguide_slice)]
    else:
        slices = waveguide_slice

    if type(axis) is Direction:
        axis = axis.value

    if not axis_rot:
        sim_params = {
            'omega': omega,
            'dxes': dxes,
            'axis': axis,
            'slices': slices,
            'polarity': polarity,
            'mu': mu
        }
        wgmode_result = solve_waveguide_mode(
            mode_number=mode_num, epsilon=eps, **sim_params)
    else:
        J_slices = copy.deepcopy(slices)
        if polarity == 1:
            J_slices[axis] = slice(J_slices[axis].stop - 1, J_slices[axis].stop)
        else:
            J_slices[axis] = slice(J_slices[axis].start,
                                   J_slices[axis].start + 1)
        # get an estimate of the wavenumber
        dxes_squeeze = copy.deepcopy(dxes)
        dxes_squeeze[0][3 - axis - axis_rot] *= np.cos(angle)
        dxes_squeeze[1][3 - axis - axis_rot] *= np.cos(angle)
        sim_params = {
            'omega': omega,
            'axis': axis,
            'slices': J_slices,
            'mu': mu
        }
        wgmode_result = solve_waveguide_mode(
            mode_number=mode_num,
            dxes=dxes_squeeze,
            epsilon=eps,
            polarity=polarity,
            **sim_params)
        wavenumber = wgmode_result['wavenumber']

        #calculate the 3D mode profile
        wgmode_result, omega_3d = solve_waveguide_mode_3d(
            mode_number=mode_num,
            dxes=dxes,
            omega_appx=omega,
            wavenumber=wavenumber,
            polarity=polarity,
            axis=axis,
            axis_rot=axis_rot,
            angle=angle,
            slices=slices,
            eps=eps,
            expand_fields=True)
        if np.abs(omega_3d - omega) / omega > 0.02:
            raise ValueError(
                'The 3D mode calculation did not match the 2D calculation')

    source_slices = copy.deepcopy(slices)
    if polarity == 1:
        source_slices[axis] = slice(source_slices[axis].stop - 1,
                                    source_slices[axis].stop)
    else:
        source_slices[axis] = slice(source_slices[axis].start,
                                    source_slices[axis].start + 1)
    arg_overlap = {
        'E': wgmode_result['E'],
        'H': wgmode_result['H'],
        'axis': axis,
        'omega': omega,
        'dxes': dxes,
        'slices': source_slices,
        'mu': vec(mu)
    }
    C = compute_overlap_e_angled(**arg_overlap)

    # Increase/decrease C to emit desired power.
    for k in range(len(C)):
        C[k] *= np.sqrt(power)
    return C


def build_photonic_crystal_waveguide_overlap(
        omega_appx: complex, wavenumber: complex, dxes: List[np.ndarray],
        eps: List[np.ndarray], mu: List[np.ndarray] or None, axis: Direction or
        int, waveguide_slice, polarity: int, mode_num: int,
        power: float) -> List[np.ndarray]:
    """ Builds a TFSF source for translationally-invariant waveguide.

    Args:
        omega_appx: frequency approximation of the mode.
        wavenumber: wavenumber mode.
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
        Overlap vector J.

    """

    if type(waveguide_slice[0]) is not slice:
        slices = [slice(i, f + 1) for i, f in zip(*waveguide_slice)]
    else:
        slices = waveguide_slice

    if type(axis) is Direction:
        axis = axis.value

    source_slices = copy.deepcopy(slices)
    if polarity == 1:
        source_slices[axis] = slice(source_slices[axis].stop - 1,
                                    source_slices[axis].stop)
    else:
        source_slices[axis] = slice(source_slices[axis].start,
                                    source_slices[axis].start + 1)

    #calculate the 3D mode profile
    wgmode_result, omega_3d = solve_waveguide_mode_3d(
        mode_number=mode_num,
        dxes=dxes,
        omega_appx=omega_appx,
        wavenumber=wavenumber,
        polarity=polarity,
        axis=axis,
        axis_rot=(axis + 1) % 3,
        angle=0,
        slices=slices,
        eps=eps,
        expand_fields=True)
    if np.abs(omega_3d - omega_appx) / omega_appx > 0.05:
        raise ValueError(
            'The 3D mode eigenvalue differs strongly from the approximated omega'
        )

    arg_overlap = {
        'E': wgmode_result['E'],
        'H': wgmode_result['H'],
        'axis': axis,
        'omega': omega_3d,
        'dxes': dxes,
        'slices': source_slices,
        'mu': vec(mu)
    }
    C = compute_overlap_e_angled(**arg_overlap)

    # Increase/decrease C to emit desired power.
    for k in range(len(C)):
        C[k] *= np.sqrt(power)

    return C, omega_3d


def expand_fields(e_eig, h_eig, axis, slices, shape, dxes_eig, bloch_vec,
                  shift):
    """ Expand field defined in a region of a simulation space over the entire space
        according to periodicity

        :param e_eig: electric fields
        :param h_eig: magnetic fields
        :param axis: propagation axis (main axis: so 0, 1 or 2)
        :param slices: slices in the simulation space where e_eig and h_eig are defined
        :param shape: shape of the simulation space
        :dxes_eig: dxes of e_eig and h_eig
        :bloch_vec: bloch vector
        :shift: shift matrix to account for an angle
    """
    # expand fields
    h_fields = np.zeros_like(shape)
    e_fields = np.zeros_like(shape)
    i_start = slices[axis].start
    i = i_start
    while i < e_fields[0].shape[axis]:
        sl = copy.deepcopy(slices)
        sl[axis] = slice(i, i + 1)
        for a in range(3):
            e_fields[a][sl] = operators.append_bloch_shift(
                A=e_unvec[a],
                axis=axis,
                ind_axis=i - i_start,
                dx=dxes_eig[0],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift)
            h_fields[a][sl] = operators.append_bloch_shift(
                A=h_unvec[a],
                axis=axis,
                ind_axis=i - i_start,
                dx=dxes_eig[1],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift)
        i += 1
    i = slices[axis].start - 1
    while i >= 0:
        sl = copy.deepcopy(slices)
        sl[axis] = slice(i, i + 1)
        for a in range(3):
            e_fields[a][sl] = operators.append_bloch_shift(
                A=e_unvec[a],
                axis=axis,
                ind_axis=i - i_start,
                dx=dxes_eig[0],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift)
            h_fields[a][sl] = operators.append_bloch_shift(
                A=h_unvec[a],
                axis=axis,
                ind_axis=i - i_start,
                dx=dxes_eig[1],
                bloch_vector=bloch_vec,
                shift_orthogonal=shift)
        i -= 1

    return e_fields, h_fields
