"""
Sparse matrix operators for use with electromagnetic wave equations.

These functions return sparse-matrix (scipy.sparse.spmatrix) representations of
 a variety of operators, intended for use with E and H fields vectorized using the
 fdfd_tools.vec() and .unvec() functions (column-major/Fortran ordering).

E- and H-field values are defined on a Yee cell; epsilon values should be calculated for
 cells centered at each E component (mu at each H component).

Many of these functions require a 'dxes' parameter, of type fdfd_tools.dx_lists_type,
 which contains grid cell width information in the following format:
 [[[dx_e_0, dx_e_1, ...], [dy_e_0, ...], [dz_e_0, ...]],
  [[dx_h_0, dx_h_1, ...], [dy_h_0, ...], [dz_h_0, ...]]]
 where dx_e_0 is the x-width of the x=0 cells, as used when calculating dE/dx,
 and dy_h_0 is  the y-width of the y=0 cells, as used when calculating dH/dy, etc.


The following operators are included:
- E-only wave operator
- H-only wave operator
- EH wave operator
- Curl for use with E, H fields
- E to H conversion
- M to J conversion
- Poynting cross products

Also available:
- Circular shifts
- Discrete derivatives
- Averaging operators
- Cross product matrices
"""

import copy
from typing import List, Tuple

import numpy as np
import scipy.sparse as sparse

from . import vec, dx_lists_t, vfield_t, GridSpacing


def e_full(omega: complex,
           dxes: dx_lists_t,
           epsilon: vfield_t,
           mu: vfield_t = None,
           pec: vfield_t = None,
           pmc: vfield_t = None,
           bloch_vec: np.ndarray = None,
           shift_orthogonal: np.array = np.zeros((3, 3))) -> sparse.spmatrix:
    """
    Wave operator del x (1/mu * del x) - omega**2 * epsilon, for use with E-field,
     with wave equation
    (del x (1/mu * del x) - omega**2 * epsilon) E = -i * omega * J

    To make this matrix symmetric, use the preconditions from e_full_preconditioners().

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Vectorized dielectric constant
    :param mu: Vectorized magnetic permeability (default 1 everywhere).
    :param pec: Vectorized mask specifying PEC cells. Any cells where pec != 0 are interpreted
        as containing a perfect electrical conductor (PEC).
        The PEC is applied per-field-component (ie, pec.size == epsilon.size)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix containing the wave operator
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    ce = curl_e(dxes, bloch_vec, shift_orthogonal)
    ch = curl_h(dxes, bloch_vec, shift_orthogonal)

    if np.any(np.equal(pec, None)):
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(np.where(pec, 0, 1))  # Set pe to (not PEC)

    if np.any(np.equal(pmc, None)):
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(np.where(pmc, 0, 1))  # set pm to (not PMC)

    e = sparse.diags(epsilon)
    if np.any(np.equal(mu, None)):
        m_div = sparse.eye(epsilon.size)
    else:
        m_div = sparse.diags(1 / mu)

    op = pe @ (ch @ pm @ m_div @ ce - omega**2 * e) @ pe
    return op


def e_full_preconditioners(
        dxes: dx_lists_t) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    """
    Left and right preconditioners (Pl, Pr) for symmetrizing the e_full wave operator.

    The preconditioned matrix A_symm = (Pl @ A @ Pr) is complex-symmetric
     (non-Hermitian unless there is no loss or PMLs).

    The preconditioner matrices are diagonal and complex, with Pr = 1 / Pl

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Preconditioner matrices (Pl, Pr)
    """
    p_squared = [
        dxes[0][0][:, None, None] * dxes[1][1][None, :, None] *
        dxes[1][2][None, None, :], dxes[1][0][:, None, None] *
        dxes[0][1][None, :, None] * dxes[1][2][None, None, :],
        dxes[1][0][:, None, None] * dxes[1][1][None, :, None] *
        dxes[0][2][None, None, :]
    ]

    p_vector = np.sqrt(vec(p_squared))
    P_left = sparse.diags(p_vector)
    P_right = sparse.diags(1 / p_vector)
    return P_left, P_right


def h_full(omega: complex,
           dxes: dx_lists_t,
           epsilon: vfield_t,
           mu: vfield_t = None,
           pec: vfield_t = None,
           pmc: vfield_t = None,
           bloch_vec: np.ndarray = None,
           shift_orthogonal: np.array = np.zeros((3, 3))) -> sparse.spmatrix:
    """
    Wave operator del x (1/epsilon * del x) - omega**2 * mu, for use with H-field,
     with wave equation
    (del x (1/epsilon * del x) - omega**2 * mu) H = i * omega * M

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Vectorized dielectric constant
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :param pec: Vectorized mask specifying PEC cells. Any cells where pec != 0 are interpreted
        as containing a perfect electrical conductor (PEC).
        The PEC is applied per-field-component (ie, pec.size == epsilon.size)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix containing the wave operator
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    ec = curl_e(dxes, bloch_vec, shift_orthogonal)
    hc = curl_h(dxes, bloch_vec, shift_orthogonal)

    if np.any(np.equal(pec, None)):
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(np.where(pec, 0, 1))  # set pe to (not PEC)

    if np.any(np.equal(pmc, None)):
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(np.where(pmc, 0, 1))  # Set pe to (not PMC)

    e_div = sparse.diags(1 / epsilon)
    if mu is None:
        m = sparse.eye(epsilon.size)
    else:
        m = sparse.diags(mu)

    A = pm @ (ec @ pe @ e_div @ hc - omega**2 * m) @ pm
    return A


def eh_full(omega: complex,
            dxes: dx_lists_t,
            epsilon: vfield_t,
            mu: vfield_t = None,
            pec: vfield_t = None,
            pmc: vfield_t = None,
            bloch_vec: np.ndarray = None,
            shift_orthogonal: np.array = np.zeros((3, 3))):
    """
    Wave operator for [E, H] field representation. This operator implements Maxwell's
     equations without cancelling out either E or H. The operator is
     [[-i * omega * epsilon,  del x],
      [del x, i * omega * mu]]

    for use with a field vector of the form hstack(vec(E), vec(H)).

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Vectorized dielectric constant
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :param pec: Vectorized mask specifying PEC cells. Any cells where pec != 0 are interpreted
        as containing a perfect electrical conductor (PEC).
        The PEC is applied per-field-component (ie, pec.size == epsilon.size)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix containing the wave operator
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    if np.any(np.equal(pec, None)):
        pe = sparse.eye(epsilon.size)
    else:
        pe = sparse.diags(np.where(pec, 0, 1))  # set pe to (not PEC)

    if np.any(np.equal(pmc, None)):
        pm = sparse.eye(epsilon.size)
    else:
        pm = sparse.diags(np.where(pmc, 0, 1))  # set pm to (not PMC)

    iwe = pe @ (1j * omega * sparse.diags(epsilon)) @ pe
    iwm = 1j * omega
    if not np.any(np.equal(mu, None)):
        iwm *= sparse.diags(mu)
    iwm = pm @ iwm @ pm

    A1 = pe @ curl_h(dxes, bloch_vec, shift_orthogonal) @ pm
    A2 = pm @ curl_e(dxes, bloch_vec, shift_orthogonal) @ pe

    A = sparse.bmat([[-iwe, A1], [A2, iwm]])
    return A


def curl_h(dxes: dx_lists_t,
           bloch_vec: np.ndarray = None,
           shift_orthogonal: np.array = np.zeros((3, 3))) -> sparse.spmatrix:
    """
    Curl operator for use with the H field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix for taking the discretized curl of the H-field
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    return cross(deriv_back_shift(dxes[1], bloch_vec, shift_orthogonal))


def curl_e(dxes: dx_lists_t,
           bloch_vec: np.ndarray = None,
           shift_orthogonal: np.array = np.zeros((3, 3))) -> sparse.spmatrix:
    """
    Curl operator for use with the E field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix for taking the discretized curl of the E-field
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    return cross(deriv_forward_shift(dxes[0], bloch_vec, shift_orthogonal))


def e2h(omega: complex,
        dxes: dx_lists_t,
        mu: vfield_t = None,
        pmc: vfield_t = None,
        bloch_vec: np.ndarray = np.zeros(3),
        shift_orthogonal: np.array = np.zeros((3, 3))) -> sparse.spmatrix:
    """
    Utility operator for converting the E field into the H field.
    For use with e_full -- assumes that there is no magnetic current M.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix for converting E to H
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    op = curl_e(dxes, bloch_vec, shift_orthogonal) / (-1j * omega)

    if not np.any(np.equal(mu, None)):
        op = sparse.diags(1 / mu) @ op

    if not np.any(np.equal(pmc, None)):
        op = sparse.diags(np.where(pmc, 0, 1)) @ op

    return op


def h2e(omega: complex,
        dxes: dx_lists_t,
        eps: vfield_t,
        pmc: vfield_t = None,
        bloch_vec: np.ndarray = np.zeros(3),
        shift_orthogonal: np.array = np.zeros((3, 3))) -> sparse.spmatrix:
    """
    Utility operator for converting the H field into the E field.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param eps: Vectorized permittivity
    :param pmc: Vectorized mask specifying PMC cells. Any cells where pmc != 0 are interpreted
        as containing a perfect magnetic conductor (PMC).
        The PMC is applied per-field-component (ie, pmc.size == epsilon.size)
    :param bloch_vec: bloch vector [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix for converting H to E
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    op = sparse.diags(1 / (1j * omega * eps)) @ curl_h(dxes, bloch_vec,
                                                       shift_orthogonal)

    #TODO: implement pmc
    #if not np.any(np.equal(pmc, None)):
    #    op = sparse.diags(np.where(pmc, 0, 1)) @ op

    return op


def m2j(omega: complex,
        dxes: dx_lists_t,
        bloch_vec: np.ndarray = np.zeros(3),
        mu: vfield_t = None,
        shift_orthogonal: np.array = np.zeros((3, 3))):
    """
    Utility operator for converting M field into J.
    Converts a magnetic current M into an electric current J.
    For use with eg. e_full.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param mu: Vectorized magnetic permeability (default 1 everywhere)
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: Sparse matrix for converting E to H
    """
    op = curl_h(dxes, bloch_vec, shift_orthogonal) / (1j * omega)

    if not np.any(np.equal(mu, None)):
        op = op @ sparse.diags(1 / mu)

    return op


def shift_with_mirror(axis: int, shape: List[int],
                      shift_distance: int = 1) -> sparse.spmatrix:
    """
    Utility operator for performing an n-element shift along a specified axis, with mirror
    boundary conditions applied to the cells beyond the receding edge.

    :param axis: Axis to shift along. x=0, y=1, z=2
    :param shape: Shape of the grid being shifted
    :param shift_distance: Number of cells to shift by. May be negative. Default 1.
    :return: Sparse matrix for performing the circular shift
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))
    if axis not in range(len(shape)):
        raise Exception('Invalid direction: {}, shape is {}'.format(
            axis, shape))
    if shift_distance >= shape[axis]:
        raise Exception('Shift ({}) is too large for axis {} of size {}'.format(
            shift_distance, axis, shape[axis]))

    def mirrored_range(n, s):
        v = np.arange(n) + s
        v = np.where(v >= n, 2 * n - v - 1, v)
        v = np.where(v < 0, -1 - v, v)
        return v

    shifts = [shift_distance if a == axis else 0 for a in range(3)]
    shifted_diags = [mirrored_range(n, s) for n, s in zip(shape, shifts)]
    ijk = np.meshgrid(*shifted_diags, indexing='ij')

    n = np.prod(shape)
    i_ind = np.arange(n)
    j_ind = ijk[0] + ijk[1] * shape[0]
    if len(shape) == 3:
        j_ind += ijk[2] * shape[0] * shape[1]

    vij = (np.ones(n), (i_ind, j_ind.flatten(order='F')))

    d = sparse.csr_matrix(vij, shape=(n, n))
    return d


def cross(B: List[sparse.spmatrix]) -> sparse.spmatrix:
    """
    Cross product operator

    :param B: List [Bx, By, Bz] of sparse matrices corresponding to the x, y, z
            portions of the operator on the left side of the cross product.
    :return: Sparse matrix corresponding to (B x), where x is the cross product
    """
    n = B[0].shape[0]
    zero = sparse.csr_matrix((n, n))
    return sparse.bmat([[zero, -B[2], B[1]], [B[2], zero, -B[0]],
                        [-B[1], B[0], zero]])


def vec_cross(b: vfield_t) -> sparse.spmatrix:
    """
    Vector cross product operator

    :param b: Vector on the left side of the cross product
    :return: Sparse matrix corresponding to (b x), where x is the cross product
    """
    if len(b.shape) == 1:
        n = b.shape[0] // 3
        B = [
            sparse.diags(b[0:n]),
            sparse.diags(b[n:2 * n]),
            sparse.diags(b[2 * n:3 * n])
        ]
    elif b.shape[1] == 3:
        B = [sparse.diags(c) for c in np.split(b, 3)]
    return cross(B)


def avgf(axis: int, shape: List[int]) -> sparse.spmatrix:
    """
    Forward average operator (x4 = (x4 + x5) / 2)

    :param axis: Axis to average along (x=0, y=1, z=2)
    :param shape: Shape of the grid to average
    :return: Sparse matrix for forward average operation
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))

    n = np.prod(shape)
    return 0.5 * (sparse.eye(n) + rotation_bloch_shift(axis, shape, 1))


def avgb(axis: int, shape: List[int]) -> sparse.spmatrix:
    """
    Backward average operator (x4 = (x4 + x3) / 2)

    :param axis: Axis to average along (x=0, y=1, z=2)
    :param shape: Shape of the grid to average
    :return: Sparse matrix for backward average operation
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))

    n = np.prod(shape)
    return 0.5 * (sparse.eye(n) + rotation_bloch_shift(axis, shape, -1))


def rotation_bloch(axis: int,
                   shape: List[int],
                   bloch_phase: complex = 0,
                   shift_distance: int = 1):
    '''
    DEPRECATED: use the rotation with bloch and shift
    Operator for performing circular shift by 1 element along the specified axis and adds bloch phase

    :param axis: Axist to shift along. x = 0, y = 1, z = 2
    :param shape: Shape of the grid being shifted
    :param bloch_phase: bloch vector component along axis multiplied by length of simulation region
    :param shift_distance: Number of cells to shift by. May be negative. Defautl is 1
    :return sparse matrix for performing the circular shift
    '''
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))
    if axis not in range(len(shape)):
        raise Exception('Invalid direction: {}, shape is {}'.format(
            axis, shape))
    if shift_distance not in (-1, 1):
        raise Exception('Shift must be in (-1,1)')

    # Setting up shifted matrices for both boundary cells and interior cells
    n = np.prod(shape)
    shifts = [1 if a == axis else 0 for a in range(3)]
    shifted_diags = [(np.arange(m) + s) % m for m, s in zip(shape, shifts)]
    shifted_diags_bloch = [
        np.array([0]) if a == axis else np.arange(m)
        for a, m in zip(range(3), shape)
    ]
    diags_bloch = [
        np.array([shape[a] - 1]) if a == axis else np.arange(m)
        for a, m in zip(range(3), shape)
    ]
    ijk_bloch = np.meshgrid(*shifted_diags_bloch, indexing='ij')
    ijk_inp_bloch = np.meshgrid(*diags_bloch, indexing='ij')
    ijk = np.meshgrid(*shifted_diags, indexing='ij')

    # Assembling the rotation matrix
    i_ind = np.arange(n)
    j_ind = ijk[0] + ijk[1] * shape[0]
    i_ind_bloch = ijk_inp_bloch[0] + ijk_inp_bloch[1] * shape[0]
    j_ind_bloch = ijk_bloch[0] + ijk_bloch[1] * shape[0]
    if (len(shape) == 3):
        j_ind = j_ind + ijk[2] * shape[0] * shape[1]
        i_ind_bloch = i_ind_bloch + ijk_inp_bloch[2] * shape[0] * shape[1]
        j_ind_bloch = j_ind_bloch + ijk_bloch[2] * shape[0] * shape[1]

    vij = (np.ones(n), (i_ind, j_ind.flatten(order='F')))
    d = sparse.csr_matrix(vij, shape=(n, n))

    vij_bloch = (np.ones(len(
        j_ind_bloch.flatten(order='F'))), (i_ind_bloch.flatten(order='F'),
                                           j_ind_bloch.flatten(order='F')))
    d_bloch = sparse.csr_matrix(vij_bloch, shape=(n, n))

    # Calculating the final matrix by subtracting the boundary element and adding them with the proper phase
    df = d - d_bloch + np.exp(-1.0j * bloch_phase) * d_bloch

    if (shift_distance == -1):
        df = df.conj().T

    return (df)


def deriv_forward(dx_e: List[np.ndarray],
                  bloch_vec: np.ndarray = None) -> List[sparse.spmatrix]:
    """
    DEPRECATED: use the rotation with bloch and shift
    Utility operators for taking discretized derivatives (forward variant).

    :param dx_e: Lists of cell sizes for all axes [[dx_0, dx_1, ...], ...].
    :param bloch_vec: bloch vector - [kx,ky,kz]
    :return: List of operators for taking forward derivatives along each axis.
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    shape = [s.size for s in dx_e]
    n = np.prod(shape)
    phase = [bloch_vec[n] * np.sum(dx_e[n]) for n in range(0, len(dx_e))]
    dx_e_expanded = np.meshgrid(*dx_e, indexing='ij')

    def deriv(axis, phase):
        return rotation_bloch(axis, shape, phase, 1) - sparse.eye(n)

    Ds = [
        sparse.diags(+1 / dx.flatten(order='F')) @ deriv(a, phase[a])
        for a, dx in enumerate(dx_e_expanded)
    ]

    return Ds


def deriv_back(dx_h: List[np.ndarray],
               bloch_vec: np.ndarray = None) -> List[sparse.spmatrix]:
    """
    DEPRECATED: use the rotation with bloch and shift
    Utility operators for taking discretized derivatives (backward variant).

    :param dx_h: Lists of cell sizes for all axes [[dx_0, dx_1, ...], ...].
    :param bloch_vec: bloch vector - [kx, ky, kz]
    :return: List of operators for taking forward derivatives along each axis.
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    shape = [s.size for s in dx_h]
    n = np.prod(shape)
    phase = [bloch_vec[n] * np.sum(dx_h[n]) for n in range(0, len(dx_h))]
    dx_h_expanded = np.meshgrid(*dx_h, indexing='ij')

    def deriv(axis, phase):
        return rotation_bloch(axis, shape, phase, -1) - sparse.eye(n)

    Ds = [
        sparse.diags(-1 / dx.flatten(order='F')) @ deriv(a, phase[a])
        for a, dx in enumerate(dx_h_expanded)
    ]

    return Ds


def rotation_bloch_shift(axis: int,
                         shape: List[int],
                         shift_distance: int = 1,
                         bloch_phase: np.array = np.zeros(3),
                         shift_orthogonal: List[int] = [0, 0, 0]):
    '''
    Operator for performing circular shift by 1 element along the specified axis and adds bloch phase

    :param axis: Axist to shift along. x = 0, y = 1, z = 2
    :param shape: Shape of the grid being shifted
    :param bloch_phase: bloch vector component along axis multiplied by length of simulation region
    :param shift_distance: Number of cells to shift by. May be negative. Defautl is 1
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return sparse matrix for performing the circular shift
    '''

    if shift_orthogonal[axis] != 0:
        raise ValueError('Orthogonal shift defined in the axis direction')

    # make index grid
    i = np.arange(0, shape[0])
    j = np.arange(0, shape[1])
    k = np.arange(0, shape[2])
    ind0 = np.meshgrid(i, j, k, indexing='ij')
    ind = np.meshgrid(i, j, k, indexing='ij')

    # apply shift to the ind grid and adjust according to shift_orthogonal
    ind[axis] += shift_distance

    ind[0][ind[axis] >= shape[axis]] += int(shift_orthogonal[0])
    ind[1][ind[axis] >= shape[axis]] += int(shift_orthogonal[1])
    ind[2][ind[axis] >= shape[axis]] += int(shift_orthogonal[2])
    ind[0][ind[axis] < 0] += -int(shift_orthogonal[0])
    ind[1][ind[axis] < 0] += -int(shift_orthogonal[1])
    ind[2][ind[axis] < 0] += -int(shift_orthogonal[2])

    # prepare data with bloch_phase
    data = np.ones_like(ind[0]).astype(complex)
    for a in range(3):
        data[ind[a] >= shape[a]] *= np.exp(-1.0j * bloch_phase[a])
        data[ind[a] < 0] *= np.exp(-1.0j * (-bloch_phase[a]))

    # build rotation matrix
    row_ind = np.ravel_multi_index(
        ind0, shape, mode='wrap', order='F').flatten(order='F')
    col_ind = np.ravel_multi_index(
        ind, shape, mode='wrap', order='F').flatten(order='F')
    A = sparse.csr_matrix((data.flatten(order='F'), (row_ind, col_ind)))

    return A


def deriv_forward_shift(dx_e: List[np.ndarray],
                        bloch_vec: np.ndarray = None,
                        shift_orthogonal: np.array = np.zeros(
                            (3, 3))) -> List[sparse.spmatrix]:
    """
    Utility operators for taking discretized derivatives (forward variant).

    :param dx_e: Lists of cell sizes for all axes [[dx_0, dx_1, ...], ...].
    :param bloch_vec: bloch vector - [kx,ky,kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: List of operators for taking forward derivatives along each axis.
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    shape = [s.size for s in dx_e]
    n = np.prod(shape)
    L_sim = np.array([np.sum(dx) for dx in dx_e])
    dx_0 = np.array([dx[0] for dx in dx_e])
    shift_distances = (np.diag(L_sim) - shift_orthogonal @ np.diag(dx_0))
    phase = shift_distances @ bloch_vec
    dx_e_expanded = np.meshgrid(*dx_e, indexing='ij')

    def deriv(axis):
        return rotation_bloch_shift(axis, shape, 1, phase,
                                    shift_orthogonal[axis]) - sparse.eye(n)

    Ds = [
        sparse.diags(+1 / dx.flatten(order='F')) @ deriv(a)
        for a, dx in enumerate(dx_e_expanded)
    ]

    return Ds


def deriv_back_shift(dx_h: List[np.ndarray],
                     bloch_vec: np.ndarray = None,
                     shift_orthogonal: np.array = np.zeros(
                         (3, 3))) -> List[sparse.spmatrix]:
    """
    Utility operators for taking discretized derivatives (backward variant).

    :param dx_h: Lists of cell sizes for all axes [[dx_0, dx_1, ...], ...].
    :param bloch_vec: bloch vector - [kx, ky, kz]
    :shift_orthogonal: shifts orthogonal to the axis directions to be taken into
                        account when applying periodc boundary conditions (the
                        diagonal can only contain zeros)
    :return: List of operators for taking forward derivatives along each axis.
    """
    if bloch_vec is None:
        bloch_vec = np.zeros(3)

    shape = [s.size for s in dx_h]
    n = np.prod(shape)
    L_sim = np.array([np.sum(dx) for dx in dx_h])
    dx_0 = np.array([dx[0] for dx in dx_h])
    shift_distances = (np.diag(L_sim) - shift_orthogonal @ np.diag(dx_0))
    phase = shift_distances @ bloch_vec
    dx_h_expanded = np.meshgrid(*dx_h, indexing='ij')

    def deriv(axis):
        return rotation_bloch_shift(axis, shape, -1, phase,
                                    shift_orthogonal[axis]) - sparse.eye(n)

    Ds = [
        sparse.diags(-1 / dx.flatten(order='F')) @ deriv(a)
        for a, dx in enumerate(dx_h_expanded)
    ]

    return Ds


def poynting_e_cross(e: vfield_t, dxes: dx_lists_t) -> sparse.spmatrix:
    """

    Operator for computing the Poynting vector, contining the (E x) portion of the Poynting vector
    (except it actually contains dA).

    Don't come here unless it's absolutely necessary.

    But if you really want to know, the Poynting vector includes the surface element dA
    so that you can simply sum the Poynting vector to get the power.

    :param e: Vectorized E-field for the ExH cross product
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Sparse matrix containing (E x) portion of Poynting cross product
    """
    shape = [len(dx) for dx in dxes[0]]

    fx, fy, fz = [avgf(i, shape) for i in range(3)]
    bx, by, bz = [avgb(i, shape) for i in range(3)]

    dxag = [
        dx.flatten(order='F') for dx in np.meshgrid(*dxes[0], indexing='ij')
    ]
    dbgx, dbgy, dbgz = [
        sparse.diags(dx.flatten(order='F'))
        for dx in np.meshgrid(*dxes[1], indexing='ij')
    ]

    Ex, Ey, Ez = [sparse.diags(ei * da) for ei, da in zip(np.split(e, 3), dxag)]

    n = np.prod(shape)
    zero = sparse.csr_matrix((n, n))

    P = sparse.bmat([[zero, -fx @ Ez @ bz @ dbgy, fx @ Ey @ by @ dbgz],
                     [fy @ Ez @ bz @ dbgx, zero, -fy @ Ex @ bx @ dbgz],
                     [-fz @ Ey @ by @ dbgx, fz @ Ex @ bx @ dbgy, zero]])
    return P


def poynting_h_cross(h: vfield_t, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Operator for computing the Poynting vector, containing the (H x) portion of the Poynting vector.

    :param h: Vectorized H-field for the HxE cross product
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Sparse matrix containing (H x) portion of Poynting cross product
    """
    shape = [len(dx) for dx in dxes[0]]

    fx, fy, fz = [avgf(i, shape) for i in range(3)]
    bx, by, bz = [avgb(i, shape) for i in range(3)]

    dxbg = [
        dx.flatten(order='F') for dx in np.meshgrid(*dxes[1], indexing='ij')
    ]
    dagx, dagy, dagz = [
        sparse.diags(dx.flatten(order='F'))
        for dx in np.meshgrid(*dxes[0], indexing='ij')
    ]

    Hx, Hy, Hz = [sparse.diags(hi * db) for hi, db in zip(np.split(h, 3), dxbg)]

    n = np.prod(shape)
    zero = sparse.csr_matrix((n, n))

    P = sparse.bmat([[zero, -by @ Hz @ fx @ dagy, bz @ Hy @ fx @ dagz],
                     [bx @ Hz @ fy @ dagx, zero, -bz @ Hx @ fy @ dagz],
                     [-bx @ Hy @ fz @ dagx, by @ Hx @ fz @ dagy, zero]])
    return P


def poynting_chew_e_cross(efield: np.ndarray,
                          dxes: GridSpacing) -> sparse.spmatrix:
    """Computes a matrix for computing Poynting vector.

    This function produces a matrix [Ex] such that `S = [Ex] @ conj(hfield)`
    gives the Poynting vector, normalized by the area element, i.e. the total
    power over a plane is simply the sum over `S` without further need to
    add any terms that involve the Yee grid spacing.

    The Poynting vector definition is taken from
    W.C. Chew. Electromagnetic theory on a lattice (1994).

    Args:
        efield: Electric field to use.
        dxes: Grid spacing.

    Returns:
        The `[Ex]` matrix.
    """
    shp = [len(dx) for dx in dxes[0]]
    shift_op = [rotation_bloch_shift(i, shp) for i in range(3)]

    e_i = np.split(efield, 3)
    zeros = sparse.csr_matrix(2 * (len(e_i[0]),))
    dia = lambda x: sparse.diags(x, shape=2 * (len(e_i[0]),))

    dxbg = [
        dx.flatten(order='F') for dx in np.meshgrid(*dxes[0], indexing='ij')
    ]
    dagx, dagy, dagz = [
        sparse.diags(dx.flatten(order='F'))
        for dx in np.meshgrid(*dxes[0], indexing='ij')
    ]
    # yapf:disable
    area = [[dagy @ dagz, zeros, zeros],
            [zeros, dagx @ dagz, zeros],
            [zeros, zeros, dagx @ dagy]]

    block_e_cross = [
        [zeros, -dia(shift_op[0] @ e_i[2]), dia(shift_op[0] @ e_i[1])],
        [dia(shift_op[1] @ e_i[2]), zeros, -dia(shift_op[1] @ e_i[0])],
        [-dia(shift_op[2] @ e_i[1]), dia(shift_op[2] @ e_i[0]), zeros]
    ]
    # yapf:enable

    return sparse.bmat(area) @ sparse.bmat(block_e_cross)


def poynting_chew_h_cross(hfield: np.ndarray,
                          dxes: GridSpacing) -> sparse.spmatrix:
    """Computes a matrix for computing Poynting vector.

    This function produces a matrix [Hx] such that `S = -conj([Hx]) @ efield`
    gives the Poynting vector, normalized by the area element, i.e. the total
    power over a plane is simply the sum over `S` without further need to
    add any terms that involve the Yee grid spacing.

    The Poynting vector definition is taken from
    W.C. Chew. Electromagnetic theory on a lattice (1994).

    Args:
        hfield: Magnetic field to use.
        dxes: Grid spacing.

    Returns:
        The `[Hx]` matrix.
    """
    shp = [len(dx) for dx in dxes[0]]
    shift_op = [rotation_bloch_shift(i, shp) for i in range(3)]

    h_i = np.split(hfield, 3)
    zeros = sparse.csr_matrix(2 * (len(h_i[0]),))
    dia = lambda x: sparse.diags(x, shape=2 * (len(h_i[0]),))

    dxbg = [
        dx.flatten(order="F") for dx in np.meshgrid(*dxes[0], indexing="ij")
    ]
    dagx, dagy, dagz = [
        sparse.diags(dx.flatten(order="F"))
        for dx in np.meshgrid(*dxes[0], indexing="ij")
    ]

    # yapf:disable
    area = [[dagy @ dagz, zeros, zeros],
            [zeros, dagx @ dagz, zeros],
            [zeros, zeros, dagx @ dagy]]

    block_h_cross = [
        [zeros, -dia(h_i[2]) @ shift_op[0], dia(h_i[1]) @ shift_op[0]],
        [dia(h_i[2]) @ shift_op[1], zeros, -dia(h_i[0]) @ shift_op[1]],
        [-dia(h_i[1]) @ shift_op[2], dia(h_i[0]) @ shift_op[2], zeros]]
    # yapf:enable

    return sparse.bmat(area) @ sparse.bmat(block_h_cross)
