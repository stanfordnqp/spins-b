"""
Various operators and helper functions for solving for waveguide modes.

Assuming a z-dependence of the from exp(-i * wavenumber * z), we can simplify Maxwell's
 equations in the absence of sources to the form

A @ [H_x, H_y] = wavenumber**2 * [H_x, H_y]

with A =
omega**2 * epsilon * mu +
epsilon * [[-Dy], [Dx]] / epsilon * [-Dy, Dx] +
[[Dx], [Dy]] / mu * [Dx, Dy] * mu

which is the form used in this file.

As the z-dependence is known, all the functions in this file assume a 2D grid
 (ie. dxes = [[[dx_e_0, dx_e_1, ...], [dy_e_0, ...]], [[dx_h_0, ...], [dy_h_0, ...]]])
 with propagation along the z axis.
"""

from typing import List, Tuple
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sparse

from . import unvec, dx_lists_t, field_t, vfield_t
from . import operators

__author__ = 'Jan Petykiewicz'


def operator(
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfield_t,
        mu: vfield_t = None,
) -> sparse.spmatrix:
    """
    Waveguide operator of the form

    omega**2 * epsilon * mu +
    epsilon * [[-Dy], [Dx]] / epsilon * [-Dy, Dx] +
    [[Dx], [Dy]] / mu * [Dx, Dy] * mu

    for use with a field vector of the form [H_x, H_y].

    This operator can be used to form an eigenvalue problem of the form
    A @ [H_x, H_y] = wavenumber**2 * [H_x, H_y]

    which can then be solved for the eigenmodes of the system (an exp(-i * wavenumber * z)
    z-dependence is assumed for the fields).

    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representation of the operator
    """
    if np.any(np.equal(mu, None)):
        mu = np.ones_like(epsilon)

    Dfx, Dfy = operators.deriv_forward(dxes[0])
    Dbx, Dby = operators.deriv_back(dxes[1])

    eps_parts = np.split(epsilon, 3)
    eps_yx = sparse.diags(np.hstack((eps_parts[1], eps_parts[0])))
    eps_z_inv = sparse.diags(1 / eps_parts[2])

    mu_parts = np.split(mu, 3)
    mu_xy = sparse.diags(np.hstack((mu_parts[0], mu_parts[1])))
    mu_z_inv = sparse.diags(1 / mu_parts[2])

    op = omega ** 2 * eps_yx @ mu_xy + \
        eps_yx @ sparse.vstack((-Dfy, Dfx)) @ eps_z_inv @ sparse.hstack((-Dby, Dbx)) + \
        sparse.vstack((Dbx, Dby)) @ mu_z_inv @ sparse.hstack((Dfx, Dfy)) @ mu_xy

    # We return the operator as a real operator - For unknown reasons the, even with the
    # bloch vector set to 0, returning a complex operator seems to interfere with the eigen solve
    return op.astype(float)


def normalized_fields(v: np.ndarray,
                      wavenumber: complex,
                      omega: complex,
                      dxes: dx_lists_t,
                      epsilon: vfield_t,
                      mu: vfield_t = None) -> Tuple[vfield_t, vfield_t]:
    """
    Given a vector v containing the vectorized H_x and H_y fields,
     returns normalized, vectorized E and H fields for the system.

    :param v: Vector containing H_x and H_y fields
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Normalized, vectorized (e, h) containing all vector components.
    """
    e = v2e(v, wavenumber, omega, dxes, epsilon, mu=mu)
    h = v2h(v, wavenumber, dxes, mu=mu)

    shape = [s.size for s in dxes[0]]
    dxes_real = [[
        np.real(d) for d in np.meshgrid(*dxes[v], indexing='ij')
    ] for v in (0, 1)]

    E = unvec(e, shape)
    H = unvec(h, shape)

    S1 = E[0] * np.roll(
        np.conj(H[1]), 1, axis=0) * dxes_real[0][1] * dxes_real[1][0]
    S2 = E[1] * np.roll(
        np.conj(H[0]), 1, axis=1) * dxes_real[0][0] * dxes_real[1][1]
    S = 0.5 * (
        (S1 + np.roll(S1, 1, axis=0)) - (S2 + np.roll(S2, 1, axis=1)))
    P = 0.5 * np.real(S.sum())
    assert P > 0, 'Found a mode propagating in the wrong direction! P={}'.format(
        P)

    norm_amplitude = 1 / np.sqrt(P)
    norm_angle = -np.angle(e[e.size // 2])
    norm_factor = norm_amplitude * np.exp(1j * norm_angle)

    e *= norm_factor
    h *= norm_factor

    return e, h


def v2h(v: np.ndarray,
        wavenumber: complex,
        dxes: dx_lists_t,
        mu: vfield_t = None) -> vfield_t:
    """
    Given a vector v containing the vectorized H_x and H_y fields,
     returns a vectorized H including all three H components.

    :param v: Vector containing H_x and H_y fields
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Vectorized H field with all vector components
    """
    Dfx, Dfy = operators.deriv_forward(dxes[0])
    op = sparse.hstack((Dfx, Dfy))

    if not np.any(np.equal(mu, None)):
        mu_parts = np.split(mu, 3)
        mu_xy = sparse.diags(np.hstack((mu_parts[0], mu_parts[1])))
        mu_z_inv = sparse.diags(1 / mu_parts[2])

        op = mu_z_inv @ op @ mu_xy

    w = op @ v / (1j * wavenumber)
    return np.hstack((v, w)).flatten()


def v2e(v: np.ndarray,
        wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        epsilon: vfield_t,
        mu: vfield_t = None) -> vfield_t:
    """
    Given a vector v containing the vectorized H_x and H_y fields,
     returns a vectorized E containing all three E components

    :param v: Vector containing H_x and H_y fields
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Vectorized E field with all vector components.
    """
    h2eop = h2e(wavenumber, omega, dxes, epsilon)
    return h2eop @ v2h(v, wavenumber, dxes, mu)


def e2h(wavenumber: complex,
        omega: complex,
        dxes: dx_lists_t,
        mu: vfield_t = None) -> sparse.spmatrix:
    """
    Returns an operator which, when applied to a vectorized E eigenfield, produces
     the vectorized H eigenfield.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Sparse matrix representation of the operator
    """
    op = curl_e(wavenumber, dxes) / (-1j * omega)
    if not np.any(np.equal(mu, None)):
        op = sparse.diags(1 / mu) @ op
    return op


def h2e(wavenumber: complex, omega: complex, dxes: dx_lists_t,
        epsilon: vfield_t) -> sparse.spmatrix:
    """
    Returns an operator which, when applied to a vectorized H eigenfield, produces
     the vectorized E eigenfield.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param epsilon: Vectorized dielectric constant grid
    :return: Sparse matrix representation of the operator
    """
    op = sparse.diags(1 / (1j * omega * epsilon)) @ curl_h(wavenumber, dxes)
    return op


def curl_e(wavenumber: complex, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Discretized curl operator for use with the waveguide E field.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :return: Sparse matrix representation of the operator
    """
    n = 1
    for d in dxes[0]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dfx, Dfy = operators.deriv_forward(dxes[0])
    return operators.cross([Dfx, Dfy, Bz])


def curl_h(wavenumber: complex, dxes: dx_lists_t) -> sparse.spmatrix:
    """
    Discretized curl operator for use with the waveguide H field.

    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :return: Sparse matrix representation of the operator
    """
    n = 1
    for d in dxes[1]:
        n *= len(d)

    Bz = -1j * wavenumber * sparse.eye(n)
    Dbx, Dby = operators.deriv_back(dxes[1])
    return operators.cross([Dbx, Dby, Bz])


def h_err(h: vfield_t,
          wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfield_t,
          mu: vfield_t = None) -> float:
    """
    Calculates the relative error in the H field

    :param h: Vectorized H field
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Relative error norm(OP @ h) / norm(h)
    """
    ce = curl_e(wavenumber, dxes)
    ch = curl_h(wavenumber, dxes)

    eps_inv = sparse.diags(1 / epsilon)

    if np.any(np.equal(mu, None)):
        op = ce @ eps_inv @ ch @ h - omega**2 * h
    else:
        op = ce @ eps_inv @ ch @ h - omega**2 * (mu * h)

    return norm(op) / norm(h)


def e_err(e: vfield_t,
          wavenumber: complex,
          omega: complex,
          dxes: dx_lists_t,
          epsilon: vfield_t,
          mu: vfield_t = None) -> float:
    """
    Calculates the relative error in the E field

    :param e: Vectorized E field
    :param wavenumber: Wavenumber satisfying A @ v == wavenumber**2 * v
    :param omega: The angular frequency of the system
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header (2D)
    :param epsilon: Vectorized dielectric constant grid
    :param mu: Vectorized magnetic permeability grid (default 1 everywhere)
    :return: Relative error norm(OP @ e) / norm(e)
    """
    ce = curl_e(wavenumber, dxes)
    ch = curl_h(wavenumber, dxes)

    if np.any(np.equal(mu, None)):
        op = ch @ ce @ e - omega**2 * (epsilon * e)
    else:
        mu_inv = sparse.diags(1 / mu)
        op = ch @ mu_inv @ ce @ e - omega**2 * (epsilon * e)

    return norm(op) / norm(e)
