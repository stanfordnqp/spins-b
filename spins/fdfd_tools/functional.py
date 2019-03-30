"""
Functional versions of many FDFD operators. These can be useful for performing
 FDFD calculations without needing to construct large matrices in memory.

The functions generated here expect inputs in the form E = [E_x, E_y, E_z], where each
 component E_* is an ndarray of equal shape.
"""
from typing import List, Callable
import numpy as np

from . import dx_lists_t, field_t

__author__ = 'Jan Petykiewicz'

functional_matrix = Callable[[field_t], field_t]


def curl_h(dxes: dx_lists_t) -> functional_matrix:
    """
    Curl operator for use with the H field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Function for taking the discretized curl of the H-field, F(H) -> curlH
    """
    dxyz_b = np.meshgrid(*dxes[1], indexing='ij')

    def dh(f, ax):
        return (f - np.roll(f, 1, axis=ax)) / dxyz_b[ax]

    def ch_fun(h: field_t) -> field_t:
        e = [
            dh(h[2], 1) - dh(h[1], 2),
            dh(h[0], 2) - dh(h[2], 0),
            dh(h[1], 0) - dh(h[0], 1)
        ]
        return e

    return ch_fun


def curl_e(dxes: dx_lists_t) -> functional_matrix:
    """
    Curl operator for use with the E field.

    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :return: Function for taking the discretized curl of the E-field, F(E) -> curlE
    """
    dxyz_a = np.meshgrid(*dxes[0], indexing='ij')

    def de(f, ax):
        return (np.roll(f, -1, axis=ax) - f) / dxyz_a[ax]

    def ce_fun(e: field_t) -> field_t:
        h = [
            de(e[2], 1) - de(e[1], 2),
            de(e[0], 2) - de(e[2], 0),
            de(e[1], 0) - de(e[0], 1)
        ]
        return h

    return ce_fun


def e_full(omega: complex,
           dxes: dx_lists_t,
           epsilon: field_t,
           mu: field_t = None) -> functional_matrix:
    """
    Wave operator del x (1/mu * del x) - omega**2 * epsilon, for use with E-field,
     with wave equation
    (del x (1/mu * del x) - omega**2 * epsilon) E = -i * omega * J

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: Function implementing the wave operator A(E) -> E
    """
    ch = curl_h(dxes)
    ce = curl_e(dxes)

    def op_1(e):
        curls = ch(ce(e))
        return [c - omega**2 * e * x for c, e, x in zip(curls, epsilon, e)]

    def op_mu(e):
        curls = ch([m * y for m, y in zip(mu, ce(e))])
        return [c - omega**2 * p * x for c, p, x in zip(curls, epsilon, e)]

    if np.any(np.equal(mu, None)):
        return op_1
    else:
        return op_mu


def eh_full(omega: complex,
            dxes: dx_lists_t,
            epsilon: field_t,
            mu: field_t = None) -> functional_matrix:
    """
    Wave operator for full (both E and H) field representation.

    :param omega: Angular frequency of the simulation
    :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
    :param epsilon: Dielectric constant
    :param mu: Magnetic permeability (default 1 everywhere)
    :return: Function implementing the wave operator A(E, H) -> (E, H)
    """
    ch = curl_h(dxes)
    ce = curl_e(dxes)

    def op_1(e, h):
        return ([c - 1j * omega * p * x for c, p, x in zip(ch(h), epsilon, e)],
                [c + 1j * omega * y for c, y in zip(ce(e), h)])

    def op_mu(e, h):
        return ([c - 1j * omega * p * x for c, p, x in zip(ch(h), epsilon, e)],
                [c + 1j * omega * m * y for c, m, y in zip(ce(e), mu, h)])

    if np.any(np.equal(mu, None)):
        return op_1
    else:
        return op_mu


def e2h(
        omega: complex,
        dxes: dx_lists_t,
        mu: field_t = None,
) -> functional_matrix:
    """
   Utility operator for converting the E field into the H field.
   For use with e_full -- assumes that there is no magnetic current M.

   :param omega: Angular frequency of the simulation
   :param dxes: Grid parameters [dx_e, dx_h] as described in fdfd_tools.operators header
   :param mu: Magnetic permeability (default 1 everywhere)
   :return: Function for converting E to H
   """
    A2 = curl_e(dxes)

    def e2h_1_1(e):
        return [y / (-1j * omega) for y in A2(e)]

    def e2h_mu(e):
        return [y / (-1j * omega * m) for y, m in zip(A2(e), mu)]

    if np.any(np.equal(mu, None)):
        return e2h_1_1
    else:
        return e2h_mu
