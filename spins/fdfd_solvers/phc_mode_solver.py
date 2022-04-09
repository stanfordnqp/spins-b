import numpy as np
from matplotlib import pyplot
import scipy

from spins.fdfd_tools import unvec, vec, dx_lists_t, vfield_t
from spins.fdfd_tools import operators

from .local_matrix_solvers import DirectSolver


def efield_operator(bloch_vec, dxes: dx_lists_t, epsilon: vfield_t,
                    mu: vfield_t, shift_orthogonal: np.array):
    '''
    Function to setup the electric field operator
    for eigen value solve.

    The eigen value equation that is being solved here is
      (1/sqrt(epsilon))*nabla x nabla x (1/sqrt(epsilon))v = omega^2 v

    where E = sqrt(epsilon) v and omega is the eigen-frequency

    INPUTS:
    bloch_vec - bloch vector
    dxes - grid dx
    epsilon - permittivity vector
    mu - permeability vector (assumed to be 1)
    TODO (rahul) - factor in permeability

    OUTPUTS:
    op - operator to solve
    eps_norm - matrix to multiply E by to get v
    eps_un_norm - matrix to multiply v by to get E
    '''

    # Setting up the forward and backward derivatives
    curl_e = operators.curl_e(dxes, bloch_vec, shift_orthogonal)
    curl_h = operators.curl_h(dxes, bloch_vec, shift_orthogonal)

    # Normalization to make the operator hermitian
    eps_norm = scipy.sparse.diags(1 / np.sqrt(epsilon))
    eps_un_norm = scipy.sparse.diags(np.sqrt(epsilon))

    # Calculating the opertor
    op = eps_norm @ curl_h @ curl_e @ eps_norm
    return op, eps_norm, eps_un_norm


def hfield_operator(bloch_vec, dxes: dx_lists_t, epsilon: vfield_t,
                    mu: vfield_t, shift_orthogonal: np.array):
    '''
    Function to setup the electric field operator
    for eigen value solve.

    The eigen value equation that is being solved here is
      nabla x nabla x (1/sqrt(epsilon))v = omega^2 v

    where E = sqrt(epsilon) v and omega is the eigen-frequency

    INPUTS:
    bloch_vec - bloch vector
    dxes - grid dx
    epsilon - permittivity vector
    mu - permeability vector (assumed to be 1)
    TODO (rahul) - factor in permeability

    OUTPUTS:
    op - operator to solve
    eps_norm - matrix to multiply E by to get v
    eps_un_norm - matrix to multiply v by to get E
    '''

    # Setting up the forward and backward derivatives
    curl_e = operators.curl_e(dxes, bloch_vec, shift_orthogonal)
    curl_h = operators.curl_h(dxes, bloch_vec, shift_orthogonal)

    # Normalization to make the operator hermitian
    eps_inv = scipy.sparse.diags(1 / epsilon)
    eps_norm = scipy.sparse.eye(3 * np.prod(epsilon.shape))

    # Calculating the opertor
    op = curl_e @ eps_inv @ curl_h
    return op, eps_norm, eps_norm


def mode_solver(bloch_vec: np.ndarray,
                omega_appx: float,
                num_modes: int,
                dxes: dx_lists_t,
                epsilon: np.ndarray,
                op_type: str,
                set_init_cond: bool,
                init_vec: np.ndarray = None,
                mu: np.ndarray = None,
                shift_orthogonal=np.zeros((3, 3))):

    if mu is None:
        mu = np.ones_like(epsilon)

    # Setting up the operator
    if op_type == 'efield':
        op, eps_norm, eps_un_norm = efield_operator(
            bloch_vec, dxes, vec(epsilon), vec(mu), shift_orthogonal)

    elif op_type == 'hfield':
        op, eps_norm, eps_norm = hfield_operator(bloch_vec, dxes, vec(epsilon),
                                                 vec(mu), shift_orthogonal)
    else:
        raise ValueError('Undefined operator type')

    if set_init_cond and init_vec is None:

        # Starting with initial condition with a FDFD simulation
        J = np.zeros_like(epsilon)
        J[0][J.shape[1] // 2, J.shape[2] // 2, J.shape[3] // 2] = 1.0
        J[1][J.shape[1] // 2, J.shape[2] // 2, J.shape[3] // 2] = 1.0
        #J[2][J.shape[1]//2, J.shape[2]//2,J.shape[3]//2] = 1.0
        solver = DirectSolver()
        sim_args = {
            'omega': omega_appx,
            'dxes': dxes,
            'epsilon': vec(epsilon),
            'mu': vec(mu),
            'J': vec(J),
            'bloch_vec': bloch_vec
        }

        E = solver.solve(**sim_args)
        elec_field = unvec(E, J[0].shape)

        if op_type == 'efield':
            mode_estimate = eps_un_norm @ E
        elif op_type == 'hfield':
            op_e2h = operators.e2h(
                omega=1.0, dxes=dxes, mu=vec(mu), bloch_vec=bloch_vec)
            mode_estimate = op_e2h @ E
        else:
            raise ValueError('Undefined operator type')

        # Solving for eigen values
        eig_value, mode_field = scipy.sparse.linalg.eigs(
            op, num_modes, sigma=omega_appx**2, v0=mode_estimate)

    elif init_vec is not None:
        eig_value, mode_field = scipy.sparse.linalg.eigs(
            op, 1, sigma=omega_appx**2, v0=init_vec)

    else:
        eig_value, mode_field = scipy.sparse.linalg.eigs(
            op, num_modes, sigma=omega_appx**2)
    omega = np.sqrt(eig_value)

    if op_type == 'efield':
        if len(eig_value) == 1:
            mode_field = eps_norm @ mode_field
        else:
            mode_field = [eps_norm @ mode for mode in mode_field]

    return omega, mode_field.transpose()
