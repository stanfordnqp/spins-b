import logging
logger = logging.getLogger(__name__)

import abc
import enum
import numpy as np
import scipy.sparse
import scipy.io
import copy
from typing import Callable, List
from scipy.sparse.linalg import aslinearoperator

import spins.invdes.parametrization as invparam
import spins.fdfd_tools as fdfd_tools
import spins.fdfd_solvers as fdfd_solvers
from spins.gridlock import Direction

from spins.invdes.problem.objective import OptimizationFunction
from spins.invdes.parametrization import Parametrization
from spins.invdes.problem.simulation import AdjointFdfdSimulation
from spins.invdes.problem.simulation import FdfdSimulation


class EmObjective(OptimizationFunction):
    """ Represents a EM problem to solve. """

    def __init__(self, sim: FdfdSimulation) -> None:
        self.sim = sim
        self.adjoint_sim = AdjointFdfdSimulation(sim)

    def calculate_gradient(self, param: Parametrization) -> np.ndarray:
        struct = param.get_structure()
        efields = self.sim.simulate(struct)

        total_df_dz = self.calculate_df_dz(efields, struct)
        # TODO(logansu): Cache gradient calculation.
        dz_dp = aslinearoperator(param.calculate_gradient())
        df_dp = np.conj(dz_dp.adjoint() @ np.conj(
            total_df_dz + self.calculate_partial_df_dz(efields, struct)))

        return df_dp

    def calculate_df_dz(self, efields, struct):
        # Calculate df/dz using adjoint.
        partial_df_dx = self.calculate_partial_df_dx(efields, struct)

        B = (self.sim.omega**2 * scipy.sparse.diags(efields, 0)
             @ self.sim.selection_matrix)
        d = self.adjoint_sim.simulate(
            struct,
            np.conj(partial_df_dx) / (-1j * self.sim.omega))
        total_df_dz = 2 * np.real(np.conj(np.transpose(d)) @ B)

        return total_df_dz

    def calculate_objective_function(self, parametrization):
        struct = parametrization.get_structure()
        efields = self.sim.simulate(struct)
        return self.calculate_f(efields, struct)

    def calculate_f(self, efields, struct):
        raise NotImplemented('calculate_f not implemented')

    def calculate_partial_df_dx(self, efields, struct):
        return 0

    def calculate_partial_df_dz(self, efields, struct):
        return 0


class OverlapMode:

    class ModeType(enum.Enum):
        wgmode = 0
        point = 1
        arbitrary = 2

    def __init__(self):
        self.type = ModeType.wgmode  # typing: OverlapModeType
        self.axis
        self.polarity
        self.mode_num
        self.pos
        self.J = None  # For arbitrary modes.

    @property
    def mode_type(self):
        return self._mode_type

    @mode_type.setter
    def mode_type(self, mode_type):
        self._mode_type = mode_type


class Overlap(EmObjective):

    def __init__(self, out, sim):
        """ Build an objective based on overlap integral.
            Minimize abs(Cx-T) with complex transmission.

        Args:
            out: Dictionary containing the parameters of the overlap
            sim: FDFD simulation to use.
        Note:
            Unlike the old basicOverlap objective, this objective only
            handles one overlap objective at a time. This means that C is a
            vector with the same size as x and that instead of out_modes, a
            list with several out dictionaries, you just give one out
            dictionary is input.
        """
        super().__init__(sim)
        self._compute_objective(out)

    def _compute_objective(self, out):
        C_shape = (3 * np.prod(self.sim.dims), 1)
        C = scipy.sparse.csr_matrix(C_shape, dtype=np.complex128)
        # Compute field design objective.
        if out['type'] == OverlapMode.ModeType.wgmode:
            sim_params = {
                'omega': self.sim.omega,
                'dxes': self.sim.dxes,
                'axis': out['axis'].value,
                'slices': [slice(i, f + 1) for i, f in zip(*out['pos'])],
                'polarity': out['polarity'],
                'mu': self.sim.mu
            }
            wgmode_result = fdfd_solvers.waveguide_mode.solve_waveguide_mode(
                mode_number=out['mode_num'],
                epsilon=self.sim.base_epsilon,
                **sim_params)
            wgmode_result.update({
                'omega':
                self.sim.omega,
                'dxes':
                self.sim.dxes,
                'axis':
                out['axis'].value,
                'slices': [slice(i, f + 1) for i, f in zip(*out['pos'])],
                'polarity':
                out['polarity'],
            })
            E_out = fdfd_solvers.waveguide_mode.compute_overlap_e(
                **wgmode_result)
        elif out['type'] == OverlapMode.ModeType.arbitrary:
            E_out = out['overlap']
        else:
            raise Exception('Unrecognized mode type: ', out['type'])

        self.objective_T = out.get('transmission', 0)
        self.objective_C = scipy.sparse.csr_matrix(
            np.matrix(fdfd_tools.vec(E_out)).H)

    def calculate_partial_df_dx(self, x, z):
        T = self.objective_T
        C = self.objective_C
        Cx = np.array(C.H.dot(x))
        Cx = np.squeeze(Cx)
        f = np.abs(Cx - T)
        df_viol_dx = 0.5 * np.conj(Cx - T) / f * C.H

        return np.squeeze(np.array(df_viol_dx.toarray()[0]))

    def calculate_f(self, x, z):
        Cx = np.array(self.objective_C.H.dot(x))
        Cx = np.squeeze(Cx)
        T = self.objective_T
        f = np.abs(Cx - T)
        return f

    def get_phase(self, param: Parametrization):
        struct = param.get_structure()
        efields = self.sim.simulate(struct)
        Cx = np.array(self.objective_C.H.dot(efields))
        Cx = np.squeeze(Cx)
        return np.angle(Cx)

    def get_overlap_norm2(self, param: Parametrization):
        """ Get the overlap norm squared given a parametrization. """
        # Get the fields and structure.
        struct = param.get_structure()
        efields = self.sim.simulate(struct)

        # Calculate overlap.
        overlap2 = np.abs(np.array(self.objective_C.H.dot(efields)))**2
        return overlap2

    def get_electric_fields(self, param: Parametrization):
        return fdfd_tools.unvec(
            self.sim.simulate(param.get_structure()), self.sim.get_dims())

    def __str__(self):
        return 'Overlap(lam={lam},T={T})'.format(
            lam=2 * np.pi / self.sim.omega, T=self.objective_T)


# TODO(logansu): delete the objective classes below
class BasicOverlapObjective(EmObjective):

    def __init__(self, out_modes, sim, power=2):
        """ Build an objective based on overlap integral.
        Args:
            out_modes: List of dictionaries, one for each output mode. Each
                mode dictionary contains the following keys:
            sim: FDFD simulation to use.
            power: Exponent used in the smooth approximation of indicator
                   function in objective.
        """
        logger.warning('BasicOverlapObjective is deprecated')
        super().__init__(sim)

        self._compute_objective(out_modes)
        # Power for objective function.
        self.pf = power

    def _compute_objective(self, out_modes):
        num_modes = len(out_modes)
        C_shape = (3 * np.prod(self.sim.dims), num_modes)
        C = scipy.sparse.csr_matrix(C_shape, dtype=np.complex128)
        alpha = np.zeros((num_modes,))
        beta = np.zeros((num_modes,))
        # Compute field design objective.
        for j, out in enumerate(out_modes):
            # TODO(logansu): Finish overlap mode stuff.
            if out['type'] == OverlapMode.ModeType.wgmode:
                sim_params = {
                    'omega': self.sim.omega,
                    'dxes': self.sim.dxes,
                    'axis': out['axis'].value,
                    'slices': [slice(i, f + 1) for i, f in zip(*out['pos'])],
                    'polarity': out['polarity'],
                    'mu': self.sim.mu
                }
                wgmode_result = fdfd_solvers.waveguide_mode.solve_waveguide_mode(
                    mode_number=out['mode_num'],
                    epsilon=self.sim.base_epsilon,
                    **sim_params)
                wgmode_result.update({
                    'omega':
                    self.sim.omega,
                    'dxes':
                    self.sim.dxes,
                    'axis':
                    out['axis'].value,
                    'slices': [slice(i, f + 1) for i, f in zip(*out['pos'])],
                    'polarity':
                    out['polarity'],
                })
                E_out = fdfd_solvers.waveguide_mode.compute_overlap_e(
                    **wgmode_result)
                E_out = fdfd_tools.vec(E_out)
            elif out['type'] == OverlapMode.ModeType.arbitrary:
                E_out = fdfd_tools.vec(out['overlap'])
            else:
                raise Exception('Unrecognized mode type: ', out.mode_type)
            # Linear algebra-ize.
            C[:, j] = scipy.sparse.csr_matrix(np.matrix(E_out).H)
            alpha[j] = np.sqrt(np.min(out['power']))
            beta[j] = np.sqrt(np.max(out['power']))
        self.objective_alpha = alpha
        self.objective_beta = beta
        self.objective_C = C

    def calculate_partial_df_dx(self, x, z):
        pf = self.pf
        alpha = np.squeeze(self.objective_alpha)
        beta = np.squeeze(self.objective_beta)
        C = self.objective_C

        Cx = np.array(C.H.dot(x))
        overlap = np.abs(Cx)

        f0 = ((overlap < alpha) * (overlap - alpha) +
              (beta < overlap) * (beta - overlap))

        df0_viol_dx = scipy.sparse.diags(
            (1 * (overlap < alpha) - 1 *
             (beta < overlap)) * 0.5 * np.conj(Cx) / overlap) @ C.H

        df_viol_dx = (
            pf * scipy.sparse.diags(f0**(pf - 1)) * df0_viol_dx).sum(axis=0)
        return np.squeeze(np.array(df_viol_dx))

    def calculate_f(self, x, z):
        overlap = np.array(np.absolute(self.objective_C.H.dot(x)))
        overlap = np.squeeze(overlap)
        alpha = self.objective_alpha
        beta = self.objective_beta
        f0 = ((overlap < alpha) * (alpha - overlap) +
              (beta < overlap) * (overlap - beta))
        f = np.sum(f0**self.pf)
        return f

    def get_phase(self, param: Parametrization):
        struct = param.get_structure()
        efields = self.sim.simulate(struct)

        Cx = np.array(self.objective_C.H.dot(efields))
        Cx = np.squeeze(Cx)

        return np.angle(Cx)

    def get_overlap_norm2(self, param: Parametrization):
        """ Get the overlap norm squared given a parametrization. """
        # Get the fields and structure.
        struct = param.get_structure()
        efields = self.sim.simulate(struct)

        # Calculate overlap.
        overlap2 = np.abs(np.array(self.objective_C.H.dot(efields)))**2
        return overlap2

    def get_electric_fields(self, param: Parametrization):
        return fdfd_tools.unvec(
            self.sim.simulate(param.get_structure()), self.sim.get_dims())

    def __str__(self):
        return 'BasicOverlap(lam={lam},p={p})'.format(
            lam=2 * np.pi / self.sim.omega, p=self.pf)


class PhaseOverlapObjective(EmObjective):

    def __init__(self, out_modes, sim):
        """ Build an objective based on overlap integral.

            minimize abs(Cx-T)^2 with the complex transmission

        Args:
            out_modes: List of dictionaries, one for each output mode. Each
                mode dictionary contains the following keys:
                out['transmission']
            sim: FDFD simulation to use.
        """
        logger.warning('PhaseOverlapObjective is deprecated')
        super().__init__(sim)

        self.pf = 2
        self._compute_objective(out_modes)

    def _compute_objective(self, out_modes):
        num_modes = len(out_modes)
        C_shape = (3 * np.prod(self.sim.dims), num_modes)
        C = scipy.sparse.csr_matrix(C_shape, dtype=np.complex128)
        T_complex = np.zeros((num_modes,), dtype=complex)
        # Compute field design objective.
        for j, out in enumerate(out_modes):
            # TODO(logansu): Finish overlap mode stuff.
            if out['type'] == OverlapMode.ModeType.wgmode:
                args = {
                    'mode_num':
                    out['mode_num'],
                    'omega':
                    self.sim.omega,
                    'dxes':
                    self.sim.dxes,
                    'axis':
                    out['axis'].value,
                    'waveguide_slice':
                    [slice(i, f + 1) for i, f in zip(*out['pos'])],
                    'polarity':
                    out['polarity'],
                    'power':
                    1,
                    'eps':
                    self.sim.base_epsilon,
                    'mu':
                    self.sim.mu
                }
                E_out = fdfd_solvers.waveguide_mode.build_overlap(**args)
                E_out = fdfd_tools.vec(E_out)
            elif out['type'] == OverlapMode.ModeType.arbitrary:
                E_out = np.conj(fdfd_tools.vec(
                    out['overlap']))  # conj to compensate for .H later
            else:
                raise Exception('Unrecognized mode type: ', out.mode_type)
            # Linear algebra-ize.
            C[:, j] = scipy.sparse.csr_matrix(np.matrix(E_out).H)
            # get T
            T_complex[j] = out['transmission']
        self.objective_T_complex = T_complex
        self.objective_C = C

    def calculate_partial_df_dx(self, x, z):
        pf = self.pf
        T_complex = self.objective_T_complex
        C = self.objective_C

        Cx = np.array(C.H.dot(x))
        Cx = np.squeeze(Cx)

        f0 = np.abs(Cx - T_complex)

        df0_viol_dx = scipy.sparse.diags(
            0.5 * np.conj(Cx - T_complex) / f0) @ C.H

        df_viol_dx = (
            pf * scipy.sparse.diags(f0**(pf - 1)) * df0_viol_dx).sum(axis=0)

        return np.squeeze(np.array(df_viol_dx))

    def calculate_f(self, x, z):
        Cx = np.array(self.objective_C.H.dot(x))
        Cx = np.squeeze(Cx)
        T_complex = self.objective_T_complex
        f0 = np.abs(Cx - T_complex)
        f = np.sum(f0**self.pf)
        return f

    def get_phase(self, param: Parametrization):
        struct = param.get_structure()
        efields = self.sim.simulate(struct)

        Cx = np.array(self.objective_C.H.dot(efields))
        Cx = np.squeeze(Cx)

        return np.angle(Cx)

    def get_overlap_norm2(self, param: Parametrization):
        """ Get the overlap norm squared given a parametrization. """
        # Get the fields and structure.
        struct = param.get_structure()
        efields = self.sim.simulate(struct)

        # Calculate overlap.
        overlap2 = np.abs(np.array(self.objective_C.H.dot(efields)))**2
        return overlap2

    def get_electric_fields(self, param: Parametrization):
        return fdfd_tools.unvec(
            self.sim.simulate(param.get_structure()), self.sim.get_dims())

    def __str__(self):
        return 'PhaseOverlap(lam={lam},p={p})'.format(
            lam=2 * np.pi / self.sim.omega, p=self.pf)
