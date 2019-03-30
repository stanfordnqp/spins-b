""" Defines FDFD solvers that rely on local matrix solvers. """
import abc
import logging
import multiprocessing
from typing import List, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from spins import fdfd_tools
from spins.fdfd_tools import operators

logger = logging.getLogger(__name__)


class LocalMatrixSolver:
    """Base class for all CPU solvers that rely on generic matrix solves."""

    @abc.abstractmethod
    def solve_matrix_equation(self, A: scipy.sparse.csr_matrix, b: np.ndarray):
        """Solve matrix equation Ax = b.

        Args:
            A: The matrix A.
            b: The vector b.

        Returns:
            x satisfying Ax = b.
        """
        raise NotImplementedError('solve_matrix_equation not implemented')

    def solve(self,
              omega: complex,
              dxes: fdfd_tools.GridSpacing,
              J: np.ndarray,
              epsilon: np.ndarray,
              pml_layers: Optional[fdfd_tools.PmlLayers] = None,
              mu: np.ndarray = None,
              pec: np.ndarray = None,
              pmc: np.ndarray = None,
              pemc: np.ndarray = None,
              bloch_vec: Optional[np.ndarray] = None,
              symmetry: Optional[np.ndarray] = None,
              adjoint: bool = False,
              E0: np.ndarray = None):
        if (symmetry is not None) and np.any(symmetry):
            raise NotImplementedError("`symmetry` is not implemented.")
        if (pemc is not None) and np.any(pemc):
            raise NotImplementedError("`pemc` is not implemented.")

        if bloch_vec is None:
            bloch_vec = np.zeros(3)

        dxes = fdfd_tools.grid.apply_scpml(dxes, pml_layers, omega)

        b0 = -1j * omega * J
        A0 = operators.e_full(
            omega,
            dxes,
            epsilon=epsilon,
            mu=mu,
            pec=pec,
            pmc=pmc,
            bloch_vec=bloch_vec)

        Pl, Pr = operators.e_full_preconditioners(dxes)

        if adjoint:
            A = (Pl @ A0 @ Pr).H
            b = Pr.H @ b0
        else:
            A = Pl @ A0 @ Pr
            b = Pl @ b0

        x = self.solve_matrix_equation(A.astype(np.complex128).tocsr(), b)

        if adjoint:
            x0 = Pl.H @ x
        else:
            x0 = Pr @ x

        return x0


def _worker_simulate(A, b, solver):
    return solver.solve_matrix_equation(A, b)


class MultiprocessingSolver(LocalMatrixSolver):

    def __init__(self, wrapped_solvee, num_processes=0):
        """ Creates a multiprocessing pool of solvers.

        Args:
            wrapped_solvee: The actual solver to call.
            num_processes: Number of processes to use in pool. If 0, then the
                number of CPUs is used. Default: 0
        """
        if num_processes == 0:
            num_processes = multiprocessing.cpu_count()

        # Handle SIGINT properly by ignoring in the worker processes.
        # See https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        # for details.
        import signal
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.pool = multiprocessing.Pool(num_processes)
        signal.signal(signal.SIGINT, original_sigint_handler)

        self.solver = wrapped_solvee

    def solve_matrix_equation(self, A: scipy.sparse.csr_matrix, b: np.ndarray):
        try:
            return self.pool.apply(_worker_simulate, (A, b, self.solver))
        except KeyboardInterrupt:
            self.pool.terminate()
            self.pool.join()


class DirectSolver(LocalMatrixSolver):
    """ Use a direct sparse matrix solver. """

    def solve_matrix_equation(self, A, b):
        return scipy.sparse.linalg.spsolve(A, b)


class QmrSolver(LocalMatrixSolver):
    """ Use QMR to solve the matrix equation. """

    def __init__(self, relative_tol=1e-6):
        """ Constructs a QMR solver.

        Args:
            relative_tol: Solver tolerance relative to b.
        """
        self.relative_tol = relative_tol

    def solve_matrix_equation(self, A, b):
        tol = self.relative_tol * np.linalg.norm(b)
        x, info = scipy.sparse.linalg.qmr(A, b, tol=tol)
        if info > 0:
            logger.error('Convergence tolerance not achieved.')
        elif info < 0:
            logger.error('Illegal input or breakdown.')
        return x


class BiCgSolver(LocalMatrixSolver):
    """ Use bi-conjugate gradient to solve the matrix equation. """

    def __init__(self, relative_tol=1e-6):
        """ Constructs a BiCG solver.

        Args:
            relative_tol: Solver tolerance relative to b.
        """
        self.relative_tol = relative_tol

    def solve_matrix_equation(self, A, b):
        tol = self.relative_tol * np.linalg.norm(b)
        x, info = scipy.sparse.linalg.bicg(A, b, tol=tol)
        if info > 0:
            logger.error('Convergence tolerance not achieved.')
        elif info < 0:
            logger.error('Illegal input or breakdown.')
        return x
