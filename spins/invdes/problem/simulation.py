import threading
from typing import Callable, List

import numpy as np
import scipy.sparse

import spins.fdfd_tools as fdfd_tools

GridType = List[np.ndarray]  # Should have three elements in list.


class FdfdSimulation:

    def __init__(self,
                 solver: Callable,
                 dims: List[int],
                 omega: complex,
                 dxes: List[GridType],
                 J: GridType,
                 selection_matrix: scipy.sparse.spmatrix,
                 epsilon: GridType,
                 mu: GridType = None,
                 pec: GridType = None,
                 pmc: GridType = None,
                 pemc: np.ndarray = np.array(6 * [0]),
                 symmetry: np.ndarray = np.zeros(3),
                 bloch_vec: np.ndarray = None,
                 cache_size: int = 1) -> None:
        """ Creates a FDFD simulation.

        Args:
            solver: EM solver to use.
            dims: Array of number of elements in the simulation grid along each
                dimension.
            omega: Frequency of FDFD simulation.
            dxes: Grid spacing. The outer list should have two elements
                corresponding to primary and dual grids. The inner list should
                have three arrays corresponding to the three axes.
            J: Current source location.
            selection_matrix:
            epsilon: List of arrays containing the permittivity.
            mu: List of arrays containing the permeability.
            cache_size: Number of simulations to store in cache.
        """
        self.dims = dims
        self.omega = omega
        self.dxes = dxes
        self.selection_matrix = selection_matrix
        self.base_epsilon = epsilon
        self.mu = mu
        self.solver = solver
        self.J = J
        self.pec = pec
        self.pmc = pmc
        self.pemc = pemc
        self.symmetry = symmetry
        if bloch_vec is None:
            self.bloch_vec = np.zeros(3)
        else:
            self.bloch_vec = bloch_vec

        # For caching uses.
        self.cache = [None] * cache_size
        self.lock = threading.Lock()

    def get_dims(self) -> List[int]:
        return self.dims

    def get_epsilon(self, z: np.ndarray) -> np.ndarray:
        # Squeezing is required to turn numpy matrix into ndarray and avoid
        # out of memory error.
        full_z = np.squeeze(self.selection_matrix.dot(z))
        return fdfd_tools.vec(self.base_epsilon) + full_z

    def simulate(self, z: np.ndarray) -> np.ndarray:
        """ Computes the electric field distribution.

        Args:
            z: The structure.
            J: The excitation current. If None, simulation
               uses the current source specified by the constructor.
        Returns:
            Vectorized form of the electric fields.
        """
        # Note that we must lock if there any ongoing solves because
        # 1) we must enforce an ordering in terms of the z's and
        # 2) the requested solve could be the same as the ongoing solve.
        with self.lock:
            # Only solve for the fields if the structure has changed.
            electric_fields = None
            # The cache is implemented as a list where the most recent
            # access is at the back of the list.

            # Search through cache for fields.
            for cache_index in range(len(self.cache)):
                cache_item = self.cache[cache_index]
                if cache_item is None:
                    continue
                cache_z, cache_fields = cache_item
                if np.array_equal(z, cache_z):
                    electric_fields = cache_fields
                    # Remove the hit entry (it will be reinserted later).
                    del self.cache[cache_index]
                    break
            if electric_fields is None:
                electric_fields = self._run_solver(z)
                # Remove the last used element.
                del self.cache[0]
            # Insert data into cache.
            self.cache.append((z, electric_fields))
        return electric_fields

    def _run_solver(self, z: np.ndarray) -> np.ndarray:
        """ Runs the solver to compute electric fields.

        Args:
            z: The structure.
        Returns:
            Vectorized form of the electric fields.
        """
        sim_args = {
            'omega': self.omega,
            'dxes': self.dxes,
            'epsilon': self.get_epsilon(z),
            'mu': fdfd_tools.vec(self.mu),
            'J': fdfd_tools.vec(self.J),
            'pec': fdfd_tools.vec(self.pec),
            'pmc': fdfd_tools.vec(self.pmc),
            'pemc': self.pemc,
            'symmetry': self.symmetry,
            'bloch_vec': self.bloch_vec
        }
        return self.solver.solve(**sim_args)


class AdjointFdfdSimulation:
    """ Represents the adjoint simulation. """

    def __init__(self, sim: FdfdSimulation) -> None:
        """ Initialize the adjoint simulation based on a normal FDFD simulation.

        Args:
            sim: The base FDFD simulation.
        """
        self.sim = sim

        # For caching uses.
        self.cache = [None] * len(sim.cache)
        self.lock = threading.Lock()

    def simulate(self, z: np.ndarray, J: np.ndarray) -> np.ndarray:
        """ Computes the electric field distribution.

        Args:
            z: The structure.
            J: The excitation current.
        Returns:
            Vectorized form of the electric fields.
        """
        # Note that we must lock if there any ongoing solves because
        # 1) we must enforce an ordering in terms of the z's and
        # 2) the requested solve could be the same as the ongoing solve.
        with self.lock:
            # Only solve for the fields if the structure has changed.
            electric_fields = None
            # The cache is implemented as a list where the most recent
            # access is at the back of the list.

            # Search through cache for fields.
            for cache_index in range(len(self.cache)):
                cache_item = self.cache[cache_index]
                if cache_item is None:
                    continue
                cache_z, cache_J, cache_fields = cache_item
                if (np.array_equal(z, cache_z) and np.array_equal(J, cache_J)):
                    electric_fields = cache_fields
                    # Remove the hit entry (it will be reinserted later).
                    del self.cache[cache_index]
                    break
            if electric_fields is None:
                electric_fields = self._run_solver(z, J)
                # Remove the last used element.
                del self.cache[0]
            # Insert data into cache.
            self.cache.append((z, J, electric_fields))
        return electric_fields

    def _run_solver(self, z: np.ndarray, J: np.ndarray) -> np.ndarray:
        """ Runs the solver to compute electric fields.

        Here I use the notation M* = conj(M), M.' = transpose(M),
        M' = ctranspose(M), M N = dot(M, N)

        Maxwell uses a symmetrized wave operator M = (L A R) = (L A R).',
         where L = inv(R) = diag(sqrt(s)) [s as in the code below] when
          solving the wave equation; it thus solves the problem
          M y = d
          => (L A R) (inv(R) x) = (L b)
          => A x = b
          with x = R y

         From the fact that M is symmetric, we can write
          M* = (L A R)* = (L A R)' = R' A' L' = R* A' L*
         We obtain M* by conjugating the contents of sim (except sim.J).
         We then multiply sim.J by (R* R*) = diag(1./s)*, obtaining
          (R* A' L*) v = (L* (R* R* b))
                       = (R* b)
         and then fix the returned value v by multiplying by
          (L* L*) = diag(s)
         leading to the result (L* L* v) = (L* L* (R* x)) = L* x

         Putting it all together, with adjoint=true, we have
          (R* A' L*) (inv(L* L* R*) x) = (L* (R* R* b))
          => (R* A' L*) (inv(L*) x) = (R* b)
          => A' x = b

        Args:
            z: The structure.
        Returns:
            Vectorized form of the electric fields.
        """
        # TODO(logansu): Support PEC/PMC.
        dxes = [[np.conj(dx) for dx in grid] for grid in self.sim.dxes]

        spx, spy, spz = np.meshgrid(
            dxes[1][0], dxes[1][1], dxes[1][2], indexing='ij')
        sdx, sdy, sdz = np.meshgrid(
            dxes[0][0], dxes[0][1], dxes[0][2], indexing='ij')
        mult = np.multiply
        s = [
            mult(mult(sdx, spy), spz),
            mult(mult(spx, sdy), spz),
            mult(mult(spx, spy), sdz)
        ]
        new_J = np.copy(fdfd_tools.unvec(J, self.sim.dims))

        for k in range(3):
            new_J[k] /= np.conj(s[k])
        new_J = fdfd_tools.vec(new_J)
        mu = None
        if self.sim.mu is not None:
            mu = np.conj(fdfd_tools.vec(self.sim.mu))
        sim_args = {
            'omega': np.conj(self.sim.omega),
            'dxes': dxes,
            'epsilon': np.conj(self.sim.get_epsilon(z)),
            'mu': mu,
            'J': new_J,
            'pec': fdfd_tools.vec(self.sim.pec),
            'pmc': fdfd_tools.vec(self.sim.pmc),
            'bloch_vec': self.sim.bloch_vec,
        }
        efields = self.sim.solver.solve(**sim_args)
        # Unvec to undo right pre-conditioner.
        efields = fdfd_tools.unvec(efields, self.sim.dims)
        for i in range(3):
            efields[i] = np.multiply(efields[i], np.conj(s[i]))
        efields = fdfd_tools.vec(efields)
        return efields
