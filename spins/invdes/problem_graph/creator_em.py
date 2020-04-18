from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import os
import scipy.sparse
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

from spins import fdfd_solvers
from spins import fdfd_tools
from spins import gridlock
from spins.invdes import problem
from spins.fdfd_solvers import local_matrix_solvers
from spins.invdes.problem_graph import grid_utils
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
# Make a style guide exception here because `simspace` is already used as a
# variable.
from spins.invdes.problem_graph.simspace import SimulationSpace

# Have a single shared direct solver object because we need to use
# multiprocessing to actually parallelize the solve.
DIRECT_SOLVER = local_matrix_solvers.MultiprocessingSolver(
    local_matrix_solvers.DirectSolver())


@optplan.register_node(optplan.WaveguideModeSource)
class WaveguideModeSource:

    def __init__(self,
                 params: optplan.WaveguideModeSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        """Creates a new waveguide mode source.

        Args:
            params: Waveguide source parameters.
            work: Workspace for this source. Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float,
                 **kwargs) -> fdfd_tools.VecField:
        """Creates the source vector.

        Args:
            simspace: Simulation space object to use.
            wlen: Wavelength to operate source.

        Returns:
            The vector field corresponding to the source.
        """
        space_inst = simspace(wlen)
        return fdfd_solvers.waveguide_mode.build_waveguide_source(
            omega=2 * np.pi / wlen,
            dxes=simspace.dxes,
            eps=space_inst.eps_bg.grids,
            mu=None,
            mode_num=self._params.mode_num,
            waveguide_slice=grid_utils.create_region_slices(
                simspace.edge_coords, self._params.center,
                self._params.extents),
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            power=self._params.power)


@optplan.register_node(optplan.PlaneWaveSource)
class PlaneWaveSource:

    def __init__(self,
                 params: optplan.PlaneWaveSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        """Creates a plane wave source.

        Args:
            params: Parameters for the plane wave source.
            work: Unused.
        """
        self._params = params

    def __call__(
            self, simspace: SimulationSpace, wlen: float, solver, **kwargs
    ) -> Union[fdfd_tools.VecField, Tuple[fdfd_tools.VecField, fdfd_tools.
                                          Vec3d]]:
        """Creates the plane wave source.

        Args:
            simspace: Simulation space to use for the source.
            wlen: Wavelength of source.

        Returns:
            If `overwrite_bloch_vector` is `True`, a tuple containing the source
            field and the Bloch vector corresponding to the plane wave source.
            Otherwise, only the source field is returned.
        """
        space_inst = simspace(wlen)

        # Calculate the border in gridpoints and igore the border if it's larger then the simulation.
        dx = simspace.dx
        if self._params.border:
            border = [int(b // dx) for b in self._params.border]
        else:
            border = [0, 0]
        # The plane wave is assumed to be in the z direction so the border is 0 for z.
        border.append(0)

        source, kvector = fdfd_tools.free_space_sources.build_plane_wave_source(
            omega=2 * np.pi / wlen,
            eps_grid=space_inst.eps_bg,
            mu=None,
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            slices=grid_utils.create_region_slices(simspace.edge_coords,
                                                   self._params.center,
                                                   self._params.extents),
            theta=self._params.theta,
            psi=self._params.psi,
            polarization_angle=self._params.polarization_angle,
            border=border,
            power=self._params.power)

        # TODO(logansu): Figure out what's wrong with mixing PMLs and
        # Bloch vector. It seems to create problems.
        # For now, we manually set Bloch to zero is PML is nonzero.
        if simspace.pml_layers[0] != 0 or simspace.pml_layers[1] != 0:
            kvector[0] = 0
        if simspace.pml_layers[2] != 0 or simspace.pml_layers[3] != 0:
            kvector[1] = 0
        if simspace.pml_layers[4] != 0 or simspace.pml_layers[5] != 0:
            kvector[2] = 0

        if self._params.normalize_by_sim:
            source = fdfd_tools.free_space_sources.normalize_source_by_sim(
                omega=2 * np.pi / wlen,
                source=source,
                eps=space_inst.eps_bg.grids,
                dxes=simspace.dxes,
                pml_layers=simspace.pml_layers,
                solver=solver,
                power=self._params.power,
                bloch_vector=kvector,
            )

        if self._params.overwrite_bloch_vector:
            return source, kvector
        return source


@optplan.register_node(optplan.GaussianSource)
class GaussianSource:

    def __init__(self,
                 params: optplan.GaussianSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        """Creates a Gaussian beam source.

        Args:
            params: Gaussian beam source parameters.
            work: Unused.
        """
        self._params = params
        if self._params.beam_center is None:
            self._params.beam_center = self._params.center

    def __call__(self, simspace: SimulationSpace, wlen: float, solver: Callable,
                 **kwargs) -> fdfd_tools.VecField:
        """Creates the source vector.

        Args:
            simspace: Simulation space.
            wlen: Wavelength of source.
            solver: If `normalize_by_source` is `True`, `solver` will be used
                to run an EM simulation to normalize the source power.

        Returns:
            The source.
        """
        space_inst = simspace(wlen)
        source, _ = fdfd_tools.free_space_sources.build_gaussian_source(
            omega=2 * np.pi / wlen,
            eps_grid=space_inst.eps_bg,
            mu=None,
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            slices=grid_utils.create_region_slices(simspace.edge_coords,
                                                   self._params.center,
                                                   self._params.extents),
            theta=self._params.theta,
            psi=self._params.psi,
            polarization_angle=self._params.polarization_angle,
            w0=self._params.w0,
            center=self._params.beam_center,
            power=self._params.power)

        if self._params.normalize_by_sim:
            source = fdfd_tools.free_space_sources.normalize_source_by_sim(
                omega=2 * np.pi / wlen,
                source=source,
                eps=space_inst.eps_bg.grids,
                dxes=simspace.dxes,
                pml_layers=simspace.pml_layers,
                solver=solver,
                power=self._params.power)

        return source


@optplan.register_node(optplan.DipoleSource)
class DipoleSource:

    def __init__(self,
                 params: optplan.DipoleSource,
                 work: Optional[workspace.Workspace] = None) -> None:
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float, solver: Callable,
                 **kwargs) -> fdfd_tools.VecField:
        """Creates the source vector.

        Args:
            simspace: Simulation space.
            wlen: Wavelength of source.
            solver: If `normalize_by_source` is `True`, `solver` will be used
                to run an EM simulation to normalize the source power.

        Returns:
            The source.
        """
        phase = self._params.phase
        if phase is None:
            phase = 0

        space_inst = simspace(wlen)
        source = fdfd_solvers.dipole.build_dipole_source(
            omega=2 * np.pi / wlen,
            dxes=simspace.dxes,
            eps=space_inst.eps_bg.grids,
            position=space_inst.eps_bg.pos2ind(self._params.position,
                                               which_shifts=None).astype(int),
            axis=self._params.axis,
            power=self._params.power,
            phase=np.exp(1j * phase))

        if self._params.normalize_by_sim:
            source = fdfd_tools.free_space_sources.normalize_source_by_sim(
                omega=2 * np.pi / wlen,
                source=source,
                eps=space_inst.eps_bg.grids,
                dxes=simspace.dxes,
                pml_layers=simspace.pml_layers,
                solver=solver,
                power=self._params.power)

        return source


class FdfdSimulation(problem.OptimizationFunction):
    """Represents a FDFD simulation.

    Simulations are cached so that repeated calls with the same permittivity
    distribution does not incur multiple simulations. However, this cache is not
    thread-safe.
    """

    def __init__(
            self,
            eps: problem.OptimizationFunction,
            solver: Callable,
            wlen: float,
            source: np.ndarray,
            simspace: SimulationSpace,
            bloch_vector: Optional[fdfd_tools.Vec3d] = None,
            cache_size: int = 1,
    ) -> None:
        """Creates a FDFD simulation.

        Args:
            eps: Permittivity distribution to simulate.
            solver: Electromagnetic solver to use.
            wlen: Wavelength of simulation.
            source: Vector corresponding to the source of the simulation.
            simspace: Simulation space.
            bloch_vector: Bloch vector to use.
            cache_size: Size of cache used to store adjoint and forward fields.
                This should normally be `1`.
        """
        super().__init__(eps, heavy_compute=True)

        self._solver = solver
        self._wlen = wlen
        self._source = source
        self._simspace = simspace
        self._bloch_vector = bloch_vector

        # For caching uses.
        self._cache = [None] * cache_size
        self._cache_adjoint = [None] * cache_size

    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        """Runs the simulation.

        Args:
            input_vals: List with single element corresponding to the
                permittivity distribution.

        Returns:
            Simulated fields.
        """
        return self._simulate(input_val[0])

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Computes gradient via a adjoint calculation.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            Gradient.
        """
        omega = 2 * np.pi / self._wlen
        efields = self._simulate(input_vals[0])
        B = omega**2 * scipy.sparse.diags(efields, 0)
        d = self._simulate_adjoint(input_vals[0],
                                   np.conj(grad_val) / (-1j * omega))
        total_df_dz = np.conj(np.transpose(d)) @ B
        # If this is a function that maps from real to complex, we have to
        # to take the real part to make gradient real.
        if np.isrealobj(input_vals[0]):
            total_df_dz = np.real(total_df_dz)

        return [total_df_dz]

    def _simulate(self, eps: np.ndarray) -> np.ndarray:
        """Computes the electric field distribution.

        Because simulations are very expensive, we cache the simulations.

        Args:
            eps: The structure.

        Returns:
            Vectorized form of the electric fields.
        """
        # Only solve for the fields if the structure has changed.
        electric_fields = None
        # The cache is implemented as a list where the most recent
        # access is at the back of the list.

        # Search through cache for fields.
        for cache_index in range(len(self._cache)):
            cache_item = self._cache[cache_index]
            if cache_item is None:
                continue
            cache_struc, cache_fields = cache_item
            if np.array_equal(eps, cache_struc):
                electric_fields = cache_fields
                # Remove the hit entry (it will be reinserted later).
                del self._cache[cache_index]
                break

        if electric_fields is None:
            # Perfrom the solve.
            electric_fields = self._solver.solve(
                omega=2 * np.pi / self._wlen,
                dxes=self._simspace.dxes,
                epsilon=eps,
                mu=None,
                J=fdfd_tools.vec(self._source),
                pml_layers=self._simspace.pml_layers,
                bloch_vec=self._bloch_vector,
            )
            # Remove the last used element.
            del self._cache[0]

        # Insert data into cache.
        self._cache.append((eps, electric_fields))
        return electric_fields

    def _simulate_adjoint(self, eps: np.ndarray,
                          source: np.ndarray) -> np.ndarray:
        """Computes an adjoint simulation.

        Args:
            eps: The structure.
            source: The excitation current.

        Returns:
            Vectorized form of the electric fields.
        """
        # Only solve for the fields if the structure has changed.
        electric_fields = None
        # The cache is implemented as a list where the most recent
        # access is at the back of the list.

        # Search through cache for fields.
        for cache_index in range(len(self._cache_adjoint)):
            cache_item = self._cache_adjoint[cache_index]
            if cache_item is None:
                continue
            cache_struc, cache_source, cache_fields = cache_item
            if (np.array_equal(eps, cache_struc) and
                    np.array_equal(source, cache_source)):
                electric_fields = cache_fields
                # Remove the hit entry (it will be reinserted later).
                del self._cache_adjoint[cache_index]
                break

        if electric_fields is None:
            electric_fields = self._solver.solve(
                omega=2 * np.pi / self._wlen,
                dxes=self._simspace.dxes,
                epsilon=eps,
                mu=None,
                J=source,
                pml_layers=self._simspace.pml_layers,
                bloch_vec=self._bloch_vector,
                adjoint=True,
            )

            # Remove the last used element.
            del self._cache_adjoint[0]

        # Insert data into cache.
        self._cache_adjoint.append((eps, source, electric_fields))

        return electric_fields

    def __str__(self):
        return "Simulation({})".format(self._wlen)


def _create_solver(solver_name: str, simspace: SimulationSpace) -> Callable:
    """Instantiates a Maxwell solver.

    Args:
        solver_name: Name of the solver.
        simspace: Simulation space.

    Returns:
         A callable solver object.
    """
    if solver_name == "maxwell_cg":
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        solver = MaxwellSolver(simspace.dims, solver="CG")
    elif solver_name == "maxwell_bicgstab":
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        solver = MaxwellSolver(simspace.dims, solver="biCGSTAB")
    elif solver_name == "local_direct":
        solver = DIRECT_SOLVER
    else:
        raise ValueError("Unknown solver, got {}".format(solver_name))

    return solver


@optplan.register_node(optplan.FdfdSimulation)
def create_fdfd_simulation(params: optplan.FdfdSimulation,
                           work: workspace.Workspace) -> FdfdSimulation:
    """Creates a `FdfdSimulation` object."""
    simspace = work.get_object(params.simulation_space)
    solver = _create_solver(params.solver, simspace)
    bloch_vector = params.get("bloch_vector", np.zeros(3))

    source = work.get_object(params.source)(simspace,
                                            params.wavelength,
                                            solver=solver)
    if isinstance(source, tuple):
        source, bloch_vector = source

    return FdfdSimulation(eps=work.get_object(params.epsilon),
                          solver=solver,
                          wlen=params.wavelength,
                          source=source,
                          simspace=simspace,
                          bloch_vector=bloch_vector,
                          cache_size=1)


class Epsilon(problem.OptimizationFunction):
    """Represents the permittivity distribution.

    This is a particular instantiation of the permittivity distribution
    described by `SimulationSapce`.
    """

    def __init__(
            self,
            input_function: problem.OptimizationFunction,
            wlen: float,
            simspace: SimulationSpace,
    ) -> None:
        """Creates a FDFD simulation Optimization function.

        Args:
            input_function: Input function corresponding to the structure.
                This should be a vector compatible with the selection matrix.
            wlen: Wavelength to evaluate permittivity.
            simspace: Simulation space from which to get permittivity.
        """
        super().__init__(input_function)

        self._wlen = wlen
        self._space = simspace(wlen)

    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        """Returns simulated fields.

        Args:
            input_vals: List of the input values.

        Returns:
            Simulated fields.
        """
        return (fdfd_tools.vec(self._space.eps_bg.grids) +
                self._space.selection_matrix @ input_val[0])

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns gradient of the epsilon calculation.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            Gradient.
        """
        # In the backprop calculation, this should really be `2 * np.real(...)`.
        # However, because in our backprop calculations, we know that eventually
        # the complex value will hit `AbsoluteValue`, which is implemented
        # assuming that the input value is real (i.e.
        # `d abs(x)/dx = conj(x)/|x|` as opposed to `d abs(x)/dx = conj(x)/2|x|`
        # so the factor of 2 is cancelled out.
        return [
            np.real(
                np.squeeze(np.asarray(grad_val @ self._space.selection_matrix)))
        ]

    def __str__(self):
        return "Epsilon({})".format(self._wlen)


@optplan.register_node(optplan.Epsilon)
def create_epsilon(params: optplan.Epsilon,
                   work: workspace.Workspace) -> Epsilon:
    return Epsilon(input_function=work.get_object(workspace.VARIABLE_NODE),
                   wlen=params.wavelength,
                   simspace=work.get_object(params.simulation_space))


@optplan.register_node(optplan.WaveguideModeOverlap)
class WaveguideModeOverlap:

    def __init__(self,
                 params: optplan.WaveguideModeOverlap,
                 work: workspace.Workspace = None) -> None:
        """Creates a new waveguide mode overlap.

        Args:
            params: Waveguide mode parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float,
                 **kwargs) -> fdfd_tools.VecField:
        space_inst = simspace(wlen)
        return fdfd_solvers.waveguide_mode.build_overlap(
            omega=2 * np.pi / wlen,
            dxes=simspace.dxes,
            eps=space_inst.eps_bg.grids,
            mu=None,
            mode_num=self._params.mode_num,
            waveguide_slice=grid_utils.create_region_slices(
                simspace.edge_coords, self._params.center,
                self._params.extents),
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            power=self._params.power)


@optplan.register_node(optplan.ImportOverlap)
class ImportOverlap:

    def __init__(self,
                 params: optplan.ImportOverlap,
                 work: workspace.Workspace = None) -> None:
        """Creates a new waveguide mode overlap.

        Args:
            params: Waveguide mode parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float = None,
                 **kwargs) -> fdfd_tools.VecField:
        matpath = os.path.join(simspace._filepath, self._params.file_name)
        overlap = sio.loadmat(matpath)

        # Use reference_grid to get coords which the overlap fields are defined on.
        reference_grid = simspace(wlen).eps_bg
        overlap_grid = np.zeros(reference_grid.grids.shape, dtype=np.complex_)

        xyz = reference_grid.xyz
        dxyz = reference_grid.dxyz
        shifts = reference_grid.shifts

        overlap_comp = ["Ex", "Ey", "Ez"]
        overlap_center = self._params.center

        overlap_coords = [
            overlap["x"][0] + overlap_center[0],
            overlap["y"][0] + overlap_center[1],
            overlap["z"][0] + overlap_center[2]
        ]

        # The interpolation done below only works on three-dimensional grids with each dimension containing
        # more than a single grid point (i.e. no two-dimensional grids). Therefore, if a dimension has a
        # singleton grid point, we duplicate along that axis to create a pseudo-3D grid.
        coord_dims = np.array([
            overlap_coords[0].size, overlap_coords[1].size,
            overlap_coords[2].size
        ])
        singleton_dims = np.where(coord_dims == 1)[0]
        if not singleton_dims.size == 0:
            for axis in singleton_dims:
                # The dx from the SPINS simulation grid is borrowed for the replication.
                dx = dxyz[axis][0]
                coord = overlap_coords[axis][0]
                overlap_coords[axis] = np.insert(overlap_coords[axis], 0,
                                                 coord - dx / 2)
                overlap_coords[axis] = np.append(overlap_coords[axis],
                                                 coord + dx / 2)
                # Repeat the overlap fields along the extended axis
                for comp in overlap_comp:
                    overlap[comp] = np.repeat(overlap[comp],
                                              overlap_coords[axis].size, axis)

        for i in range(0, 3):

            # Interpolate the user-specified overlap fields for use on the simulation grids
            overlap_interp_function = RegularGridInterpolator(
                (overlap_coords[0], overlap_coords[1], overlap_coords[2]),
                overlap[overlap_comp[i]],
                bounds_error=False,
                fill_value=0.0)

            # Grid coordinates for each component of Electric field. Shifts due to Yee lattice offsets.
            # See documentation of ``Grid" class for more detailed explanation.
            xs = xyz[0] + dxyz[0] * shifts[i, 0]
            ys = xyz[1] + dxyz[1] * shifts[i, 1]
            zs = xyz[2] + dxyz[2] * shifts[i, 2]

            # Evaluate the interpolated overlap fields on simulationg rids
            eval_coord_grid = np.meshgrid(xs, ys, zs, indexing='ij')
            eval_coord_points = np.reshape(eval_coord_grid, (3, -1),
                                           order='C').T
            interp_overlap = overlap_interp_function(eval_coord_points)
            overlap_grid[i] = np.reshape(interp_overlap,
                                         (len(xs), len(ys), len(zs)),
                                         order='C')

        return overlap_grid


# TODO(logansu): This function appears just to be an inner product.
# Why is this a separate function right now?
class OverlapFunction(problem.OptimizationFunction):
    """Represents an optimization function for overlap."""

    def __init__(self, input_function: problem.OptimizationFunction,
                 overlap: np.ndarray):
        """Constructs the objective C*x.

        Args:
            input_function: Input objectives(typically a simulation).
            overlap: Vector to overlap with.
        """
        super().__init__(input_function)

        self._input = input_function
        self.overlap_vector = overlap

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input values.

        Returns:
            Vector product of overlap and the input.
        """
        return self.overlap_vector @ input_vals[0]

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            gradient.
        """
        return [grad_val * self.overlap_vector]

    def __str__(self):
        return "Overlap({})".format(self._input)


@optplan.register_node(optplan.Overlap)
def create_overlap_function(params: optplan.ProblemGraphNode,
                            work: workspace.Workspace):
    simspace = work.get_object(params.simulation.simulation_space)
    wlen = params.simulation.wavelength
    overlap = fdfd_tools.vec(work.get_object(params.overlap)(simspace, wlen))
    return OverlapFunction(input_function=work.get_object(params.simulation),
                           overlap=overlap)


# TODO(logansu): Merge this into `AbsoluteValue`.
class DiffEpsilon(problem.OptimizationFunction):
    """Computes a L2 norm between two permittivity distributions.

    Specifically, this function computes `np.sum(np.abs(eps - eps_ref)**2)`.
    """

    def __init__(self, epsilon: problem.OptimizationFunction,
                 epsilon_ref: Callable[[], np.ndarray]) -> None:
        """Creates new `DiffEpsilon` function.

        Here we accept a callable because we may want to evaluate the target
        permittivity distribution dynamically (e.g. it may depend on the
        current value of a parametrization).

        Args:
            epsilon: Permittivity distribution that will be differentiated.
            epsilon_ref: Callable that returns a permittivity to which to
                compare `epsilon`.
        """
        super().__init__(epsilon)

        self._get_eps_ref = epsilon_ref

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args:
            input_vals: List of the input values.

        Returns:
            Integrated sum of the difference between `epsilon` and
            `epsilon_ref`.
        """
        return np.sum(np.abs(input_vals[0] - self._get_eps_ref())**2)

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.

        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.

        Returns:
            Gradient.
        """
        diff = np.conj(input_vals[0] - self._get_eps_ref())
        grad = 2 * diff
        return [grad_val * grad]


@optplan.register_node(optplan.DiffEpsilon)
def create_diff_epsilon(params: optplan.DiffEpsilon,
                        work: workspace.Workspace) -> DiffEpsilon:

    if params.epsilon_ref.type == "gds":
        space = work.get_object(params.epsilon.simulation_space)
        from spins.invdes.problem_graph.simspace import _create_grid
        eps = _create_grid(params.epsilon_ref, space._edge_coords,
                           params.epsilon.wavelength, space._ext_dir,
                           space._filepath)
        eps_vec = fdfd_tools.vec(eps.grids)

    def epsilon_ref() -> np.ndarray:
        if params.epsilon_ref.type == "parametrization":
            space = work.get_object(params.epsilon_ref.simulation_space)(
                params.epsilon_ref.wavelength)
            structure = work.get_object(
                params.epsilon_ref.parametrization).get_structure()
            return (fdfd_tools.vec(space.eps_bg.grids) +
                    space.selection_matrix @ structure)
        elif params.epsilon_ref.type == "gds":
            return eps_vec
        else:
            raise NotImplementedError(
                "Epsilon spec with type {} not yet supported".format(
                    params.epsilon_ref.type))

    return DiffEpsilon(epsilon=work.get_object(params.epsilon),
                       epsilon_ref=epsilon_ref)
