from typing import Callable, List, Optional, Tuple

import copy
import dataclasses
import os

import numpy as np

from spins import goos
from spins.goos_sim import maxwell
from spins.goos_sim.maxwell import render
from spins.goos_sim.maxwell import simspace
from spins import fdfd_solvers
from spins import fdfd_tools
from spins import gridlock
from spins.fdfd_solvers import local_matrix_solvers

# Have a single shared direct solver object because we need to use
# multiprocessing to actually parallelize the solve.
DIRECT_SOLVER = local_matrix_solvers.MultiprocessingSolver(
    local_matrix_solvers.DirectSolver())


@dataclasses.dataclass
class FdfdSimProp:
    """Represents properties of a FDFD simulation."""
    eps: np.ndarray
    source: np.ndarray
    wlen: float
    dxes: List[np.ndarray]
    pml_layers: List[int]
    bloch_vec: np.ndarray = None
    fields: np.ndarray = None
    grid: gridlock.Grid = None
    solver: Callable = None
    symmetry: np.ndarray = goos.np_zero_field(3)


class SimSource(goos.Model):
    pass


class SimSourceImpl:

    def before_sim(self, sim: FdfdSimProp) -> None:
        pass


@goos.polymorphic_model()
class GaussianSource(SimSource):
    """Represents a gaussian source.

    Attributes:
        type: Must be "source.gaussian_beam".
        normalize_by_sim: If `True`, normalize the power by running a
            simulation.
    """
    type = goos.ModelNameType("source.gaussian_beam")
    w0 = goos.types.FloatType()
    center = goos.Vec3d()
    beam_center = goos.Vec3d()
    extents = goos.Vec3d()
    normal = goos.Vec3d()
    theta = goos.types.FloatType()
    psi = goos.types.FloatType()
    polarization_angle = goos.types.FloatType()
    power = goos.types.FloatType()
    normalize_by_sim = goos.types.BooleanType(default=False)


@maxwell.register(GaussianSource)
class GaussianSourceImpl(SimSourceImpl):

    def __init__(self, params: GaussianSource) -> None:
        """Creates a Gaussian beam source.

        Args:
            params: Gaussian beam source parameters.
        """
        self._params = params

    def before_sim(self, sim: FdfdSimProp) -> None:
        beam_center = self._params.beam_center
        if beam_center is None:
            beam_center = self._params.center

        eps_grid = copy.deepcopy(sim.grid)
        eps_grid.grids = sim.eps

        source, _ = fdfd_tools.free_space_sources.build_gaussian_source(
            omega=2 * np.pi / sim.wlen,
            eps_grid=eps_grid,
            mu=None,
            axis=gridlock.axisvec2axis(self._params.normal),
            polarity=gridlock.axisvec2polarity(self._params.normal),
            slices=simspace.create_region_slices(sim.grid.exyz,
                                                 self._params.center,
                                                 self._params.extents),
            theta=self._params.theta,
            psi=self._params.psi,
            polarization_angle=self._params.polarization_angle,
            w0=self._params.w0,
            center=beam_center,
            power=self._params.power)

        if self._params.normalize_by_sim:
            source = fdfd_tools.free_space_sources.normalize_source_by_sim(
                omega=2 * np.pi / sim.wlen,
                source=source,
                eps=sim.eps,
                dxes=sim.dxes,
                pml_layers=sim.pml_layers,
                solver=sim.solver,
                power=self._params.power)

        sim.source += source


@goos.polymorphic_model()
class WaveguideModeSource(SimSource):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = goos.ModelNameType("source.waveguide_mode")
    center = goos.Vec3d()
    extents = goos.Vec3d()
    normal = goos.Vec3d()
    mode_num = goos.types.IntType()
    power = goos.types.FloatType()


@maxwell.register(WaveguideModeSource)
class WaveguideModeSourceImpl(SimSourceImpl):

    def __init__(self, source: WaveguideModeSource) -> None:
        self._src = source
        self._wg_mode = None

    def before_sim(self, sim: FdfdSimProp) -> None:
        if self._wg_mode is None:
            self._wg_mode = fdfd_solvers.waveguide_mode.build_waveguide_source(
                omega=2 * np.pi / sim.wlen,
                dxes=sim.dxes,
                eps=sim.eps,
                mu=None,
                mode_num=self._src.mode_num,
                waveguide_slice=simspace.create_region_slices(
                    sim.grid.exyz, self._src.center, self._src.extents),
                axis=gridlock.axisvec2axis(self._src.normal),
                polarity=gridlock.axisvec2polarity(self._src.normal),
                power=self._src.power)

        sim.source += self._wg_mode


@goos.polymorphic_model()
class DipoleSource(SimSource):
    """Represents a dipole source.

    Attributes:
        position: Position of the dipole (will snap to grid).
        axis: Direction of the dipole (x:0, y:1, z:2).
        phase: Phase of the dipole source (in radian).
        power: Power assuming uniform dielectric space with the permittivity.
    """
    type = goos.ModelNameType("source.dipole_source")
    position = goos.Vec3d()
    axis = goos.types.IntType()
    phase = goos.types.FloatType(default=0)
    power = goos.types.FloatType(default=1)


@maxwell.register(DipoleSource)
class DipoleSourceImpl(SimSourceImpl):

    def __init__(self, source: DipoleSource) -> None:
        self._src = source
        self._J = None

    def before_sim(self, sim: FdfdSimProp) -> None:
        if self._J is None:
            self._J = fdfd_solvers.dipole.build_dipole_source(
                omega=2 * np.pi / sim.wlen,
                dxes=sim.dxes,
                eps=sim.eps,
                position=sim.grid.pos2ind(self._src.position,
                                          which_shifts=None),
                axis=self._src.axis,
                power=self._src.power,
                phase=np.exp(1j * self._src.phase))

        sim.source += self._J


class SimOutput(goos.Model):
    name = goos.types.StringType(default=None)


class SimOutputImpl:
    """Represents an object that helps produce a simulation output.

    Each `SimOutputImpl` type corresponds to a `SimOutput` schema. Each
    `SimOutputImpl` can have the following hooks:
    - before_sim: Called before the simulation is started.
    - eval: Called to retrieve the output.
    - before_adjoint_sim: Called before the adjoint simulation is started.
    """

    def __init__(self, params: SimOutput) -> None:
        pass

    def before_sim(self, sim: FdfdSimProp) -> None:
        pass

    def eval(self, sim: FdfdSimProp) -> goos.Flow:
        raise NotImplemented("{} cannot be evaluated.".format(type(self)))

    def before_adjoint_sim(self, adjoint_sim: FdfdSimProp,
                           grad_val: goos.Flow.Grad) -> None:
        # If an implementation is not provided, then the gradient must be
        # `None` to indicate that the gradient is not needed.
        if grad_val:
            raise NotImplemented(
                "Cannot differentiate with respect to {}".format(type(self)))


@goos.polymorphic_model()
class Epsilon(SimOutput):
    """Displays the permittivity distribution.

    Attributes:
        wavelength: Wavelength to show permittivity.
    """
    type = goos.ModelNameType("output.epsilon")
    wavelength = goos.types.FloatType()


@maxwell.register(Epsilon, output_type=goos.Function)
class EpsilonImpl(SimOutputImpl):

    def eval(self, sim: FdfdSimProp) -> goos.NumericFlow:
        return goos.NumericFlow(sim.eps)


@goos.polymorphic_model()
class ElectricField(SimOutput):
    """Retrieves the electric field at a particular wavelength.

    Attributes:
        wavelength: Wavelength to retrieve fields.
    """
    type = goos.ModelNameType("output.electric_field")
    wavelength = goos.types.FloatType()


@maxwell.register(ElectricField, output_type=goos.Function)
class ElectricFieldImpl(SimOutputImpl):

    def eval(self, sim: FdfdSimProp) -> goos.NumericFlow:
        """Computes frequency-domain fields.

        Frequency domain fields are computed for all three components and
        stacked into one numpy array.

        Returns:
            A `NumericFlow` where the ith entry of the array corresponds to
            electric field component i (e.g. 0th entry is Ex).
        """
        return goos.NumericFlow(sim.fields)


@goos.polymorphic_model()
class WaveguideModeOverlap(SimOutput):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        wavelength: Wavelength at which to evaluate overlap.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
        normalize: If `True`, normalize the overlap by the square of the total
            power emitted at the `wavelength`.
    """
    type = goos.ModelNameType("overlap.waveguide_mode")
    wavelength = goos.types.FloatType()
    center = goos.Vec3d()
    extents = goos.Vec3d()
    normal = goos.Vec3d()
    mode_num = goos.types.IntType()
    power = goos.types.FloatType()
    normalize = goos.types.BooleanType(default=True)


@maxwell.register(WaveguideModeOverlap, output_type=goos.Function)
class WaveguideModeOverlapImpl(SimOutputImpl):

    def __init__(self, overlap: WaveguideModeOverlap) -> None:
        self._overlap = overlap
        self._wg_overlap = None

    def before_sim(self, sim: FdfdSimProp) -> None:
        # Calculate the eigenmode if we have not already.
        if self._wg_overlap is None:
            self._wg_overlap = fdfd_solvers.waveguide_mode.build_overlap(
                omega=2 * np.pi / sim.wlen,
                dxes=sim.dxes,
                eps=sim.eps,
                mu=None,
                mode_num=self._overlap.mode_num,
                waveguide_slice=simspace.create_region_slices(
                    sim.grid.exyz, self._overlap.center, self._overlap.extents),
                axis=gridlock.axisvec2axis(self._overlap.normal),
                polarity=gridlock.axisvec2polarity(self._overlap.normal),
                power=self._overlap.power)

            self._wg_overlap = np.stack(self._wg_overlap, axis=0)

    def eval(self, sim: FdfdSimProp) -> goos.NumericFlow:
        return goos.NumericFlow(np.sum(self._wg_overlap * sim.fields))

    def before_adjoint_sim(self, adjoint_sim: FdfdSimProp,
                           grad_val: goos.NumericFlow.Grad) -> None:
        adjoint_sim.source += np.conj(grad_val.array_grad * self._wg_overlap)


class Solver(goos.Model):
    pass


class SolverImpl:
    """Represents a matrix solver."""

    def __init__(self, solver: Solver, dims: Tuple[int, int, int]):
        """Initializes the solver.

        Args:
            solver: Solver parameters.
            dims: Dimensions of the simulation.
        """
        self._solver = solver
        self._dims = dims

    def solve(
        self,
        omega: complex,
        dxes: List[List[np.ndarray]],
        J: np.ndarray,
        epsilon: np.ndarray,
        pml_layers: Optional[fdfd_tools.PmlLayers] = None,
        mu: Optional[np.ndarray] = None,
        bloch_vec: Optional[np.ndarray] = None,
    ) -> None:
        pass


@goos.polymorphic_model()
class DirectSolver(Solver):
    """Defines a direct solver.

    Attributes:
        multiprocessing: If `True`, uses a multiprocessing solver, which enables
            solves to happen concurrently.
    """
    type = goos.ModelNameType("solver.direct")
    multiprocessing = goos.types.BooleanType(default=True)


@maxwell.register(DirectSolver)
class DirectSolverImpl:

    def __init__(self, solver: DirectSolver, dims: Tuple[int, int,
                                                         int]) -> None:
        if solver.multiprocessing:
            self._solver = DIRECT_SOLVER
        else:
            self._solver = local_matrix_solvers.DirectSolver()

    def solve(self, *args, **kwargs):
        return self._solver.solve(*args, **kwargs)


@goos.polymorphic_model()
class MaxwellSolver(Solver):
    """Defines a GPU-accelerated Maxwell solver.

    The Maxwell solver uses a GPU-accelerated iterative algorithm to solve
    the matrix.

    Attributes:
        solver: The specific solver to use in Maxwell.
        server: Location of the Maxwell server.
        err_thresh: Error threshold to use for the solve.
        max_iters: Maximum number of iterations in iterative solve.
    """
    type = goos.ModelNameType("solver.maxwell")
    solver = goos.types.StringType(default="maxwell_cg",
                                   choices=("maxwell_cg", "maxwell_bicgstab",
                                            "maxwell_eig"))
    server = goos.types.StringType()
    err_thresh = goos.types.FloatType(default=1e-5)
    max_iters = goos.types.FloatType(default=20000)


@maxwell.register(MaxwellSolver)
class MaxwellSolverImpl:

    def __init__(self, solver: DirectSolver, dims: Tuple[int, int,
                                                         int]) -> None:
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        server = solver.server
        if server is None:
            server = os.getenv("MAXWELL_SERVER", "localhost:9041")
        self._solver = MaxwellSolver(dims,
                                     solver=solver.solver,
                                     server=server,
                                     err_thresh=solver.err_thresh,
                                     max_iters=solver.max_iters)

    def solve(self, *args, **kwargs):
        return self._solver.solve(*args, **kwargs)


class SimulateNode(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
    node_type = "sim.maxwell.simulate_node"

    def __init__(
        self,
        eps: goos.Function,
        sources: List[SimSource],
        wavelength: float,
        simulation_space: simspace.SimulationSpace,
        outputs: List[SimOutput],
        solver: str = None,
        solver_info: Solver = None,
    ) -> None:
        # Determine the output flow types.
        output_flow_types = [
            maxwell.SIM_REGISTRY.get(out.type).meta["output_type"]
            for out in outputs
        ]
        output_names = [out.name for out in outputs]

        super().__init__([eps],
                         flow_names=output_names,
                         flow_types=output_flow_types,
                         heavy_compute=True)

        # Create an empty grid to have access to `dxes` and `shape`.
        self._grid = gridlock.Grid(simspace.create_edge_coords(
            simulation_space.sim_region,
            simulation_space.mesh.dx,
            reflection_symmetry=simulation_space.reflection_symmetry),
                                   ext_dir=gridlock.Direction.z,
                                   initial=0,
                                   num_grids=3)
        self._dxes = [self._grid.dxyz, self._grid.autoshifted_dxyz()]

        self._solver = _create_solver(solver, solver_info, self._grid.shape)
        self._simspace = simulation_space
        self._bloch_vector = [0, 0, 0]
        self._wlen = wavelength
        self._pml_layers = [
            int(length / self._simspace.mesh.dx)
            for length in self._simspace.pml_thickness
        ]
        self._symmetry = simulation_space.reflection_symmetry

        self._sources = _create_sources(sources)
        self._outputs = _create_outputs(outputs)

        # Handle caching of simulation results.
        self._last_results = None
        self._last_eps = None

    def eval(self, input_vals: List[goos.Flow]) -> goos.ArrayFlow:
        """Runs the simulation.

        Args:
            input_vals: List with single element corresponding to the
                permittivity distribution.

        Returns:
            Simulated fields.
        """
        if self._last_eps is None or self._last_eps != input_vals[0]:
            eps = input_vals[0].array
            sim = FdfdSimProp(eps=eps,
                              source=np.zeros_like(eps),
                              wlen=self._wlen,
                              dxes=self._dxes,
                              pml_layers=self._pml_layers,
                              bloch_vec=self._bloch_vector,
                              grid=self._grid,
                              solver=self._solver,
                              symmetry=self._symmetry)

            for src in self._sources:
                src.before_sim(sim)

            for out in self._outputs:
                out.before_sim(sim)

            fields = self._solver.solve(
                omega=2 * np.pi / sim.wlen,
                dxes=sim.dxes,
                epsilon=fdfd_tools.vec(sim.eps),
                mu=None,
                J=fdfd_tools.vec(sim.source),
                pml_layers=sim.pml_layers,
                bloch_vec=sim.bloch_vec,
                symmetry=sim.symmetry
            )
            fields = fdfd_tools.unvec(fields, eps[0].shape)
            sim.fields = np.stack(fields, axis=0)

            self._last_eps = input_vals[0]
            self._last_results = sim

        return goos.ArrayFlow(
            [out.eval(self._last_results) for out in self._outputs])

    def grad(self, input_vals: List[goos.Flow],
             grad_val: goos.ArrayFlow.Grad) -> List[goos.Flow.Grad]:
        eps = input_vals[0].array
        sim = FdfdSimProp(eps=eps,
                          source=np.zeros_like(eps),
                          wlen=self._wlen,
                          dxes=self._dxes,
                          pml_layers=self._pml_layers,
                          bloch_vec=self._bloch_vector,
                          grid=self._grid,
                          symmetry=self._symmetry)

        omega = 2 * np.pi / sim.wlen
        for out, g in zip(self._outputs, grad_val.flows_grad):
            out.before_adjoint_sim(sim, g)

        adjoint_fields = self._solver.solve(
            omega=2 * np.pi / sim.wlen,
            dxes=sim.dxes,
            epsilon=fdfd_tools.vec(sim.eps),
            mu=None,
            J=fdfd_tools.vec(sim.source),
            pml_layers=sim.pml_layers,
            bloch_vec=sim.bloch_vec,
            symmetry=sim.symmetry,
            adjoint=True,
        )
        adjoint_fields = np.stack(fdfd_tools.unvec(adjoint_fields,
                                                   eps[0].shape),
                                  axis=0)
        grad = -1j * omega * np.conj(adjoint_fields) * self._last_results.fields

        if np.isrealobj(eps):
            grad = 2 * np.real(grad)

        return [goos.NumericFlow.Grad(array_grad=grad)]


def _create_solver(solver_name: str, solver_info: Optional[Solver],
                   dims: Tuple[int, int, int]) -> Callable:
    """Instantiates a Maxwell solver.

    Args:
        solver_name: Name of the solver.
        simspace: Simulation space.

    Returns:
         A callable solver object.
    """
    if solver_info:
        solver = maxwell.SIM_REGISTRY.get(solver_info.type).creator(
            solver_info, dims)
    else:
        if solver_name == "maxwell_cg":
            from spins.fdfd_solvers.maxwell import MaxwellSolver
            solver = MaxwellSolver(dims, solver="CG")
        elif solver_name == "maxwell_bicgstab":
            from spins.fdfd_solvers.maxwell import MaxwellSolver
            solver = MaxwellSolver(dims, solver="biCGSTAB")
        elif solver_name == "maxwell_eig":
            from spins.fdfd_solvers.maxwell import MaxwellSolver
            solver = MaxwellSolver(dims, solver="Jacobi-Davidson")
        elif solver_name == "local_direct":
            solver = DIRECT_SOLVER
        else:
            raise ValueError("Unknown solver, got {}".format(solver_name))

    return solver


def _create_outputs(outputs: List[SimOutput]) -> List[SimOutputImpl]:
    output_impls = []
    for out in outputs:
        cls = maxwell.SIM_REGISTRY.get(out.type)
        if cls is None:
            raise ValueError("Unsupported output type, got {}".format(out.type))
        output_impls.append(cls.creator(out))
    return output_impls


def _create_sources(sources: List[SimSource]) -> List[SimSourceImpl]:
    source_impls = []
    for src in sources:
        cls = maxwell.SIM_REGISTRY.get(src.type)
        if cls is None:
            raise ValueError("Unsupported output type, got {}".format(src.type))
        source_impls.append(cls.creator(src))
    return source_impls


def fdfd_simulation(wavelength: float,
                    eps: goos.Shape,
                    background: goos.material.Material,
                    simulation_space: simspace.SimulationSpace,
                    return_eps: bool = False,
                    **kwargs) -> SimulateNode:
    eps = render.RenderShape(
        eps,
        region=simulation_space.sim_region,
        simulation_symmetry=simulation_space.reflection_symmetry,
        mesh=simulation_space.mesh,
        background=background,
        wavelength=wavelength)
    sim = SimulateNode(eps=eps,
                       wavelength=wavelength,
                       simulation_space=simulation_space,
                       **kwargs)
    if return_eps:
        return sim, eps
    else:
        return sim


@dataclasses.dataclass
class EigSimProp:
    """Represents properties of a FDFD simulation."""
    eps: np.ndarray
    source: np.ndarray
    wlen: float
    dxes: List[np.ndarray]
    pml_layers: List[int]
    bloch_vec: np.ndarray = None
    symmetry: np.ndarray = goos.np_zero_field(3)
    fields: List[np.ndarray] = None
    omegas: List[complex] = None
    grid: gridlock.Grid = None


class SimulateEigNode(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
    node_type = "sim.maxwell.simulate_eig_node"

    def __init__(
        self,
        eps: goos.Function,
        sources: List[SimSource],
        solver: str,
        wavelength: float,
        bloch_vector: List[float],
        simulation_space: simspace.SimulationSpace,
        outputs: List[SimOutput],
    ) -> None:
        # Determine the output flow types.
        output_flow_types = [
            maxwell.SIM_REGISTRY.get(out.type).meta["output_type"]
            for out in outputs
        ]
        output_names = [out.name for out in outputs]

        super().__init__([eps],
                         flow_names=output_names,
                         flow_types=output_flow_types)

        # Create an empty grid to have access to `dxes` and `shape`.
        self._grid = gridlock.Grid(simspace.create_edge_coords(
            simulation_space.sim_region,
            simulation_space.mesh.dx,
            reflection_symmetry=simulation_space.reflection_symmetry),
                                   ext_dir=gridlock.Direction.z,
                                   initial=0,
                                   num_grids=3)
        self._dxes = [self._grid.dxyz, self._grid.autoshifted_dxyz()]

        eigen_solvers = ["maxwell_eig", "local_direct_eig"]
        if solver in eigen_solvers:
            self._solver = _create_solver(solver, None, self._grid.shape)
        else:
            raise ValueError(
                "Invalid solver, solver for eigensolves need to be in " +
                str(eigen_solvers) + ".")
        self._simspace = simulation_space
        if bloch_vector:
            self._bloch_vector = bloch_vector
        else:
            self._bloch_vector = np.array([0, 0, 0])
        self._wlen = wavelength  #this will be used for the initial guess
        self._pml_layers = [
            int(length / self._simspace.mesh.dx)
            for length in self._simspace.pml_thickness
        ]
        self._symmetry = simulation_space.reflection_symmetry

        self._sources = _create_sources(sources)
        self._outputs = _create_outputs(outputs)

        # Handle caching of simulation results.
        self._last_results = None
        self._last_eps = None

    def eval(self, input_vals: List[goos.Flow]) -> goos.ArrayFlow:
        """Runs the simulation.

        Args:
            input_vals: List with single element corresponding to the
                permittivity distribution.

        Returns:
            Simulated fields.
        """
        if self._last_eps is None or self._last_eps != input_vals[0]:
            eps = input_vals[0].array
            sim = EigSimProp(eps=eps,
                             source=np.zeros_like(eps),
                             wlen=self._wlen,
                             dxes=self._dxes,
                             pml_layers=self._pml_layers,
                             bloch_vec=self._bloch_vector,
                             symmetry=self._symmetry,
                             grid=self._grid)

            for src in self._sources:
                src.before_sim(sim)

            for out in self._outputs:
                out.before_sim(sim)

            #TODO vcruysse: for now we limit to 1 eigenvalue
            fields, omega = self._solver.solve(omega=2 * np.pi / sim.wlen,
                                               dxes=sim.dxes,
                                               J=fdfd_tools.vec(sim.source),
                                               epsilon=fdfd_tools.vec(sim.eps),
                                               mu=None,
                                               bloch_vec=sim.bloch_vec,
                                               pml_layers=sim.pml_layers,
                                               symmetry=sim.symmetry,
                                               n_eig=1)
            fields = fdfd_tools.unvec(fields[0], eps[0].shape)
            sim.fields = np.stack(fields, axis=0)
            sim.omegas = np.real(omega[0])

            self._last_eps = input_vals[0]
            self._last_results = sim

        return goos.ArrayFlow(
            [out.eval(self._last_results) for out in self._outputs])

    def grad(self, input_vals: List[goos.Flow],
             grad_val: goos.ArrayFlow.Grad) -> List[goos.Flow.Grad]:
        """Runs the simulation.

        Args:
            input_vals: List with single element corresponding to the
                permittivity distribution.

        Returns:
            Simulated fields.
        """
        if self._last_eps is None or self._last_eps != input_vals[0]:
            eps = input_vals[0].array
            sim = EigSimProp(eps=eps,
                             source=np.zeros_like(eps),
                             wlen=self._wlen,
                             dxes=self._dxes,
                             pml_layers=self._pml_layers,
                             bloch_vec=self._bloch_vector,
                             grid=self._grid)

            for out in self._outputs:
                out.before_adjoint_sim(sim)

            #TODO vcruysse: for now we limit to 1 eigenvalue
            fields, omega = self._solver.solve(omega=2 * np.pi / sim.wlen,
                                               dxes=sim.dxes,
                                               J=fdfd_tools.vec(sim.source),
                                               epsilon=fdfd_tools.vec(sim.eps),
                                               mu=None,
                                               bloch_vec=sim.bloch_vec,
                                               pml_layers=sim.pml_layers,
                                               symmetry=sim.symmetry,
                                               n_eig=1)
            fields = fdfd_tools.unvec(fields[0], eps[0].shape)
            sim.fields = np.stack(fields, axis=0)
            sim.omegas = np.real(omega[0])

            self._last_eps = input_vals[0]
            self._last_results = sim

        fields = self._last_results.fields
        omega = self._last_results.omegas
        eps = self._last_results.eps
        domega_deps = np.real(-omega / 2 * 1 /
                              (fields.flatten().conj() @ fields.flatten()) *
                              fields.conj() * eps**(-1) * fields)

        # find the grad for omega
        for out, grad in zip(self._outputs, grad_val):
            if isinstance(out, EigenValueImpl):
                df_domega = grad.array_grad[0]

        return [goos.NumericFlow.Grad(array_grad=df_domega * domega_deps)]


def eig_simulation(wavelength: float,
                   eps: goos.Shape,
                   background: goos.material.Material,
                   simulation_space: simspace.SimulationSpace,
                   bloch_vector: List[float],
                   get_eps: bool = False,
                   **kwargs):
    eps = render.RenderShape(
        eps,
        region=simulation_space.sim_region,
        simulation_symmetry=simulation_space.reflection_symmetry,
        mesh=simulation_space.mesh,
        background=background,
        wavelength=wavelength)
    sim = SimulateEigNode(eps=eps,
                          wavelength=wavelength,
                          simulation_space=simulation_space,
                          bloch_vector=bloch_vector,
                          **kwargs)
    if get_eps:
        return sim, eps
    else:
        return sim


@goos.polymorphic_model()
class EigenValue(SimOutput):
    """Retrieves an eigen value from an eigensimulation.

    Attributes:
        bloch_vector: Bloch_vector to retrieve eigenvalue.
    """
    type = goos.ModelNameType("output.eigen_value")
    bloch_vector = goos.types.FloatType()


@maxwell.register(EigenValue, output_type=goos.Function)
class EigenValueImpl(SimOutputImpl):

    def eval(self, sim: EigSimProp) -> goos.NumericFlow:
        return goos.NumericFlow(sim.omegas)
