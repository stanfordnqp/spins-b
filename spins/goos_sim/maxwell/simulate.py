from typing import Callable, List, Tuple

import copy
import dataclasses

import numpy as np

from spins import goos
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


class SimSource(goos.Model):
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


class SimOutput(goos.Model):
    name = goos.types.StringType(default=None)


@goos.polymorphic_model()
class Epsilon(SimOutput):
    """Displays the permittivity distribution.

    Attributes:
        wavelength: Wavelength to show permittivity.
    """
    type = goos.ModelNameType("output.epsilon")
    wavelength = goos.types.FloatType()

    class Attributes:
        output_type = goos.Function


@goos.polymorphic_model()
class ElectricField(SimOutput):
    """Retrieves the electric field at a particular wavelength.

    Attributes:
        wavelength: Wavelength to retrieve fields.
    """
    type = goos.ModelNameType("output.electric_field")
    wavelength = goos.types.FloatType()

    class Attributes:
        output_type = goos.Function


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

    class Attributes:
        output_type = goos.Function


class SimulateNode(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
    node_type = "sim.maxwell.simulate_node"

    def __init__(
            self,
            eps: goos.Function,
            sources: List[SimSource],
            solver: str,
            wavelength: float,
            simulation_space: simspace.SimulationSpace,
            outputs: List[SimOutput],
    ) -> None:
        # Determine the output flow types.
        output_flow_types = [out.Attributes.output_type for out in outputs]
        output_names = [out.name for out in outputs]

        super().__init__([eps],
                         flow_names=output_names,
                         flow_types=output_flow_types)

        # Create an empty grid to have access to `dxes` and `shape`.
        self._grid = gridlock.Grid(simspace.create_edge_coords(
            simulation_space.sim_region, simulation_space.mesh.dx),
                                   ext_dir=gridlock.Direction.z,
                                   initial=0,
                                   num_grids=3)
        self._dxes = [self._grid.dxyz, self._grid.autoshifted_dxyz()]

        self._solver = _create_solver(solver, self._grid.shape)
        self._simspace = simulation_space
        self._bloch_vector = [0, 0, 0]
        self._wlen = wavelength
        self._pml_layers = [
            int(length / self._simspace.mesh.dx)
            for length in self._simspace.pml_thickness
        ]

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
                              solver=self._solver)

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
                          grid=self._grid)

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
            adjoint=True,
        )
        adjoint_fields = np.stack(fdfd_tools.unvec(adjoint_fields,
                                                   eps[0].shape),
                                  axis=0)
        grad = -1j * omega * np.conj(adjoint_fields) * self._last_results.fields

        if np.isrealobj(eps):
            grad = 2 * np.real(grad)

        return [goos.NumericFlow.Grad(array_grad=grad)]


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


def _create_solver(solver_name: str, dims: Tuple[int, int, int]) -> Callable:
    """Instantiates a Maxwell solver.

    Args:
        solver_name: Name of the solver.
        simspace: Simulation space.

    Returns:
         A callable solver object.
    """
    if solver_name == "maxwell_cg":
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        solver = MaxwellSolver(dims, solver="CG")
    elif solver_name == "maxwell_bicgstab":
        from spins.fdfd_solvers.maxwell import MaxwellSolver
        solver = MaxwellSolver(dims, solver="biCGSTAB")
    elif solver_name == "local_direct":
        solver = DIRECT_SOLVER
    else:
        raise ValueError("Unknown solver, got {}".format(solver_name))

    return solver


class SimOutputImpl:
    """Represents an object that helps produce a simulation output.

    Each `SimOutputImpl` type corresponds to a `SimOutput` schema. Each
    `SimOutputImpl` can have the following hooks:
    - before_sim: Called before the simulation is started.
    - eval: Called to retrieve the output.
    - before_adjoint_sim: Called before the adjoint simulation is started.
    """

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


class EpsilonImpl(SimOutputImpl):

    def eval(self, sim: FdfdSimProp) -> goos.NumericFlow:
        return goos.NumericFlow(sim.eps)


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


def _create_outputs(outputs: List[SimOutput]) -> List[SimOutputImpl]:
    output_impls = []
    for out in outputs:
        if type(out) == Epsilon:
            output_impl = EpsilonImpl()
        elif type(out) == ElectricField:
            output_impl = ElectricFieldImpl()
        elif type(out) == WaveguideModeOverlap:
            output_impl = WaveguideModeOverlapImpl(out)
        else:
            raise ValueError("Unsupported output type, got {}".format(out.type))
        output_impls.append(output_impl)
    return output_impls


class SimSourceImpl:

    def before_sim(self, sim: FdfdSimProp) -> None:
        pass


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


def _create_sources(sources: List[SimSource]) -> List[SimSourceImpl]:
    source_impls = []
    for src in sources:
        if type(src) == WaveguideModeSource:
            source_impls.append(WaveguideModeSourceImpl(src))
        elif type(src) == GaussianSource:
            source_impls.append(GaussianSourceImpl(src))
        else:
            raise ValueError("Unsupported source type, got {}".format(src.type))
    return source_impls


class FdfdSimulation(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
    node_type = "sim.maxwell.fdfd_simulation"

    def __init__(self, eps: goos.Shape, source: goos.Function, solver: str,
                 wavelengths: List[float],
                 simulation_space: simspace.SimulationSpace,
                 background: goos.material.Material,
                 outputs: List[SimOutput]) -> None:
        # Determine the output flow types.
        output_flow_types = [out.Attributes.output_type for out in outputs]

        super().__init__([eps, source], flow_types=output_flow_types)

        # Internally setup a plan to handle the simulation.
        self._plan = goos.OptimizationPlan()
        self._sims = []
        self._eps = eps
        with self._plan:
            for wlen in wavelengths:
                # TODO(logansu): Use a placeholder dummy for the shape to
                # reduce runtime and save memory.
                eps_rendered = render.RenderShape(
                    self._eps,
                    region=simulation_space.sim_region,
                    mesh=simulation_space.mesh,
                    background=background,
                    wavelength=wlen)

                sim_result = SimulateNode(
                    eps=eps_rendered,
                    source=source,
                    wavelength=wlen,
                    simulation_space=simulation_space,
                    solver=solver,
                    outputs=outputs,
                )

                self._sims.append(sim_result)

    def eval(self, inputs):
        with self._plan:
            from spins.goos import graph_executor
            override_map = {
                self._eps: (inputs[0],
                            goos.NodeFlags(
                                const_flags=goos.NumericFlow.ConstFlags(False),
                                frozen_flags=goos.NumericFlow.ConstFlags(True)))
            }
            return goos.ArrayFlow(
                graph_executor.eval_fun(self._sims, override_map))


def fdfd_simulation(wavelength: float,
                    eps: goos.Shape,
                    background: goos.material.Material,
                    simulation_space: simspace.SimulationSpace,
                    return_eps: bool = False,
                    **kwargs) -> SimulateNode:
    eps = render.RenderShape(eps,
                             region=simulation_space.sim_region,
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
