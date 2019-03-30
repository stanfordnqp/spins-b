import os
from typing import Callable, Tuple

import numpy as np

from spins import fdfd_tools
from spins import gridlock
from spins.fdfd_solvers import local_matrix_solvers
from spins.invdes import parametrization
from spins.invdes import problem
from spins.invdes.problem import graph_executor
from spins.invdes.problem_graph import creator_em
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import simspace
from spins.invdes.problem_graph import workspace

TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")


def eval_grad_brute(vec: np.ndarray,
                    fun: Callable[[np.ndarray], float],
                    delta: float = 1e-6) -> np.ndarray:
    """Evaluates a gradient using central difference.

    Args:
        vec: Input vector at which to evaluate gradient.
        fun: Function whose gradient to evaluate.
        delta: Step size for gradient.

    Returns:
        Gradient.
    """
    grad = np.zeros_like(vec)
    for i in range(len(grad)):
        unit_vec = np.zeros_like(vec)
        unit_vec[i] = delta
        vec_forward = vec + unit_vec
        vec_backward = vec - unit_vec

        grad[i] = (fun(vec_forward) - fun(vec_backward)) / (2 * delta)

    return grad


def make_simspace():
    mat_stack = optplan.GdsMaterialStack(
        background=optplan.Material(mat_name="Air"),
        stack=[
            optplan.GdsMaterialStackLayer(
                foreground=optplan.Material(mat_name="SiO2"),
                background=optplan.Material(mat_name="SiO2"),
                gds_layer=[101, 0],
                extents=[-10000, -110],
            ),
            optplan.GdsMaterialStackLayer(
                foreground=optplan.Material(mat_name="Si"),
                background=optplan.Material(mat_name="Air"),
                gds_layer=[100, 0],
                extents=[-110, 110],
            ),
        ],
    )
    simspace_spec = optplan.SimulationSpace(
        mesh=optplan.UniformMesh(dx=110),
        eps_fg=optplan.GdsEps(gds="WDM_example_fg.gds", mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds="WDM_example_bg.gds", mat_stack=mat_stack),
        sim_region=optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 300]),
        pml_thickness=[10, 10, 10, 10, 0, 0],
    )
    return simspace.SimulationSpace(simspace_spec, TESTDATA)


def test_epsilon_eval():
    space = make_simspace()
    space_inst = space(1550)
    param_shape = np.prod(space.design_dims)
    vec = np.ones(param_shape)
    vec[10:20] = 0.6
    vec[30:40] = 0.2

    eps = creator_em.Epsilon(problem.Variable(1), 1550, space)

    eps_actual = eps.eval([vec])
    eps_expected = (fdfd_tools.vec(space_inst.eps_bg.grids) +
                    space_inst.selection_matrix @ vec)

    np.testing.assert_array_equal(eps_actual, eps_expected)


def test_epsilon_grad():
    # Compute derivative of `sum(epsilon)`.
    space = make_simspace()
    space_inst = space(1550)
    param_shape = np.prod(space.design_dims)
    vec = np.ones(param_shape) * 0.4
    vec[10:20] = 0.6
    vec[20:40] = 0.2

    eps = creator_em.Epsilon(problem.Variable(1), 1550, space)

    # Brute force gradient.
    fun = lambda vec: np.sum(eps.eval([vec]))
    grad_brute = eval_grad_brute(vec, fun)
    grad_actual = eps.grad([vec], np.ones(3 * np.prod(space.dims)))
    np.testing.assert_array_almost_equal(grad_actual[0], grad_brute, decimal=4)


def test_epsilon_string():
    space = make_simspace()
    space_inst = space(1550)
    eps = creator_em.Epsilon(problem.Variable(1), 1550, space)

    assert str(eps) == "Epsilon(1550)"


def test_fdfd_simulation_grad():
    # Create a 3x3 2D grid to brute force check adjoint gradients.
    shape = [3, 3, 1]
    # Setup epsilon (pure vacuum).
    epsilon = [np.ones(shape) for i in range(3)]
    # Setup dxes. Assume dx = 40.
    dxes = [[np.ones(shape[i]) * 40 for i in range(3)] for j in range(2)]
    # Setup a point source in the center.
    J = [np.zeros(shape).astype(complex) for i in range(3)]
    J[2][1, 0, 0] = 1.2j
    J[2][1, 1, 0] = 1

    # Setup target fields.
    target_fields = [np.zeros(shape).astype(np.complex128) for i in range(3)]
    target_fields[2][:, :, 0] = 20j + 1
    overlap_vec = fdfd_tools.vec(target_fields)

    # TODO(logansu): Deal with this.
    class SimspaceMock:

        @property
        def dxes(self):
            return dxes

        @property
        def pml_layers(self):
            return [0] * 6

    eps_param = parametrization.DirectParam(
        fdfd_tools.vec(epsilon), bounds=[0, 100])
    eps_fun = problem.Variable(len(fdfd_tools.vec(epsilon)))
    sim_fun = creator_em.FdfdSimulation(
        eps=eps_fun,
        solver=local_matrix_solvers.DirectSolver(),
        wlen=1500,
        source=J,
        simspace=SimspaceMock(),
    )
    obj_fun = problem.AbsoluteValue(
        objective=creator_em.OverlapFunction(sim_fun, overlap_vec))**2

    grad_actual = obj_fun.calculate_gradient(eps_param)

    def eval_fun(vec: np.ndarray):
        eps_param.from_vector(vec)
        return obj_fun.calculate_objective_function(eps_param)

    grad_brute = eval_grad_brute(fdfd_tools.vec(epsilon), eval_fun)
    np.testing.assert_array_almost_equal(grad_actual, grad_brute, decimal=0)


def test_straight_waveguide_power():
    """Tests that a straight waveguide with a single source and overlap."""

    # TODO(logansu): Refactor.
    class Simspace:

        def __init__(self, filepath, params: optplan.SimulationSpace):
            # Setup the grid.
            self._dx = params.mesh.dx
            from spins.invdes.problem_graph.simspace import _create_edge_coords
            self._edge_coords = _create_edge_coords(params.sim_region, self._dx)
            self._ext_dir = gridlock.Direction.z  # Currently always extrude in z.
            # TODO(logansu): Factor out grid functionality and drawing.
            # Create a grid object just so we can calculate dxes.
            self._grid = gridlock.Grid(
                self._edge_coords, ext_dir=self._ext_dir, num_grids=3)

            self._pml_layers = params.pml_thickness
            self._filepath = filepath
            self._eps_bg = params.eps_bg

        @property
        def dx(self) -> float:
            return self._dx

        @property
        def dxes(self) -> fdfd_tools.GridSpacing:
            return [self._grid.dxyz, self._grid.autoshifted_dxyz()]

        @property
        def pml_layers(self) -> fdfd_tools.PmlLayers:
            return self._pml_layers

        @property
        def dims(self) -> Tuple[int, int, int]:
            return [
                len(self._edge_coords[0]) - 1,
                len(self._edge_coords[1]) - 1,
                len(self._edge_coords[2]) - 1
            ]

        @property
        def edge_coords(self) -> fdfd_tools.GridSpacing:
            return self._edge_coords

        def __call__(self, wlen: float):
            from spins.invdes.problem_graph.simspace import _create_grid
            from spins.invdes.problem_graph.simspace import SimulationSpaceInstance
            eps_bg = _create_grid(self._eps_bg, self._edge_coords, wlen,
                                  self._ext_dir, self._filepath)
            return SimulationSpaceInstance(eps_bg=eps_bg, selection_matrix=None)

    space = Simspace(
        TESTDATA,
        optplan.SimulationSpace(
            pml_thickness=[10, 10, 10, 10, 0, 0],
            mesh=optplan.UniformMesh(dx=40),
            sim_region=optplan.Box3d(
                center=[0, 0, 0],
                extents=[5000, 5000, 40],
            ),
            eps_bg=optplan.GdsEps(
                gds="straight_waveguide.gds",
                mat_stack=optplan.GdsMaterialStack(
                    background=optplan.Material(mat_name="air"),
                    stack=[
                        optplan.GdsMaterialStackLayer(
                            gds_layer=[100, 0],
                            extents=[-80, 80],
                            foreground=optplan.Material(mat_name="Si"),
                            background=optplan.Material(mat_name="air"),
                        ),
                    ],
                ),
            ),
        ))

    source = creator_em.WaveguideModeSource(
        optplan.WaveguideModeSource(
            power=1.0,
            extents=[40, 1500, 600],
            normal=[1.0, 0.0, 0.0],
            center=[-1770, 0, 0],
            mode_num=0,
        ))

    overlap = creator_em.WaveguideModeOverlap(
        optplan.WaveguideModeOverlap(
            power=1.0,
            extents=[40, 1500, 600],
            normal=[1.0, 0.0, 0.0],
            center=[1770, 0, 0],
            mode_num=0,
        ))

    wlen = 1550
    eps_grid = space(wlen).eps_bg.grids
    source_grid = source(space, wlen)
    overlap_grid = overlap(space, wlen)

    eps = problem.Constant(fdfd_tools.vec(eps_grid))
    sim = creator_em.FdfdSimulation(
        eps=eps,
        solver=local_matrix_solvers.DirectSolver(),
        wlen=wlen,
        source=fdfd_tools.vec(source_grid),
        simspace=space,
    )
    overlap_fun = creator_em.OverlapFunction(sim, fdfd_tools.vec(overlap_grid))

    efield_grid = fdfd_tools.unvec(
        graph_executor.eval_fun(sim, None), eps_grid[0].shape)

    # Calculate emitted power.
    edotj = np.real(
        fdfd_tools.vec(efield_grid) * np.conj(
            fdfd_tools.vec(source_grid))) * 40**3
    power = -0.5 * np.sum(edotj)
    # Allow for 4% error in emitted power.
    assert power > 0.96 and power < 1.04

    # Check that overlap observes nearly unity power.
    np.testing.assert_almost_equal(
        np.abs(graph_executor.eval_fun(overlap_fun, None))**2, 1, decimal=2)
