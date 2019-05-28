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
from spins.invdes.problem_graph import grid_utils
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import simspace
from spins.invdes.problem_graph.functions import poynting

TESTDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testdata")


def eval_grad_brute_wirt(vec: np.ndarray,
                         fun: Callable[[np.ndarray], float],
                         delta: float = 1e-6) -> np.ndarray:
    """Evaluates Wirtinger derivative using symmetric difference.

    By the definition, the gradient d/dz = 0.5 (d/dx - i d/dy).

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

        grad_x = (fun(vec_forward) - fun(vec_backward)) / (2 * delta)

        unit_vec = np.zeros_like(vec)
        unit_vec[i] = delta * 1j
        vec_forward = vec + unit_vec
        vec_backward = vec - unit_vec

        grad_y = (fun(vec_forward) - fun(vec_backward)) / (2 * delta)

        grad[i] = 0.5 * (grad_x - 1j * grad_y)

    return grad


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


def test_straight_waveguide_power_poynting():
    """Tests that total through straight waveguide is unity."""
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

    wlen = 1550
    eps_grid = space(wlen).eps_bg.grids
    source_grid = source(space, wlen)

    eps = problem.Constant(fdfd_tools.vec(eps_grid))
    sim = creator_em.FdfdSimulation(
        eps=eps,
        solver=local_matrix_solvers.DirectSolver(),
        wlen=wlen,
        source=fdfd_tools.vec(source_grid),
        simspace=space,
    )
    power_fun = poynting.PowerTransmissionFunction(
        field=sim,
        simspace=space,
        wlen=wlen,
        plane_slice=grid_utils.create_region_slices(
            space.edge_coords, [1770, 0, 0], [40, 1500, 600]),
        axis=gridlock.axisvec2axis([1, 0, 0]),
        polarity=gridlock.axisvec2polarity([1, 0, 0]))
    # Same as `power_fun` but with opposite normal vector. Should give same
    # answer but with a negative sign.
    power_fun_back = poynting.PowerTransmissionFunction(
        field=sim,
        simspace=space,
        wlen=wlen,
        plane_slice=grid_utils.create_region_slices(
            space.edge_coords, [1770, 0, 0], [40, 1500, 600]),
        axis=gridlock.axisvec2axis([-1, 0, 0]),
        polarity=gridlock.axisvec2polarity([-1, 0, 0]))

    efield_grid = fdfd_tools.unvec(
        graph_executor.eval_fun(sim, None), eps_grid[0].shape)

    # Calculate emitted power.
    edotj = np.real(
        fdfd_tools.vec(efield_grid) * np.conj(
            fdfd_tools.vec(source_grid))) * 40**3
    power = -0.5 * np.sum(edotj)

    np.testing.assert_almost_equal(
        graph_executor.eval_fun(power_fun, None), power, decimal=4)
    np.testing.assert_almost_equal(
        graph_executor.eval_fun(power_fun_back, None), -power, decimal=4)


def test_plane_power_grad():
    space = Simspace(
        TESTDATA,
        optplan.SimulationSpace(
            pml_thickness=[0, 0, 0, 0, 0, 0],
            mesh=optplan.UniformMesh(dx=40),
            sim_region=optplan.Box3d(
                center=[0, 0, 0],
                extents=[80, 80, 80],
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

    wlen = 1550
    power_fun = poynting.PowerTransmissionFunction(
        field=problem.Variable(1),
        simspace=space,
        wlen=wlen,
        plane_slice=grid_utils.create_region_slices(space.edge_coords,
                                                    [0, 0, 0], [40, 80, 80]),
        axis=gridlock.axisvec2axis([1, 0, 0]),
        polarity=gridlock.axisvec2polarity([1, 0, 0]))

    field = np.arange(np.prod(space.dims) * 3).astype(np.complex128) * 1j

    grad_actual = power_fun.grad([field], 1)
    fun = lambda vec: power_fun.eval([vec])
    grad_brute = eval_grad_brute_wirt(field, fun)

    np.testing.assert_array_almost_equal(grad_actual[0], grad_brute, decimal=4)
