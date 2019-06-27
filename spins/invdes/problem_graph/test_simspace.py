import os

import numpy as np

from spins import fdfd_tools
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import simspace
from spins.invdes.problem_graph.simspace import _get_mat_index

TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")


def test_simspace_direct():
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
        mesh=optplan.UniformMesh(dx=40),
        eps_fg=optplan.GdsEps(gds="WDM_example_fg.gds", mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds="WDM_example_bg.gds", mat_stack=mat_stack),
        sim_region=optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 500]),
        pml_thickness=[10, 10, 10, 10, 0, 0],
    )
    space = simspace.SimulationSpace(simspace_spec, TESTDATA)
    space_inst = space(1550)

    eps_bg = space_inst.eps_bg.grids
    eps_fg = fdfd_tools.unvec(
        fdfd_tools.vec(space_inst.eps_bg.grids) +
        space_inst.selection_matrix @ np.ones(np.prod(space.design_dims)),
        space_inst.eps_bg.shape)

    assert space_inst.selection_matrix.shape == (609375, 10000)

    np.testing.assert_array_equal(eps_bg[2][:, :, -3], 1)
    np.testing.assert_allclose(eps_bg[2][10, 10, 2], 2.0852)

    np.testing.assert_allclose(eps_fg[2][107, 47, 2], 2.0852)
    np.testing.assert_allclose(eps_fg[2][107, 47, 6], 12.086617)
    np.testing.assert_allclose(eps_fg[2][112, 57, 6], 1)
    np.testing.assert_allclose(eps_fg[2][107, 47, 10], 1)


def test_simspace_reduced():
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
        mesh=optplan.UniformMesh(dx=40),
        eps_fg=optplan.GdsEps(gds="WDM_example_fg.gds", mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds="WDM_example_bg.gds", mat_stack=mat_stack),
        sim_region=optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 500]),
        pml_thickness=[10, 10, 10, 10, 0, 0],
        selection_matrix_type=optplan.SelectionMatrixType.REDUCED.value,
    )
    space = simspace.SimulationSpace(simspace_spec, TESTDATA)
    space_inst = space(1550)

    eps_bg = space_inst.eps_bg.grids
    eps_fg = fdfd_tools.unvec(
        fdfd_tools.vec(space_inst.eps_bg.grids) +
        space_inst.selection_matrix @ np.ones(np.prod(space._design_dims)),
        space_inst.eps_bg.shape)

    assert space_inst.selection_matrix.shape == (609375, 2601)

    np.testing.assert_array_equal(eps_bg[2][:, :, -3], 1)
    np.testing.assert_allclose(eps_bg[2][10, 10, 2], 2.0852)

    np.testing.assert_allclose(eps_fg[2][107, 47, 2], 2.0852)
    np.testing.assert_allclose(eps_fg[2][107, 47, 6], 12.086617)
    np.testing.assert_allclose(eps_fg[2][112, 57, 6], 1)
    np.testing.assert_allclose(eps_fg[2][107, 47, 10], 1)


def test_get_mat_index_from_name():
    mat = optplan.Material(mat_name="air")
    assert _get_mat_index(mat, 1550) == 1


def test_get_mat_index_from_index():
    mat = optplan.Material(index=optplan.ComplexNumber(real=2, imag=3))
    assert _get_mat_index(mat) == 2 + 3j


def test_simspace_mesh_list():
    """Checks parity between `GdsMeshEps` and `GdsEps`."""

    # First create using `GdsEps`.
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
        mesh=optplan.UniformMesh(dx=40),
        eps_fg=optplan.GdsEps(gds="WDM_example_fg.gds", mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds="WDM_example_bg.gds", mat_stack=mat_stack),
        sim_region=optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 500]),
        pml_thickness=[10, 10, 10, 10, 0, 0],
    )
    space = simspace.SimulationSpace(simspace_spec, TESTDATA)
    space_inst = space(1550)

    eps_bg = space_inst.eps_bg.grids
    eps_fg = fdfd_tools.unvec(
        fdfd_tools.vec(space_inst.eps_bg.grids) +
        space_inst.selection_matrix @ np.ones(np.prod(space.design_dims)),
        space_inst.eps_bg.shape)

    # Validate that `GdsEps` behaves as expected.
    assert space_inst.selection_matrix.shape == (609375, 10000)

    np.testing.assert_array_equal(eps_bg[2][:, :, -3], 1)
    np.testing.assert_allclose(eps_bg[2][10, 10, 2], 2.0852)

    np.testing.assert_allclose(eps_fg[2][107, 47, 2], 2.0852)
    np.testing.assert_allclose(eps_fg[2][107, 47, 6], 12.086617)
    np.testing.assert_allclose(eps_fg[2][112, 57, 6], 1)
    np.testing.assert_allclose(eps_fg[2][107, 47, 10], 1)

    # Now create space using `GdsMeshEps`.
    mesh_list = [
        optplan.SlabMesh(
            material=optplan.Material(mat_name="SiO2"), extents=[-10000, -110]),
        optplan.GdsMesh(
            material=optplan.Material(mat_name="Si"),
            extents=[-110, 110],
            gds_layer=[100, 0],
        ),
    ]
    simspace_spec = optplan.SimulationSpace(
        mesh=optplan.UniformMesh(dx=40),
        eps_fg=optplan.GdsMeshEps(
            gds="WDM_example_fg.gds",
            background=optplan.Material(mat_name="Air"),
            mesh_list=mesh_list),
        eps_bg=optplan.GdsMeshEps(
            gds="WDM_example_bg.gds",
            background=optplan.Material(mat_name="Air"),
            mesh_list=mesh_list),
        sim_region=optplan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 500]),
        pml_thickness=[10, 10, 10, 10, 0, 0],
    )
    space_mesh = simspace.SimulationSpace(simspace_spec, TESTDATA)
    space_inst_mesh = space(1550)

    eps_bg_mesh = space_inst_mesh.eps_bg.grids
    eps_fg_mesh = fdfd_tools.unvec(
        fdfd_tools.vec(space_inst_mesh.eps_bg.grids) +
        space_inst_mesh.selection_matrix @ np.ones(
            np.prod(space_mesh.design_dims)), space_inst_mesh.eps_bg.shape)

    # Verify that the two methods yield the same permittivities.
    np.testing.assert_allclose(eps_bg_mesh, eps_bg)
    np.testing.assert_allclose(eps_fg_mesh, eps_fg)
