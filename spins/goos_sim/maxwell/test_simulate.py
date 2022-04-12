import logging

import numpy as np

from spins import goos
from spins.goos_sim import maxwell
from spins.goos_sim.maxwell import simspace
from spins import gridlock
from spins import fdfd_tools

logging.basicConfig(format=goos.LOG_FORMAT)
logging.getLogger("").setLevel(logging.DEBUG)


def test_simulate_2d():
    with goos.OptimizationPlan() as plan:
        sim_region = goos.Box3d(center=[0, 0, 0], extents=[4000, 4000, 40])
        sim_mesh = maxwell.UniformMesh(dx=40)

        waveguide = goos.Cuboid(pos=goos.Constant([0, 0, 0]),
                                extents=goos.Constant([6000, 500, 40]),
                                material=goos.material.Material(index=3.45))

        eps = maxwell.RenderShape(waveguide,
                                  region=sim_region,
                                  mesh=sim_mesh,
                                  background=goos.material.Material(index=1.0),
                                  wavelength=1550)

        sim = maxwell.SimulateNode(
            wavelength=1550,
            simulation_space=maxwell.SimulationSpace(
                sim_region=sim_region,
                mesh=sim_mesh,
                pml_thickness=[400, 400, 400, 400, 0, 0],
            ),
            eps=eps,
            sources=[
                maxwell.WaveguideModeSource(
                    center=[-500, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=0,
                    power=1,
                )
            ],
            solver_info=maxwell.DirectSolver(),
            outputs=[
                maxwell.Epsilon(name="eps"),
                maxwell.ElectricField(name="fields"),
                maxwell.WaveguideModeOverlap(name="overlap_forward",
                                             wavelength=1550,
                                             center=[1000, 0, 0],
                                             extents=[0, 2500, 0],
                                             normal=[1, 0, 0],
                                             mode_num=0,
                                             power=1),
                maxwell.WaveguideModeOverlap(name="overlap_backward",
                                             wavelength=1550,
                                             center=[-1000, 0, 0],
                                             extents=[0, 2500, 0],
                                             normal=[-1, 0, 0],
                                             mode_num=0,
                                             power=1),
                maxwell.WaveguideModeOverlap(name="overlap_forward2",
                                             wavelength=1550,
                                             center=[0, 0, 0],
                                             extents=[0, 2500, 0],
                                             normal=[1, 0, 0],
                                             mode_num=0,
                                             power=1),
            ])
        out = sim.get()

        # Power transmitted should be unity but numerical dispersion could
        # affect the actual transmitted power.
        assert np.abs(out[4].array)**2 >= 0.99
        assert np.abs(out[4].array)**2 <= 1.01

        # Check that waveguide power is roughly constant along waveguide.
        np.testing.assert_allclose(np.abs(out[2].array)**2,
                                   np.abs(out[4].array)**2,
                                   rtol=1e-2)

        # Check that we get minimal leakage of power flowing backwards.
        assert np.abs(out[3].array)**2 < 0.01


def test_simulate_wg_opt():
    with goos.OptimizationPlan() as plan:
        wg_in = goos.Cuboid(pos=goos.Constant([-2000, 0, 0]),
                            extents=goos.Constant([3000, 800, 220]),
                            material=goos.material.Material(index=3.45))
        wg_out = goos.Cuboid(pos=goos.Constant([2000, 0, 0]),
                             extents=goos.Constant([3000, 800, 220]),
                             material=goos.material.Material(index=3.45))

        def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            np.random.seed(247)
            return np.random.random(size) * 0.1 + np.ones(size) * 0.5

        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([0, 0, 0]),
            extents=[1000, 800, 220],
            pixel_size=[40, 40, 40],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.45),
        )
        eps = goos.GroupShape([wg_in, wg_out, design])
        sim = maxwell.fdfd_simulation(
            eps=eps,
            wavelength=1550,
            solver="local_direct",
            sources=[
                maxwell.WaveguideModeSource(
                    center=[-1000, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=2,
                    power=1,
                )
            ],
            simulation_space=maxwell.SimulationSpace(
                mesh=maxwell.UniformMesh(dx=40),
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 40],
                ),
                pml_thickness=[400, 400, 400, 400, 0, 0]),
            background=goos.material.Material(index=1.0),
            outputs=[
                maxwell.Epsilon(),
                maxwell.ElectricField(),
                maxwell.WaveguideModeOverlap(wavelength=1550,
                                             center=[1000, 0, 0],
                                             extents=[0, 2500, 0],
                                             normal=[1, 0, 0],
                                             mode_num=0,
                                             power=1),
            ])

        obj = -goos.abs(sim[2])

        goos.opt.scipy_minimize(obj,
                                "L-BFGS-B",
                                monitor_list=[obj],
                                max_iters=3)
        plan.run()

        assert obj.get().array < -0.30


def test_simulate_wg_opt_grad():
    with goos.OptimizationPlan() as plan:
        wg_in = goos.Cuboid(pos=goos.Constant([-2000, 0, 0]),
                            extents=goos.Constant([3000, 800, 220]),
                            material=goos.material.Material(index=3.45))
        wg_out = goos.Cuboid(pos=goos.Constant([2000, 0, 0]),
                             extents=goos.Constant([3000, 800, 220]),
                             material=goos.material.Material(index=3.45))

        def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            np.random.seed(247)
            return np.random.random(size)

        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([0, 0, 0]),
            extents=[1000, 800, 220],
            pixel_size=[500, 400, 220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.45),
        )
        eps = goos.GroupShape([wg_in, wg_out, design])
        sim = maxwell.fdfd_simulation(
            eps=eps,
            wavelength=1550,
            solver="local_direct",
            sources=[
                maxwell.WaveguideModeSource(
                    center=[-1000, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=2,
                    power=1,
                )
            ],
            simulation_space=maxwell.SimulationSpace(
                mesh=maxwell.UniformMesh(dx=40),
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 40],
                ),
                pml_thickness=[400, 400, 400, 400, 0, 0]),
            background=goos.material.Material(index=1.0),
            outputs=[
                maxwell.Epsilon(),
                maxwell.ElectricField(),
                maxwell.WaveguideModeOverlap(wavelength=1550,
                                             center=[1000, 0, 0],
                                             extents=[0, 2500, 0],
                                             normal=[1, 0, 0],
                                             mode_num=0,
                                             power=1),
            ])

        obj = -goos.abs(sim[2])

        adjoint_grad = obj.get_grad([var])[0].array_grad

        # Calculate brute force gradient.
        var_val = var.get().array
        eps = 0.001
        num_grad = np.zeros_like(var_val)
        for i in range(var_val.shape[0]):
            for j in range(var_val.shape[1]):
                temp_val = var_val.copy()
                temp_val[i, j] += eps
                var.set(temp_val)
                fplus = obj.get(run=True).array

                temp_val = var_val.copy()
                temp_val[i, j] -= eps
                var.set(temp_val)
                fminus = obj.get(run=True).array

                num_grad[i, j] = (fplus - fminus) / (2 * eps)

        np.testing.assert_array_almost_equal(adjoint_grad, num_grad, decimal=3)


def test_gaussian_source_2d():
    grid = gridlock.Grid(simspace.create_edge_coords(
        goos.Box3d(center=[0, 0, 0], extents=[14000, 40, 3000]), 40),
                         ext_dir=gridlock.Direction.z,
                         initial=1,
                         num_grids=3)
    grid.render()
    eps = np.array(grid.grids)
    dxes = [grid.dxyz, grid.autoshifted_dxyz()]
    sim = maxwell.FdfdSimProp(eps=eps,
                              source=np.zeros_like(eps),
                              wlen=1550,
                              dxes=dxes,
                              pml_layers=[10, 10, 0, 0, 10, 10],
                              grid=grid,
                              solver=maxwell.DIRECT_SOLVER)
    src = maxwell.GaussianSourceImpl(
        maxwell.GaussianSource(w0=5200,
                               center=[0, 0, 0],
                               extents=[14000, 0, 0],
                               normal=[0, 0, -1],
                               power=1,
                               theta=0,
                               psi=0,
                               polarization_angle=np.pi / 2,
                               normalize_by_sim=True))

    src.before_sim(sim)

    fields = maxwell.DIRECT_SOLVER.solve(
        omega=2 * np.pi / sim.wlen,
        dxes=sim.dxes,
        epsilon=fdfd_tools.vec(sim.eps),
        mu=None,
        J=fdfd_tools.vec(sim.source),
        pml_layers=sim.pml_layers,
        bloch_vec=sim.bloch_vec,
    )
    fields = fdfd_tools.unvec(fields, grid.shape)
    field_y = fields[1].squeeze()

    np.testing.assert_allclose(field_y[:, 53],
                               np.zeros_like(field_y[:, 53]),
                               atol=1e-4)

    # Calculate what the amplitude of the Gaussian field should look like.
    # Amplitude determined empirically through simulation.
    coords = (np.arange(len(field_y[:, 20])) - len(field_y[:, 20]) / 2) * 40
    target_gaussian = np.exp(-coords**2 / 5200**2) * 0.00278
    np.testing.assert_allclose(np.abs(field_y[:, 20]),
                               target_gaussian,
                               atol=1e-4)


if __name__ == "__main__":
    # test_simulate_wg_opt()
    test_simulate_wg_opt_grad()
