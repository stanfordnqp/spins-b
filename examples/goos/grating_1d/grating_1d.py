"""Runs optimization for a simple bardcode grating coupler in

The simulation lies in the xz-plane and the Gaussian beam is polarized along
the y-axis.
"""
import dataclasses
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos_sim import maxwell


@dataclasses.dataclass
class Options:
    """Maintains list of options for the optimization.

    Attributes:
        coupler_len: Length of the grating coupler.
        wg_width: Width of the grating coupler. Only relevant for GDS file
            generation.
        wg_len: Length of the waveguide to which the grating coupler couples.
        wg_thickness: Thickness of the waveguide.
        etch_frac: Etch fraction of the grating.
        min_features: Minimum feature sizes.
        box_size: Thickness of the buried oxide layer.
        source_angle_deg: Angle of the Gaussian beam in degrees relative to
            the normal.

        buffer_len: Additional distance to add to the top and bottom of the
            simulation for simulation accuracy.

        eps_bg: Refractive index of the background.
        eps_fg: Refraction index of the waveguide/grating.

        beam_dist: Distance of the Gaussian beam from the grating.
        beam_width: Diameter of the Gaussian beam.
        beam_extents: Length of the Gaussian beam to use in the simulation.

        wlen: Wavelength to simulate at.
        dx: Grid spacing to use in the simulation.
        pixel_size: Pixel size of the continuous grating coupler
            parametrization.
    """
    coupler_len: float = 12000
    wg_width: float = 10000
    wg_len: float = 2400
    wg_thickness: float = 220
    etch_frac: float = 0.5
    min_features: float = 100
    box_size: float = 2000
    source_angle_deg: float = -10

    buffer_len: float = 2000

    eps_bg: float = 1.444
    eps_wg: float = 3.4765

    beam_dist: float = 1000
    beam_width: float = 10400
    beam_extents: float = 14000

    wlen: float = 1550
    dx: float = 20
    pixel_size: float = 20


def main(save_folder: str, visualize: bool = False) -> None:
    goos.util.setup_logging(save_folder)

    params = Options()

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        substrate = goos.Cuboid(
            pos=goos.Constant([
                params.coupler_len / 2, 0,
                -params.box_size - params.wg_thickness / 2 - 5000
            ]),
            extents=goos.Constant([params.coupler_len + 10000, 1000, 10000]),
            material=goos.material.Material(index=params.eps_wg))
        waveguide = goos.Cuboid(
            pos=goos.Constant([-params.wg_len / 2, 0, 0]),
            extents=goos.Constant(
                [params.wg_len, params.wg_width, params.wg_thickness]),
            material=goos.material.Material(index=params.eps_wg))

        wg_bottom = goos.Cuboid(
            pos=goos.Constant([
                params.coupler_len / 2, 0,
                -params.wg_thickness / 2 * params.etch_frac
            ]),
            extents=goos.Constant([
                params.coupler_len, params.wg_width,
                params.wg_thickness * (1 - params.etch_frac)
            ]),
            material=goos.material.Material(index=params.eps_wg))

        def initializer(size):
            return np.random.random(size)

        # Continuous optimization.
        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([
                params.coupler_len / 2, 0,
                params.wg_thickness / 2 * (1 - params.etch_frac)
            ]),
            extents=[
                params.coupler_len, params.wg_width,
                params.wg_thickness * params.etch_frac
            ],
            material=goos.material.Material(index=params.eps_bg),
            material2=goos.material.Material(index=params.eps_wg),
            pixel_size=[
                params.pixel_size, params.wg_width, params.wg_thickness
            ])

        obj, sim = make_objective(
            goos.GroupShape([substrate, waveguide, wg_bottom, design]), "cont",
            params)

        goos.opt.scipy_minimize(
            obj,
            "L-BFGS-B",
            monitor_list=[sim["eps"], sim["field"], sim["overlap"], obj],
            max_iters=60,
            name="opt_cont")

        # Prevent optimization from optimizing over continuous variable.
        var.freeze()

        # Run discretization.
        grating_var, height_var, design_disc = goos.grating.discretize_to_pixelated_grating(
            var,
            height_fracs=[0, 1],
            pixel_size=params.pixel_size,
            start_height_ind=1,
            end_height_ind=1,
            min_features=params.min_features,
            pos=[
                params.coupler_len / 2, 0,
                params.wg_thickness / 2 * (1 - params.etch_frac)
            ],
            extents=[
                params.coupler_len, params.wg_width,
                params.wg_thickness * params.etch_frac
            ],
            material=goos.material.Material(index=params.eps_bg),
            material2=goos.material.Material(index=params.eps_wg),
            grating_dir=0,
            grating_dir_spacing=20,
            etch_dir=2,
            etch_dir_divs=1)

        obj, sim = make_objective(
            goos.GroupShape([substrate, waveguide, wg_bottom, design_disc]),
            "disc", params)

        goos.opt.scipy_minimize(
            obj,
            "L-BFGS-B",
            monitor_list=[sim["eps"], sim["field"], sim["overlap"], obj],
            max_iters=100,
            name="opt_disc",
            ftol=1e-8)

        plan.save()
        plan.run()

        if visualize:
            goos.util.visualize_eps(sim["eps"].get().array[2])


def make_objective(eps: goos.Shape, stage: str, params: Options):
    """Creates the objective.

    The function sets up the simulation and the objective function for the
    grating optimization. The simulation is a FDFD simulation with a
    Gaussian beam source that couples into a the waveguide.
    The optimization minimizes `(1 - coupling_eff)**2` where `coupling_eff` is
    the fiber-to-chip coupling efficiency.

    Args:
        eps: The permittivity distribution, including the waveguide and
            grating.
        stage: Name of the optimization stage. Used to name the nodes.
        params: Options for the optimization problem.

    Returns:
        A tuple `(obj, sim)` where `obj` is the objective and `sim` is the
        simulation.
    """
    solver = "local_direct"

    sim_left_x = -params.wg_len
    sim_right_x = params.coupler_len + params.buffer_len
    pml_thick = params.dx * 10
    sim_z_center = (params.wg_thickness / 2 + params.beam_dist -
                    params.box_size) / 2
    sim_z_extent = (params.wg_thickness + params.beam_dist + params.box_size +
                    2000 + pml_thick * 2)

    sim = maxwell.fdfd_simulation(
        name="sim_{}".format(stage),
        wavelength=params.wlen,
        eps=eps,
        solver=solver,
        sources=[
            maxwell.GaussianSource(
                w0=params.beam_width / 2,
                center=[
                    params.coupler_len / 2, 0,
                    params.wg_thickness / 2 + params.beam_dist
                ],
                extents=[params.beam_extents, 0, 0],
                normal=[0, 0, -1],
                power=1,
                theta=np.deg2rad(params.source_angle_deg),
                psi=np.pi / 2,
                polarization_angle=0,
                normalize_by_sim=True)
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=params.dx),
            sim_region=goos.Box3d(
                center=[(sim_left_x + sim_right_x) / 2, 0, sim_z_center],
                extents=[sim_right_x - sim_left_x, 0, sim_z_extent],
            ),
            pml_thickness=[pml_thick, pml_thick, 0, 0, pml_thick, pml_thick]),
        background=goos.material.Material(index=params.eps_bg),
        outputs=[
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(name="overlap",
                                         center=[-params.wg_len / 2, 0, 0],
                                         extents=[0, 1000, 2000],
                                         normal=[-1, 0, 0],
                                         mode_num=0,
                                         power=1),
        ],
    )

    obj = (1 - goos.abs(sim["overlap"]))**2
    obj = goos.rename(obj, name="obj_{}".format(stage))
    return obj, sim


def visualize(folder: str, step: int):
    """Visualizes result of the optimization.

    This is a quick visualization tool to plot the permittivity and electric
    field distribution at a particular save step. The function automatically
    determines whether the optimization is in continuous or discrete and
    plot the appropriate data.

    Args:
       folder: Save folder location.
       step: Save file step to load.
    """
    if step < 0:
        step = goos.util.get_latest_log_step(folder)

    with open(os.path.join(folder, "step{0}.pkl".format(step)), "rb") as fp:
        data = pickle.load(fp)

    if data["action"] == "opt_cont":
        stage = "cont"
    elif data["action"] == "opt_disc":
        stage = "disc"

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.abs(data["monitor_data"]["sim_{}.eps".format(stage)][1].squeeze()))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.abs(data["monitor_data"]["sim_{}.field".format(stage)][1].squeeze()))
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "view"))
    parser.add_argument("save_folder")
    parser.add_argument("--step", default=-1)

    args = parser.parse_args()
    if args.action == "run":
        main(save_folder=args.save_folder, visualize=True)
    elif args.action == "view":
        visualize(args.save_folder, int(args.step))
