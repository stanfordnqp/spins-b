"""Optimizes a 90 degree waveguide bend.

Usage:

Run optimization:

    $ python wdm2.py run [save-folder]

View results:

    $ python wdm2.py view [save-folder]
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos import compat
from spins.goos_sim import maxwell
from spins.invdes.problem_graph import optplan


def main(save_folder: str,
         min_feature: float = 100,
         use_cubic: bool = True,
         sim_3d: bool = False,
         visualize: bool = False) -> None:
    goos.util.setup_logging(save_folder)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        wg_in = goos.Cuboid(pos=goos.Constant([-2000, 0, 0]),
                            extents=goos.Constant([3000, 400, 220]),
                            material=goos.material.Material(index=3.45))
        wg_out = goos.Cuboid(pos=goos.Constant([0, 2000, 0]),
                             extents=goos.Constant([400, 3000, 220]),
                             material=goos.material.Material(index=3.45))

        def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            np.random.seed(247)
            return np.random.random(size) * 0.2 + 0.5

        # Continuous optimization.
        if use_cubic:
            var, design = goos.cubic_param_shape(
                initializer=initializer,
                pos=goos.Constant([0, 0, 0]),
                extents=[2000, 2000, 220],
                pixel_spacing=40,
                control_point_spacing=1.5 * min_feature,
                material=goos.material.Material(index=1),
                material2=goos.material.Material(index=3.45),
                var_name="var_cont")
        else:
            var, design = goos.pixelated_cont_shape(
                initializer=initializer,
                pos=goos.Constant([0, 0, 0]),
                extents=[2000, 2000, 220],
                material=goos.material.Material(index=1),
                material2=goos.material.Material(index=3.45),
                pixel_size=[40, 40, 220],
                var_name="var_cont")

        sigmoid_factor = goos.Variable(4, parameter=True, name="discr_factor")
        design = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design - 1)),
                           goos.Shape)
        eps = goos.GroupShape([wg_in, wg_out, design])

        # This node is purely for debugging purposes.
        eps_rendered = maxwell.RenderShape(
            design,
            region=goos.Box3d(center=[0, 0, 0], extents=[3000, 3000, 0]),
            mesh=maxwell.UniformMesh(dx=40),
            wavelength=1550,
        )
        if visualize:
            goos.util.visualize_eps(eps_rendered.get().array[2])

        obj, sim = make_objective(eps, "cont", sim_3d=sim_3d)

        for factor in [4, 6, 8]:
            sigmoid_factor.set(factor)
            goos.opt.scipy_minimize(
                obj,
                "L-BFGS-B",
                monitor_list=[sim["eps"], sim["field"], sim["overlap"], obj],
                max_iters=20,
                name="opt_cont{}".format(factor))

        plan.save()
        plan.run()

        if visualize:
            goos.util.visualize_eps(eps_rendered.get().array[2])


def make_objective(eps: goos.Shape, stage: str, sim_3d: bool):
    if sim_3d:
        sim_z_extent = 2500
        solver_info = maxwell.MaxwellSolver(solver="maxwell_cg",
                                            err_thresh=1e-2)
        pml_thickness = [400] * 6
    else:
        sim_z_extent = 40
        solver_info = maxwell.DirectSolver()
        pml_thickness = [400, 400, 400, 400, 0, 0]

    sim = maxwell.fdfd_simulation(
        name="sim_{}".format(stage),
        wavelength=1550,
        eps=eps,
        solver_info=solver_info,
        sources=[
            maxwell.WaveguideModeSource(center=[-1400, 0, 0],
                                        extents=[0, 2500, 1000],
                                        normal=[1, 0, 0],
                                        mode_num=0,
                                        power=1)
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=40),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[4000, 4000, sim_z_extent],
            ),
            pml_thickness=pml_thickness),
        background=goos.material.Material(index=1.0),
        outputs=[
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(name="overlap",
                                         center=[0, 1400, 0],
                                         extents=[2500, 0, 1000],
                                         normal=[0, 1, 0],
                                         mode_num=0,
                                         power=1),
        ],
    )

    obj = goos.rename(-goos.abs(sim["overlap"]), name="obj_{}".format(stage))
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
    if step is None:
        step = goos.util.get_latest_log_step(folder)

    step = int(step)

    with open(os.path.join(folder, "step{0}.pkl".format(step)), "rb") as fp:
        data = pickle.load(fp)

    plt.figure()
    plt.subplot(1, 2, 1)
    eps = np.linalg.norm(data["monitor_data"]["sim_cont.eps"], axis=0)
    plt.imshow(eps[:, :, eps.shape[2] // 2].squeeze())
    plt.colorbar()
    plt.subplot(1, 2, 2)
    field_norm = np.linalg.norm(data["monitor_data"]["sim_cont.field"], axis=0)
    plt.imshow(field_norm[:, :, field_norm.shape[2] // 2].squeeze())
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "view"))
    parser.add_argument("save_folder")
    parser.add_argument("--step")

    args = parser.parse_args()
    if args.action == "run":
        main(args.save_folder, visualize=False)
    elif args.action == "view":
        visualize(args.save_folder, args.step)
