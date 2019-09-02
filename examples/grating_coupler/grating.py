"""2D fiber-to-chip grating coupler optimization code.

This is a simple spins example that optimizes a fiber-to-chip grating coupler
for the SOI platform. See Su et al. Optics Express (2018) for details.

To run an optimization:
$ python3 grating.py run save-folder

To view results:
$ python3 grating.py view save-folder

To resume an optimization:
$ python3 grating.py resume save-folder
"""
import os

import numpy as np
from typing import List, Tuple

# `spins.invdes.problem_graph` contains the high-level spins code.
from spins.invdes import problem_graph
# Import module for handling processing optimization logs.
from spins.invdes.problem_graph import log_tools
# `spins.invdes.problem_graph.optplan` contains the optimization plan schema.
from spins.invdes.problem_graph import optplan


def run_opt(save_folder: str) -> None:
    """Main optimization script.

    This function setups the optimization and executes it.

    Args:
        save_folder: Location to save the optimization data.
    """
    os.makedirs(save_folder)

    sim_space = create_sim_space(
        "sim_fg.gds",
        "sim_bg.gds",
        box_thickness=2000,
        wg_thickness=220,
        etch_frac=0.5)
    obj, monitors = create_objective(sim_space)
    trans_list = create_transformations(
        obj, monitors, 50, 200, sim_space, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    # Save the optimization plan so we have an exact record of all the
    # parameters.
    with open(os.path.join(save_folder, "optplan.json"), "w") as fp:
        fp.write(optplan.dumps(plan))

    # Execute the optimization and indicate that the current folder (".") is
    # the project folder. The project folder is the root folder for any
    # auxiliary files (e.g. GDS files).
    problem_graph.run_plan(plan, ".", save_folder=save_folder)


def create_sim_space(
        gds_fg: str,
        gds_bg: str,
        box_thickness: float = 2000,
        wg_thickness: float = 220,
        etch_frac: float = 0.5,
) -> optplan.SimulationSpace:
    """Creates the simulation space.

    The simulation space contains information about the boundary conditions,
    gridding, and design region of the simulation.

    Args:
        gds_fg: Location of the foreground GDS file.
        gds_bg: Location of the background GDS file.
        box_thickness: Thickness of BOX layer in nm.
        wg_thickness: Thickness of the waveguide.
        etch_frac: Etch fraction of the grating. 1.0 indicates a fully-etched
            grating.

    Returns:
        A `SimulationSpace` description.
    """
    # The BOX layer/silicon device interface is set at `z = 0`.
    #
    # Describe materials in each layer.
    # We actually have four material layers:
    # 1) Silicon substrate
    # 2) Silicon oxide BOX layer
    # 3) Bottom part of grating that is not etched
    # 4) Top part of grating that can be etched.
    #
    # The last two layers put together properly describe a partial etch.
    #
    # Note that the layer numbering in the GDS file is arbitrary. In our case,
    # layer 100 and 101 correspond to actual structure. Layer 300 is a dummy
    # layer; it is used for layers that only have one material (i.e. the
    # background and foreground indices are identical) so the actual structure
    # used does not matter.
    stack = [
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="Si"),
            background=optplan.Material(mat_name="Si"),
            # Note that layer number here does not actually matter because
            # the foreground and background are the same material.
            gds_layer=[300, 0],
            extents=[-10000, -box_thickness],
        ),
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="SiO2"),
            background=optplan.Material(mat_name="SiO2"),
            gds_layer=[300, 0],
            extents=[-box_thickness, 0],
        ),
    ]
    # If `etch-frac` is 1, then we do not need two separate layers.
    if etch_frac != 1:
        stack.append(
            optplan.GdsMaterialStackLayer(
                foreground=optplan.Material(mat_name="Si"),
                background=optplan.Material(mat_name="SiO2"),
                gds_layer=[100, 0],
                extents=[0, wg_thickness * (1 - etch_frac)],
            ))
    stack.append(
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(mat_name="Si"),
            background=optplan.Material(mat_name="SiO2"),
            gds_layer=[101, 0],
            extents=[wg_thickness * (1 - etch_frac), wg_thickness],
        ))

    mat_stack = optplan.GdsMaterialStack(
        # Any region of the simulation that is not specified is filled with
        # oxide.
        background=optplan.Material(mat_name="SiO2"),
        stack=stack,
    )

    sim_z_start = -box_thickness - 1000
    sim_z_end = wg_thickness + 1500

    # Create a simulation space for both continuous and discrete optimization.
    dx = 40
    return optplan.SimulationSpace(
        name="simspace",
        mesh=optplan.UniformMesh(dx=dx),
        eps_fg=optplan.GdsEps(gds=gds_fg, mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds=gds_bg, mat_stack=mat_stack),
        # Note that we explicitly set the simulation region. Anything
        # in the GDS file outside of the simulation extents will not be drawn.
        sim_region=optplan.Box3d(
            center=[0, 0, (sim_z_start + sim_z_end) / 2],
            extents=[16000, dx, sim_z_end - sim_z_start],
        ),
        selection_matrix_type="uniform",
        # Here we are specifying periodic boundary conditions (Bloch boundary
        # conditions with zero k-vector).
        boundary_conditions=[optplan.BlochBoundary()] * 6,
        # PMLs are applied on x- and z-axes. No PMLs are applied along y-axis
        # because it is the axis of translational symmetry.
        pml_thickness=[10, 10, 0, 0, 10, 10],
    )


def create_objective(sim_space: optplan.SimulationSpace
                    ) -> Tuple[optplan.Function, List[optplan.Monitor]]:
    """Creates an objective function.

    The objective function is what is minimized during the optimization.

    Args:
        sim_space: The simulation space description.

    Returns:
        A tuple `(obj, monitor_list)` where `obj` is an objectivce function that
        tries to maximize the coupling efficiency of the grating coupler and
        `monitor_list` is a list of monitors (values to keep track of during
        the optimization.
    """
    # Keep track of metrics and fields that we want to monitor.
    monitor_list = []

    wlen = 1550
    epsilon = optplan.Epsilon(
        simulation_space=sim_space,
        wavelength=wlen,
    )
    monitor_list.append(optplan.FieldMonitor(name="mon_eps", function=epsilon))

    sim = optplan.FdfdSimulation(
        source=optplan.GaussianSource(
            polarization_angle=np.pi / 2,
            theta=0,
            psi=0,
            center=[0, 0, 920],
            extents=[14000, 14000, 0],
            normal=[0, 0, -1],
            power=1,
            w0=5200,
            normalize_by_sim=True,
        ),
        solver="local_direct",
        wavelength=wlen,
        simulation_space=sim_space,
        epsilon=epsilon,
    )
    monitor_list.append(
        optplan.FieldMonitor(
            name="mon_field",
            function=sim,
            normal=[0, 1, 0],
            center=[0, 0, 0],
        ))

    overlap = optplan.Overlap(
        simulation=sim,
        overlap=optplan.WaveguideModeOverlap(
            center=[-7000, 0, 110.0],
            extents=[0.0, 1500, 1500.0],
            mode_num=0,
            normal=[-1.0, 0.0, 0.0],
            power=1.0,
        ),
    )

    power = optplan.abs(overlap)**2
    monitor_list.append(optplan.SimpleMonitor(name="mon_power", function=power))

    # Spins minimizes the objective function, so to make `power` maximized,
    # we minimize `1 - power`.
    obj = 1 - power

    return obj, monitor_list


def create_transformations(
        obj: optplan.Function,
        monitors: List[optplan.Monitor],
        cont_iters: int,
        disc_iters: int,
        sim_space: optplan.SimulationSpaceBase,
        min_feature: float = 100,
        cont_to_disc_factor: float = 1.1,
) -> List[optplan.Transformation]:
    """Creates a list of transformations for the optimization.

    The grating coupler optimization proceeds as follows:
    1) Continuous optimization whereby each pixel can vary between device and
       background permittivity.
    2) Discretization whereby the continuous pixel parametrization is
       transformed into a discrete grating (Note that L2D is implemented here).
    3) Further optimization of the discrete grating by moving the grating
       edges.

    Args:
        opt: The objective function to minimize.
        monitors: List of monitors to keep track of.
        cont_iters: Number of iterations to run in continuous optimization.
        disc_iters: Number of iterations to run in discrete optimization.
        sim_space: Simulation space ot use.
        min_feature: Minimum feature size in nanometers.
        cont_to_disc_factor: Discretize the continuous grating with feature size
            constraint of `min_feature * cont_to_disc_factor`.
            `cont_to_disc_factor > 1` gives discrete optimization more wiggle
            room.

    Returns:
        A list of transformations.
    """
    # Setup empty transformation list.
    trans_list = []

    # First do continuous relaxation optimization.
    cont_param = optplan.PixelParametrization(
        simulation_space=sim_space,
        init_method=optplan.UniformInitializer(min_val=0, max_val=1))
    trans_list.append(
        optplan.Transformation(
            name="opt_cont",
            parametrization=cont_param,
            transformation=optplan.ScipyOptimizerTransformation(
                optimizer="L-BFGS-B",
                objective=obj,
                monitor_lists=optplan.ScipyOptimizerMonitorList(
                    callback_monitors=monitors,
                    start_monitors=monitors,
                    end_monitors=monitors),
                optimization_options=optplan.ScipyOptimizerOptions(
                    maxiter=cont_iters),
            ),
        ))

    # Discretize. Note we add a little bit of wiggle room by discretizing with
    # a slightly larger feature size that what our target is (by factor of
    # `cont_to_disc_factor`). This is to give the optimization a bit more wiggle
    # room later on.
    disc_param = optplan.GratingParametrization(
        simulation_space=sim_space, inverted=True)
    trans_list.append(
        optplan.Transformation(
            name="cont_to_disc",
            parametrization=disc_param,
            transformation=optplan.GratingEdgeFitTransformation(
                parametrization=cont_param,
                min_feature=cont_to_disc_factor * min_feature)))

    # Discrete optimization.
    trans_list.append(
        optplan.Transformation(
            name="opt_disc",
            parametrization=disc_param,
            transformation=optplan.ScipyOptimizerTransformation(
                optimizer="SLSQP",
                objective=obj,
                constraints_ineq=[
                    optplan.GratingFeatureConstraint(
                        min_feature_size=min_feature,
                        simulation_space=sim_space,
                        boundary_constraint_scale=1.0,
                    )
                ],
                monitor_lists=optplan.ScipyOptimizerMonitorList(
                    callback_monitors=monitors,
                    start_monitors=monitors,
                    end_monitors=monitors),
                optimization_options=optplan.ScipyOptimizerOptions(
                    maxiter=disc_iters),
            ),
        ))
    return trans_list


def view_opt(save_folder: str) -> None:
    """Shows the result of the optimization.

    This runs the auto-plotter to plot all the relevant data.
    See `examples/wdm2` IPython notebook for more details on how to process
    the optimization logs.

    Args:
        save_folder: Location where the log files are saved.
    """
    log_df = log_tools.create_log_data_frame(
        log_tools.load_all_logs(save_folder))
    monitor_descriptions = log_tools.load_from_yml(
        os.path.join(os.path.dirname(__file__), "monitor_spec.yml"))
    log_tools.plot_monitor_data(log_df, monitor_descriptions)


def resume_opt(save_folder: str) -> None:
    """Resumes a stopped optimization.

    This restarts an optimization that was stopped prematurely. Note that
    resuming an optimization will not lead the exact same results as if the
    optimization were finished the first time around.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())

    # Run the plan with the `resume` flag to restart.
    problem_graph.run_plan(plan, ".", save_folder=save_folder, resume=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("run", "view", "resume"),
        help="Must be either \"run\" to run an optimization, \"view\" to "
        "view the results, or \"resume\" to resume an optimization.")
    parser.add_argument(
        "save_folder", help="Folder containing optimization logs.")

    args = parser.parse_args()
    if args.action == "run":
        run_opt(args.save_folder)
    elif args.action == "view":
        view_opt(args.save_folder)
    elif args.action == "resume":
        resume_opt(args.save_folder)
