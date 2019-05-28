"""Contains function to actually execute an optimization plan.

This file can also be directly executed from command line.
"""
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import spins
from spins.invdes.problem_graph import creator
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


def run_plan(plan: optplan.OptimizationPlan,
             project_folder: str,
             save_folder: Optional[str] = None,
             resume: bool = False) -> None:
    """Executes an optimization plan.

    The GDS files referenced from the JSON file should be referenced from
    `project_folder`. Any logs will also be saved in `project_folder.`

    Args:
        plan: Plan to execute.
        project_folder: Folder containing additional project files.
        save_folder: Folder to save optimization data. Defaults to
            `project_folder` if None.
        resume: If `True`, attempt to resume optimization. This only works
            if `save_folder` exists. Optimization will resume from the last
            log file in `save_folder`.
    """
    if not save_folder:
        save_folder = project_folder

    setup_logging(save_folder)

    console_logger = logging.getLogger(__name__)

    # Make workspace.
    console_logger.info("Setting up workspace.")
    work = workspace.Workspace(project_folder, save_folder)

    # Keep track of which transformation we are executing. This exists
    # in order to handle optimization resuming in which we start executing
    # a later transformation.
    transform_index = 0

    # Most recent logging data. Used to resume transformations.
    event_data = None

    if resume:
        transform_index, event_data = restore_workspace(plan, work, save_folder,
                                                        console_logger)

    # Run over the parametrization.
    for transformation_param in plan.transformations[transform_index:]:
        console_logger.info("Running transformation %s.",
                            transformation_param.name)
        work.run(transformation_param, event_data)
        # Save the state after transformation executes.
        work.logger.write_checkpoint(transformation_param.name)
        event_data = None

    # Make a GDS if needed.
    final_parametrization = plan.transformations[-1].parametrization
    parametrization = work.get_object(final_parametrization)
    # TODO(logansu): Have a better way of generating GDS than this.
    if hasattr(parametrization, "generate_polygons"):
        simspace_name = final_parametrization.simulation_space
        poly_coords = parametrization.generate_polygons(
            work.get_object(simspace_name).dx)

        console_logger.info("Exporting GDS of final design.")
        spins.gds.gen_gds(poly_coords,
                          os.path.join(save_folder, "spins_design.gds"))

    console_logger.info("Spins finished.")


def restore_workspace(plan: optplan.OptimizationPlan, work: workspace.Workspace,
                      save_folder: str,
                      console_logger) -> Tuple[int, Optional[Dict]]:
    """Restores the workspace state for resuming optimization plans.

    This function resumes the state of workspace (all parametrization values
    and parameter values) based on the saved data. This is done in the following
    steps:
    1) The checkpoint file corresponding to the last completed transformation
        is found.
    2) All parametrization and parameter values are restored according to the
       checkpoint file.
    3) The last log file is found. If the last log file corresponds to the
        the next transformation that should be executed, the parametrization
        value is restored and the event data from the log file is extracted.

    Args:
        plan: Optimization plan that saved the log data previously.
        work: Workspace to restore.
        save_folder: Folder containing saved log files.
        console_logger: A logging object for logging restoring info.

    Returns:
        A tuple `(tranform_index, event_data)` where `transform_index` is the
        index of the transformation in the optimization plan that should be
        executed next and `event_data` is a dictionary containing the event
        data of the last saved log file.
    """
    transform_index = 0
    event_data = None

    # Find the latest transformation with a checkpoint file.
    # Set `transform_index` to the index of the transformation that should
    # be run next (i.e. there exists a checkpoint for the
    # `transform_index - 1` transformation.
    for i, transform in enumerate(plan.transformations):
        if os.path.exists(
                os.path.join(save_folder,
                             "{}.chkpt.pkl".format(transform.name))):
            transform_index = i + 1

    # Load the checkpoint data.
    # If `transform_index` is zero, this means that there is no previous
    # checkpoint so we're on the first transformation in the plan and
    # there is no need to restore any previous parametrizations or
    # parameters.
    if transform_index > 0:
        chkpt_file = os.path.join(
            save_folder, "{}.chkpt.pkl".format(
                plan.transformations[transform_index - 1].name))
        console_logger.info("Restoring from checkpoint {}".format(chkpt_file))
        with open(chkpt_file, "rb") as fp:
            chkpt_data = pickle.load(fp)

        # Iterate through all the previous transformations, restoring any
        # parametrizations and parameters along the way. It is not strictly
        # necessary to restore parametrizations/parameters along the way,
        # but it was done out of implementation convenience.
        for transform in plan.transformations[:transform_index]:
            # Restore any parametrizations.
            work.get_object(transform.parametrization).deserialize(
                chkpt_data["parametrizations"][transform.parametrization.name])

            # Add any parameter descriptions. Actual values restored below.
            if transform.parameter_list:
                for set_param in transform.parameter_list:
                    work._add_node(set_param.parameter)

        # Now restore all parameter values.
        for param, param_value in chkpt_data["parameters"].items():
            work.get_object(param).set_parameter_value(param_value)

    if transform_index >= len(plan.transformations):
        # Optimization plan is complete already.
        return transform_index, None

    # At this point, the state should be exactly the same as when
    # `plan.transformations[transform_index]` started.
    # Now we see if we should restore to the middle of the next
    # transformation.

    # Load the log file with largest step.
    log_file = workspace.get_latest_log_file(save_folder)
    if not log_file:
        return transform_index, None

    console_logger.info("Restoring from log {}".format(log_file))
    with open(log_file, "rb") as fp:
        log_data = pickle.load(fp)

    if log_data["transformation"] == plan.transformations[transform_index].name:
        # The log file is in the next transformation so restore the
        # current parametrization value.
        work.get_object(
            plan.transformations[transform_index].parametrization).deserialize(
                log_data["parametrization"])
        event_data = log_data["event"]

    # TODO(logansu): Remove hack.
    work.logger._log_counter = log_data["log_counter"]

    return transform_index, event_data


def setup_logging(save_folder: str) -> None:
    """Setup logging.

    This will setup logging to stdout as well as to a `spins.log` file within
    `save_folder`.

    Args:
        save_folder: Folder to save logs.
    """
    # Setup logging.
    log_format = "[%(asctime)-15s][%(levelname)s][%(module)s][%(funcName)s] %(message)s"
    logging.basicConfig(format=log_format)

    # Now also log to file.
    log_file_handler = logging.FileHandler(
        os.path.join(save_folder, "spins.log"))
    log_file_handler.setFormatter(logging.Formatter(log_format))
    # Add handler to root logger.
    logging.getLogger("").addHandler(log_file_handler)

    logging.getLogger("").setLevel(logging.INFO)
    # Disable requests logging because it is very verbose.
    logging.getLogger("requests").setLevel(logging.ERROR)


def main() -> None:
    """Executes an optimization plan from command line."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str)
    args = parser.parse_args()

    # Read in json.
    with open(args.json_file, "r") as fp:
        plan = optplan.loads(fp.read())

    run_plan(plan, os.path.dirname(args.json_file))


if __name__ == "__main__":
    main()
