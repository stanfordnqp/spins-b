import os
import shutil

import grating
from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _copyfiles(src_folder, dest_folder, filenames):
    for filename in filenames:
        shutil.copyfile(
            os.path.join(src_folder, filename),
            os.path.join(dest_folder, filename))


def test_grating(tmpdir):
    folder = str(tmpdir.mkdir("grating"))
    _copyfiles(CUR_DIR, folder, ["sim_fg.gds", "sim_bg.gds"])

    sim_space = grating.create_sim_space(
        os.path.join(folder, "sim_fg.gds"),
        os.path.join(folder, "sim_bg.gds"),
        box_thickness=2000,
        wg_thickness=220,
        etch_frac=0.5)
    obj, monitors = grating.create_objective(sim_space)
    trans_list = grating.create_transformations(
        obj, monitors, 1, 2, sim_space, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    problem_graph.run_plan(plan, folder)
