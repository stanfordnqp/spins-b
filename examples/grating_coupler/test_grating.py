import os
import shutil

import grating
from spins.invdes import problem_graph
from spins.invdes.problem_graph import optplan

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def test_grating(tmpdir):
    folder = str(tmpdir.mkdir("grating"))

    sim_space = grating.create_sim_space(
        os.path.join(folder, "sim_fg.gds"),
        os.path.join(folder, "sim_bg.gds"),
        box_thickness=2000,
        wg_thickness=220,
        etch_frac=0.5)
    obj, monitors = grating.create_objective(
        sim_space, wg_thickness=220, grating_len=10000)
    trans_list = grating.create_transformations(
        obj, monitors, 1, 2, sim_space, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    problem_graph.run_plan(plan, folder)
