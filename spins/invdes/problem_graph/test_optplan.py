import copy
import json
import os

import pytest

from schematics import models
from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils
from spins.invdes.problem_graph.optplan import io


def generate_wdm_2d():
    optplan.reset_graph()

    # Alias optplan to plan to reduce length.
    plan = optplan

    def material(index: float):
        return plan.Material(index=plan.ComplexNumber(real=index))

    mat_stack = plan.GdsMaterialStack(
        background=material(1.0),
        stack=[
            plan.GdsMaterialStackLayer(
                foreground=plan.Material(mat_name="SiO2"),
                background=plan.Material(mat_name="Air"),
                gds_layer=[101, 0],
                extents=[-10000, -110],
            ),
            plan.GdsMaterialStackLayer(
                foreground=plan.Material(mat_name="Si"),
                background=plan.Material(mat_name="Air"),
                gds_layer=[100, 0],
                extents=[-110, 110],
            ),
        ],
    )
    simspace0 = plan.SimulationSpace(
        name="simspace_cont",
        mesh=plan.UniformMesh(dx=40),
        eps_fg=plan.GdsEps(gds="WDM_example_fg.gds", mat_stack=mat_stack),
        eps_bg=plan.GdsEps(gds="WDM_example_bg.gds", mat_stack=mat_stack),
        sim_region=plan.Box3d(center=[0, 0, 0], extents=[5000, 5000, 40]),
        boundary_conditions=[plan.BlochBoundary()] * 6,
        pml_thickness=[10, 10, 10, 10, 0, 0],
    )
    simspace1 = copy.deepcopy(simspace0)
    simspace1.name = "simspace_discrete"
    simspace1.eps_fg.gds = "WDM_example_discrete_fg.gds"
    simspace1.eps_bg.gds = "WDM_example_discrete_bg.gds"

    source = plan.WaveguideModeSource(
        center=[-1620, 0, 0],
        extents=[40, 1500, 40],
        normal=[1, 0, 0],
        mode_num=0,
        power=1,
    )
    overlap_top = plan.WaveguideModeOverlap(
        center=[1620, 650, 0],
        extents=[40, 1500, 40],
        normal=[1, 0, 0],
        mode_num=0,
        power=1,
    )
    overlap_bot = plan.WaveguideModeOverlap(
        center=[1620, -650, 0],
        extents=[40, 1500, 40],
        normal=[1, 0, 0],
        mode_num=0,
        power=1,
    )

    param_cont = plan.CubicParametrization(
        undersample=8,
        simulation_space=simspace0,
        init_method=plan.UniformInitializer(min_val=0.6, max_val=1.0),
    )
    param_disc = plan.HermiteLevelSetParametrization(
        undersample=6,
        simulation_space=simspace1,
        init_method=plan.UniformInitializer(min_val=-0.2, max_val=0.2),
    )

    def build_objective(simspace: plan.SimulationSpace, opt_stage: str):
        sim1550 = plan.FdfdSimulation(
            simulation_space=simspace,
            source=source,
            wavelength=1550,
            solver="local_direct",
        )
        sim1300 = plan.FdfdSimulation(
            simulation_space=simspace,
            source=source,
            wavelength=1300,
            solver="local_direct",
        )
        eps1550 = plan.Epsilon(
            simulation_space=simspace,
            wavelength=1550,
        )
        eps1300 = plan.Epsilon(
            simulation_space=simspace,
            wavelength=1300,
        )
        sim_mons = []
        for sim, name in zip([sim1550, sim1300], ["sim1550", "sim1300"]):
            sim_mons.append(
                plan.FieldMonitor(
                    function=sim,
                    name=opt_stage + "_" + name + "_mon",
                    center=[0, 0, 0],
                    normal=[0, 0, 1],
                ))
        eps_mons = []
        for eps, name in zip([eps1550, eps1300], ["eps1550", "eps1300"]):
            eps_mons.append(
                plan.FieldMonitor(
                    function=eps,
                    name=opt_stage + "_" + name + "_mon",
                    center=[0, 0, 0],
                    normal=[0, 0, 1],
                ))

        def powercomp(overlap: plan.Overlap, target: float):
            return plan.PowerComp(
                function=plan.Power(function=plan.Abs(function=overlap), exp=2),
                value=target,
                range=0.01,
                exp=2)

        overlaps = [
            plan.Overlap(simulation=sim1550, overlap=overlap_top),
            plan.Overlap(simulation=sim1550, overlap=overlap_bot),
            plan.Overlap(simulation=sim1300, overlap=overlap_top),
            plan.Overlap(simulation=sim1300, overlap=overlap_bot),
        ]
        overlap_mons = []
        for overlap, name in zip(
                overlaps,
            ["sim1550_top", "sim1550_bot", "sim1300_top", "sim1300_bot"]):
            overlap_mons.append(
                plan.SimpleMonitor(
                    name=opt_stage + "_" + name + "_mon", function=overlap))

        return (plan.Sum(functions=[
            powercomp(over, val) for over, val in zip(overlaps, [1, 0, 0, 1])
        ]), overlap_mons, sim_mons, eps_mons)

    # Make objectives.
    obj_cont, overlaps_cont, sims_cont, eps_cont = build_objective(
        simspace0, "cont")
    obj_disc, overlaps_disc, sims_disc, eps_disc = build_objective(
        simspace1, "disc")
    obj_diff_eps = plan.DiffEpsilon(
        epsilon=plan.Epsilon(
            simulation_space=simspace1,
            wavelength=1550,
        ),
        epsilon_ref=plan.ParamEps(
            parametrization=param_cont,
            simulation_space=simspace0,
            wavelength=1550,
        ),
    )

    # Make Fabcon constraint.
    fabcon = plan.FabricationConstraint(
        minimum_curvature_diameter=120,
        minimum_gap=120,
        simulation_space=simspace1,
        method="gap_and_curve",
        apply_factors=True)

    # Make parameters.
    em_weight = plan.Parameter(initial_value=1)
    fabcon_weight = plan.Parameter(initial_value=1)
    fit_weight = plan.Parameter(initial_value=1)

    # Make weighted objectives and constraints.
    obj_disc_weighted = plan.Product(functions=[em_weight, obj_disc])
    obj_diff_eps_weighted = plan.Product(functions=[fit_weight, obj_diff_eps])
    fabcon_weighted = plan.Product(functions=[fabcon_weight, fabcon])

    transformations = [
        plan.Transformation(
            name="continuous/0",
            parametrization=param_cont,
            transformation=plan.ScipyOptimizerTransformation(
                optimizer="L-BFGS-B",
                objective=obj_cont,
                optimization_options=plan.ScipyOptimizerOptions(maxiter=10),
                monitor_lists=plan.ScipyOptimizerMonitorList(
                    callback_monitors=overlaps_cont,
                    start_monitors=overlaps_cont + sims_cont + eps_cont,
                    end_monitors=overlaps_cont + sims_cont + eps_cont,
                ),
            ),
        ),
        plan.Transformation(
            name="change_k/0",
            parametrization=param_cont,
            transformation=plan.CubicParamSigmoidStrength(value=8)),
        plan.Transformation(
            name="continuous/1",
            parametrization=param_cont,
            transformation=plan.ScipyOptimizerTransformation(
                optimizer="L-BFGS-B",
                objective=obj_cont,
                optimization_options=plan.ScipyOptimizerOptions(maxiter=10),
                monitor_lists=plan.ScipyOptimizerMonitorList(
                    callback_monitors=overlaps_cont,
                    start_monitors=overlaps_cont + sims_cont + eps_cont,
                    end_monitors=overlaps_cont + sims_cont + eps_cont,
                ),
            ),
        ),
        plan.Transformation(
            name="cont_to_disc",
            parametrization=param_disc,
            transformation=plan.ScipyOptimizerTransformation(
                optimizer="L-BFGS-B",
                objective=obj_diff_eps,
                optimization_options=plan.ScipyOptimizerOptions(maxiter=10),
                monitor_lists=plan.ScipyOptimizerMonitorList(
                    start_monitors=sims_disc,
                    end_monitors=sims_disc,
                ),
            ),
        ),
        plan.Transformation(
            name="fix_borders/0",
            parametrization=param_disc,
            transformation=plan.HermiteParamFixBorder(
                border_layers=[3, 3, 3, 3])),
        plan.Transformation(
            name="fit_cont2disc_constraint",
            parametrization=param_disc,
            parameter_list=[
                plan.SetParam(
                    parameter=fabcon_weight,
                    function=fabcon,
                    parametrization=param_disc,
                    inverse=True),
                plan.SetParam(
                    parameter=fit_weight,
                    function=obj_diff_eps,
                    parametrization=param_disc,
                    inverse=True)
            ],
            transformation=plan.PenaltyTransformation(
                optimizer="L-BFGS-B",
                objective=obj_diff_eps_weighted,
                constraints_ineq=[fabcon_weighted],
                monitor_lists=plan.ScipyOptimizerMonitorList(
                    end_monitors=sims_disc + eps_disc),
                optimization_options=plan.PenaltyOptimizerOptions(
                    maxiter=30, num_cycles=4))),
        plan.Transformation(
            name="discrete/0",
            parametrization=param_disc,
            parameter_list=[
                plan.SetParam(
                    parameter=em_weight,
                    function=obj_disc,
                    parametrization=param_disc,
                    inverse=True)
            ],
            transformation=plan.PenaltyTransformation(
                optimizer="L-BFGS-B",
                objective=obj_disc_weighted,
                constraints_ineq=[fabcon_weighted],
                optimization_options=plan.PenaltyOptimizerOptions(
                    maxiter=30, num_cycles=4),
                monitor_lists=plan.ScipyOptimizerMonitorList(
                    callback_monitors=overlaps_disc,
                    start_monitors=overlaps_disc + sims_disc + eps_disc,
                    end_monitors=overlaps_disc + sims_disc + eps_disc)),
        ),
    ]

    return plan.OptimizationPlan(transformations=transformations)


def test_dump_and_load():
    plan = generate_wdm_2d()
    serialized_plan = optplan.dumps(plan)
    deserialized_plan = optplan.loads(serialized_plan)


def test_generate_name():
    optplan.reset_graph()

    assert optplan.generate_name("testtype") == "testtype.0"
    assert optplan.generate_name("testtype") == "testtype.1"
    assert optplan.generate_name("testtype2") == "testtype2.0"


class ModelA(optplan.ProblemGraphNode):
    type = types.StringType(default="ModelA")
    int_field = types.IntType()
    string_field = types.StringType()
    ref_field = optplan.ReferenceType(optplan.ProblemGraphNode)
    ref_field2 = optplan.ReferenceType(optplan.ProblemGraphNode)


class ModelB(optplan.ProblemGraphNode):
    type = types.StringType(default="ModelB")
    int_field = types.IntType()


def test_autoname():
    modela = ModelA()
    modelb = ModelB()
    modela.ref_field = modelb
    modela.ref_field2 = "user_set_name"

    optplan.autoname(modela)

    assert modela.name
    assert modelb.name
    assert modela.ref_field.name == modelb.name
    assert modela.ref_field2 == "user_set_name"


def test_autoname_dups():
    """Tests `autoname` where the same model shows up twice in the graph."""
    modela = ModelA()
    modelb = ModelB()
    modela.ref_field = modelb
    modela.ref_field2 = modelb

    optplan.autoname(modela)

    assert modela.name
    assert modelb.name
    assert modela.ref_field.name == modelb.name
    assert modela.ref_field2.name == modelb.name


def test_autoname_lists():
    """Tests that autoname works with lists."""

    class Model(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = types.ListType(types.ModelType(ModelB))

    modelb1 = ModelB(int_field=1)
    modelb2 = ModelB(int_field=2)
    model = Model(value=[modelb1, modelb2])

    optplan.autoname(model)

    assert model.name
    assert modelb1.name
    assert modelb2.name


def test_autoname_dicts():
    """Tests that autoname works with lists."""

    class Model(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = types.DictType(types.ModelType(ModelB))

    modelb1 = ModelB(int_field=1)
    modelb2 = ModelB(int_field=2)
    model = Model(value={"1": modelb1, "2": modelb2})

    optplan.autoname(model)

    assert model.name
    assert modelb1.name
    assert modelb2.name


def test_autoname_nested():

    class OuterModel(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = optplan.ReferenceType(optplan.ProblemGraphNode)

    class InnerModel(optplan.ProblemGraphNode):
        type = types.StringType(default="Model2")
        value = optplan.ReferenceType(optplan.ProblemGraphNode)

    modelb = ModelB()
    inner_model = InnerModel(value=modelb)
    outer_model = OuterModel(value=inner_model)

    optplan.autoname(outer_model)

    assert outer_model.name
    assert inner_model.name
    assert modelb.name


def test_extract_nodes():
    modela = ModelA()
    modelb = ModelB()
    modela.ref_field = modelb
    modela.ref_field2 = "user_set_name"

    model_list = []
    io._extract_nodes(modela, model_list)

    assert len(model_list) == 2
    assert modela in model_list
    assert modelb in model_list


def test_extract_nodes_dups():
    """Tests where the same model shows up twice in the graph."""
    modela = ModelA()
    modelb = ModelB()
    modela.ref_field = modelb
    modela.ref_field2 = modelb

    model_list = []
    io._extract_nodes(modela, model_list)

    assert len(model_list) == 2
    assert modela in model_list
    assert modelb in model_list


def test_extract_nodes_lists():
    """Tests that it works with lists."""

    class Model(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = types.ListType(types.ModelType(ModelB))

    modelb1 = ModelB(int_field=1)
    modelb2 = ModelB(int_field=2)
    model = Model(value=[modelb1, modelb2])

    model_list = []
    io._extract_nodes(model, model_list)

    assert len(model_list) == 3
    assert model in model_list
    assert modelb1 in model_list
    assert modelb2 in model_list


def test_extract_nodes_dicts():
    """Tests that autoname works with lists."""

    class Model(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = types.DictType(types.ModelType(ModelB))

    modelb1 = ModelB(int_field=1)
    modelb2 = ModelB(int_field=2)
    model = Model(value={"1": modelb1, "2": modelb2})

    model_list = []
    io._extract_nodes(model, model_list)

    assert len(model_list) == 3
    assert model in model_list
    assert modelb1 in model_list
    assert modelb2 in model_list


def test_extract_nodes_nested():

    class OuterModel(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = optplan.ReferenceType(optplan.ProblemGraphNode)

    class InnerModel(optplan.ProblemGraphNode):
        type = types.StringType(default="Model2")
        value = optplan.ReferenceType(optplan.ProblemGraphNode)

    modelb = ModelB()
    inner_model = InnerModel(value=modelb)
    outer_model = OuterModel(value=inner_model)

    model_list = []
    io._extract_nodes(outer_model, model_list)

    assert len(model_list) == 3
    assert modelb in model_list
    assert inner_model in model_list
    assert outer_model in model_list


def test_replace_ref_nodes_with_names():
    modelb = ModelB()
    modelb.name = "modelbname"

    modela = ModelA()
    modela.int_field = 2
    modela.ref_field = modelb
    modela.ref_field2 = "user_set_name"

    model_list = [modelb, modela]
    io._replace_ref_nodes_with_names(modela, model_list)

    assert modela.ref_field == "modelbname"
    assert modela.ref_field2 == "user_set_name"


def test_replace_ref_nodes_with_names_lists():
    """Tests that it works with lists."""

    class Model(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = types.ListType(optplan.ReferenceType(optplan.ProblemGraphNode))

    modelb1 = ModelB(name="m1", int_field=1)
    modelb2 = ModelB(name="m2", int_field=2)
    model = Model(name="m3", value=[modelb1, modelb2])

    model_list = [modelb1, modelb2, model]
    io._replace_ref_nodes_with_names(model, model_list)

    assert model.value == [modelb1.name, modelb2.name]


def test_replace_ref_nodes_with_names_dicts():
    """Tests that it works with lists."""

    class Model(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = types.DictType(optplan.ReferenceType(optplan.ProblemGraphNode))

    modelb1 = ModelB(name="m1", int_field=1)
    modelb2 = ModelB(name="m2", int_field=2)
    model = Model(name="m3", value={"1": modelb1, "2": modelb2})

    model_list = [modelb1, modelb2, model]
    io._replace_ref_nodes_with_names(model, model_list)

    assert model.value == {"1": modelb1.name, "2": modelb2.name}


def test_replace_ref_nodes_with_names_nested():

    class OuterModel(optplan.ProblemGraphNode):
        type = types.StringType(default="Model")
        value = optplan.ReferenceType(optplan.ProblemGraphNode)

    class InnerModel(optplan.ProblemGraphNode):
        type = types.StringType(default="Model2")
        value = optplan.ReferenceType(optplan.ProblemGraphNode)

    modelb = ModelB(name="m1")
    inner_model = InnerModel(name="m2", value=modelb)
    outer_model = OuterModel(name="m3", value=inner_model)

    model_list = [outer_model, inner_model, modelb]
    io._replace_ref_nodes_with_names(outer_model, model_list)

    assert outer_model.value == inner_model.name
    assert inner_model.value == modelb.name


def test_dumps():
    plan = optplan.OptimizationPlan(
        nodes=[
            optplan.Sum(
                functions=[
                    optplan.Constant(value=optplan.ComplexNumber(real=2)),
                    optplan.Constant(value=optplan.ComplexNumber(real=3)),
                ],)
        ],)

    plan_dict = json.loads(optplan.dumps(plan))


def test_dumps_duplicate_name_raises_value_error():
    plan = optplan.OptimizationPlan(
        nodes=[
            optplan.Sum(
                functions=[
                    optplan.Constant(
                        name="const", value=optplan.ComplexNumber(real=2)),
                    optplan.Constant(
                        name="const", value=optplan.ComplexNumber(real=3)),
                ],)
        ],)

    with pytest.raises(ValueError, match="Nonunique name found"):
        optplan.dumps(plan)


def test_custom_function_node():

    @optplan.register_node_type(optplan.NodeMetaType.OPTPLAN_NODE)
    class CustomOp(optplan.Function):
        type = schema_utils.polymorphic_model_type("custom_op")
        int_val = types.IntType()

    plan = optplan.OptimizationPlan(
        nodes=[
            optplan.Sum(
                functions=[
                    optplan.Constant(value=optplan.ComplexNumber(real=2)),
                    CustomOp(int_val=3),
                ],)
        ],)

    optplan.loads(optplan.dumps(plan))


def test_problem_node_bad_name_raises_value_error():
    with pytest.raises(ValueError, match="cannot start with two underscores"):
        optplan.Sum(name="__doubleunderscore")


def test_problem_node_with_bad_reference_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        optplan.Power(function=optplan.SimulationSpace())


def test_validate_references_values_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        optplan.validate_references(
            optplan.Power(function=optplan.SimulationSpace()))


def test_validate_references_lists_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        optplan.validate_references(
            optplan.Sum(functions=[optplan.SimulationSpace()]))


def test_validate_references_nested_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        optplan.validate_references(
            optplan.Sum(functions=[
                optplan.Power(
                    function=optplan.Sum(functions=[
                        optplan.make_constant(2),
                        optplan.SimulationSpace()
                    ])),
                optplan.make_constant(2)
            ]))
