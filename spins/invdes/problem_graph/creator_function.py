from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spins.invdes import problem
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


@optplan.register_node(optplan.Parameter)
def create_parameter(params: optplan.Parameter,
                     work: workspace.Workspace) -> problem.Parameter:
    return problem.Parameter(initial_value=params.initial_value)


@optplan.register_node(optplan.Sum)
def create_sum(params: optplan.Sum, work: workspace.Workspace) -> problem.Sum:
    fun_list = [work.get_object(fun) for fun in params.functions]
    return problem.Sum(objectives=fun_list)


@optplan.register_node(optplan.Abs)
def create_abs(params: optplan.Abs,
               work: workspace.Workspace) -> problem.AbsoluteValue:
    return problem.AbsoluteValue(objective=work.get_object(params.function))


@optplan.register_node(optplan.Power)
def create_power(params: optplan.Power,
                 work: workspace.Workspace) -> problem.Power:
    # TODO(logansu): For consistency, `obj` should be `objective`.
    return problem.Power(obj=work.get_object(params.function), power=params.exp)


@optplan.register_node(optplan.PowerComp)
def create_power_comparison(params: optplan.PowerComp, work: workspace.Workspace
                           ) -> problem.PowerComparison:
    value_range = params.value + np.array([-0.5, 0.5]) * params.range
    return problem.PowerComparison(
        objective=work.get_object(params.function),
        value_range=value_range,
        power=params.exp)


@optplan.register_node(optplan.Product)
def create_product(params: optplan.Product,
                   work: workspace.Workspace) -> problem.Product:
    fun_list = [work.get_object(fun) for fun in params.functions]
    return problem.Product(objectives=fun_list)


@optplan.register_node(optplan.Constant)
def create_constant(params: optplan.Constant,
                    work: workspace.Workspace) -> problem.Constant:
    return problem.Constant(value=params.value.real + 1j * params.value.imag)
