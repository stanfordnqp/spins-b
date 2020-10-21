from typing import List, Union

import json

import numpy as np
import pytest
from schematics import models
from schematics import types

from spins import goos
from spins.goos import optplan
from spins.goos.optplan import schema


def test_construct_schema_1positional():

    def fun(x: int):
        pass

    model_type = schema.construct_schema("FunSchema", fun, skip_first_arg=False)
    assert model_type({"x": 5}).to_native() == {"x": 5}
    with pytest.raises(models.DataError):
        model_type({"x": "hi"}).to_native() == {"x": "hi"}


def test_construct_schema_2positional():

    def fun(x: int, y: float):
        pass

    model_type = schema.construct_schema("FunSchema", fun, skip_first_arg=False)
    assert model_type({"x": 5, "y": 2}).to_native() == {"x": 5, "y": 2}


def test_construct_schema_default():

    def fun(x: int = 3):
        pass

    model_type = schema.construct_schema("FunSchema", fun, skip_first_arg=False)
    assert model_type().to_native() == {"x": 3}
    assert model_type({"x": 5}).to_native() == {"x": 5}


def test_construct_schema_list():

    def fun(x: List[int]):
        pass

    model_type = schema.construct_schema("FunSchema", fun)
    assert model_type({"x": [1, 2, 3]}).to_native() == {"x": [1, 2, 3]}


def test_construct_schema_union():

    def fun(x: Union[int, str]):
        pass

    model_type = schema.construct_schema("FunSchema", fun)
    assert model_type({"x": 1}).to_native() == {"x": 1}
    assert model_type({"x": "hi"}).to_native() == {"x": "hi"}


def test_construct_schema_nested():

    def fun(x: Union[int, List[Union[int, str]]]):
        pass

    model_type = schema.construct_schema("FunSchema", fun)
    assert model_type({"x": 1}).to_native() == {"x": 1}
    assert model_type({"x": ["hi"]}).to_native() == {"x": ["hi"]}
    #TODO(@jskarda,@jesse): Determine why this commented assert fails
    #   when gitlab runs all tests:
    #assert model_type({"x": [2, "hi"]}).to_native() == {"x": [2, "hi"]}
    with pytest.raises(models.DataError):
        model_type({"x": "hi"}).to_native() == {"x": "hi"}


def test_construct_schema_complex():

    def fun(x: complex):
        pass

    model_type = schema.construct_schema("FunSchema", fun)
    assert model_type({"x": 1 + 2j}).to_native() == {"x": 1 + 2j}


def test_construct_schema_numpy():

    def fun(x: np.ndarray):
        pass

    model_type = schema.construct_schema("FunSchema", fun)
    model_native = model_type({"x": np.array([1, 2, 3])}).to_native()
    assert list(model_native.keys()) == ["x"]
    np.testing.assert_array_equal(model_native["x"], [1, 2, 3])


def test_construct_schema_schematics_model():

    @optplan.polymorphic_model()
    class Model(optplan.Model):
        type = optplan.ModelNameType("model")
        y = optplan.types.IntType()

    def fun(x: Model):
        pass

    model_type = schema.construct_schema("FunSchema", fun)
    assert model_type({
        "x": {
            "type": "model",
            "y": 3
        }
    }).to_native() == {
        "x": {
            "type": "model",
            "y": 3
        }
    }


def test_construct_schema_optplan_base_model():

    def fun(x: int):
        pass

    model_type = schema.construct_schema(
        "FunSchema", fun, base_classes=(optplan.Model,))
    data = model_type(x=3)
    data.validate()
    assert data.to_native() == {"x": 3}


class ModelA(optplan.ProblemGraphNode.Schema):
    type = types.StringType(default="ModelA")
    int_field = types.IntType()
    string_field = types.StringType()
    ref_field = optplan.ReferenceType(optplan.ProblemGraphNode.Schema)
    ref_field2 = optplan.ReferenceType(optplan.ProblemGraphNode.Schema)


class ModelB(optplan.ProblemGraphNode.Schema):
    type = types.StringType(default="ModelB")
    int_field = types.IntType()


class CompoundModel(optplan.ProblemGraphNode.Schema):
    type = optplan.ModelNameType("compound_model")
    functions = optplan.types.ListType(optplan.ReferenceType(ModelB))


class CompoundModel2(optplan.ProblemGraphNode.Schema):
    type = optplan.ModelNameType("compound_model2")
    function = optplan.ReferenceType(ModelB)


def test_extract_nodes():
    modela = ModelA()
    modelb = ModelB()
    modela.ref_field = modelb
    modela.ref_field2 = "user_set_name"

    model_list = []
    schema._extract_nodes(modela, model_list)

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
    schema._extract_nodes(modela, model_list)

    assert len(model_list) == 2
    assert modela in model_list
    assert modelb in model_list


def test_extract_nodes_lists():
    """Tests that it works with lists."""

    class Model(optplan.ProblemGraphNode.Schema):
        type = types.StringType(default="Model")
        value = types.ListType(types.ModelType(ModelB))

    modelb1 = ModelB(int_field=1)
    modelb2 = ModelB(int_field=2)
    model = Model(value=[modelb1, modelb2])

    model_list = []
    schema._extract_nodes(model, model_list)

    assert len(model_list) == 3
    assert model in model_list
    assert modelb1 in model_list
    assert modelb2 in model_list


def test_extract_nodes_dicts():
    """Tests that autoname works with lists."""

    class Model(optplan.ProblemGraphNode.Schema):
        type = types.StringType(default="Model")
        value = types.DictType(types.ModelType(ModelB))

    modelb1 = ModelB(int_field=1)
    modelb2 = ModelB(int_field=2)
    model = Model(value={"1": modelb1, "2": modelb2})

    model_list = []
    schema._extract_nodes(model, model_list)

    assert len(model_list) == 3
    assert model in model_list
    assert modelb1 in model_list
    assert modelb2 in model_list


def test_extract_nodes_nested():

    class OuterModel(optplan.ProblemGraphNode.Schema):
        type = types.StringType(default="Model")
        value = optplan.ReferenceType(optplan.ProblemGraphNode.Schema)

    class InnerModel(optplan.ProblemGraphNodeSchema):
        type = types.StringType(default="Model2")
        value = optplan.ReferenceType(optplan.ProblemGraphNode.Schema)

    modelb = ModelB()
    inner_model = InnerModel(value=modelb)
    outer_model = OuterModel(value=inner_model)

    model_list = []
    schema._extract_nodes(outer_model, model_list)

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
    schema._replace_ref_nodes_with_names(modela, model_list)

    assert modela.ref_field == "modelbname"
    assert modela.ref_field2 == "user_set_name"


def test_replace_ref_nodes_with_names_lists():
    """Tests that it works with lists."""

    class Model(optplan.ProblemGraphNodeSchema):
        type = types.StringType(default="Model")
        value = types.ListType(
            optplan.ReferenceType(optplan.ProblemGraphNodeSchema))

    modelb1 = ModelB(name="m1", int_field=1)
    modelb2 = ModelB(name="m2", int_field=2)
    model = Model(name="m3", value=[modelb1, modelb2])

    model_list = [modelb1, modelb2, model]
    schema._replace_ref_nodes_with_names(model, model_list)

    assert model.value == [modelb1.name, modelb2.name]


def test_replace_ref_nodes_with_names_dicts():
    """Tests that it works with lists."""

    class Model(optplan.ProblemGraphNodeSchema):
        type = types.StringType(default="Model")
        value = types.DictType(
            optplan.ReferenceType(optplan.ProblemGraphNodeSchema))

    modelb1 = ModelB(name="m1", int_field=1)
    modelb2 = ModelB(name="m2", int_field=2)
    model = Model(name="m3", value={"1": modelb1, "2": modelb2})

    model_list = [modelb1, modelb2, model]
    schema._replace_ref_nodes_with_names(model, model_list)

    assert model.value == {"1": modelb1.name, "2": modelb2.name}


def test_replace_ref_nodes_with_names_nested():

    class OuterModel(optplan.ProblemGraphNode.Schema):
        type = types.StringType(default="Model")
        value = optplan.ReferenceType(optplan.ProblemGraphNode.Schema)

    class InnerModel(optplan.ProblemGraphNode.Schema):
        type = types.StringType(default="Model2")
        value = optplan.ReferenceType(optplan.ProblemGraphNode.Schema)

    modelb = ModelB(name="m1")
    inner_model = InnerModel(name="m2", value=modelb)
    outer_model = OuterModel(name="m3", value=inner_model)

    model_list = [outer_model, inner_model, modelb]
    schema._replace_ref_nodes_with_names(outer_model, model_list)

    assert outer_model.value == inner_model.name
    assert inner_model.value == modelb.name


def test_dumps():
    plan = optplan.OptimizationPlanSchema(
        nodes=[
            goos.Sum.Schema(
                functions=[
                    goos.Constant.Schema(value=2),
                    goos.Constant.Schema(value=3),
                ],)
        ],)

    plan_dict = json.loads(schema.dumps(plan))


def test_dumps_duplicate_name_raises_value_error():
    plan = optplan.OptimizationPlanSchema(
        nodes=[
            goos.Sum.Schema(
                functions=[
                    goos.Constant.Schema(name="const", value=2),
                    goos.Constant.Schema(name="const", value=3),
                ],)
        ],)

    with pytest.raises(ValueError, match="Nonunique name found"):
        schema.dumps(plan)


def test_validate_references_values_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        schema.validate_references(CompoundModel2(function=ModelA()))


def test_validate_references_lists_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        schema.validate_references(CompoundModel(functions=[ModelA()]))


def test_validate_references_nested_raises_value_error():
    with pytest.raises(ValueError, match="Expected type"):
        schema.validate_references(
            CompoundModel(functions=[
                CompoundModel2(
                    function=CompoundModel(functions=[
                        ModelB(),
                        ModelA(),
                    ])),
            ]))
