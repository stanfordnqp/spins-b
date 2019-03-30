"""Tests utility functions for schematics."""
import copy

from schematics import models
from schematics import types

from spins.invdes.problem_graph import schema_utils


class ModelB(models.Model, schema_utils.CopyModelMixin):
    value = types.IntType()


class ModelA(models.Model, schema_utils.CopyModelMixin):
    int_field = types.IntType()
    list_int_field = types.ListType(types.IntType())
    model_field = types.ModelType(ModelB)
    list_model_field = types.ListType(types.ModelType(ModelB))


def test_copy():
    modela = ModelA({
        "int_field": 1,
        "list_int_field": [2, 3],
        "model_field": {
            "value": 4
        },
        "list_model_field": [{
            "value": 5
        }, {
            "value": 6
        }],
    })

    modela_copy = copy.copy(modela)

    # Check equality of values.
    assert modela_copy.int_field == 1
    assert modela_copy.list_int_field == [2, 3]
    assert modela_copy.model_field.value == 4
    assert modela_copy.list_model_field[0].value == 5
    assert modela_copy.list_model_field[1].value == 6

    # Now check for identity equality.
    assert modela_copy.list_int_field is modela.list_int_field
    assert modela_copy.model_field is modela.model_field
    assert modela_copy.list_model_field is modela.list_model_field
    assert modela_copy.list_model_field[0] is modela.list_model_field[0]
    assert modela_copy.list_model_field[1] is modela.list_model_field[1]

    # Check that modifying object values result in state in change in the
    # other model by modifying primitive values does not.
    modela.model_field.value = 7
    assert modela_copy.model_field.value == 7

    modela.int_field = 8
    assert modela_copy.int_field == 1


def test_deepcopy():
    modela = ModelA({
        "int_field": 1,
        "list_int_field": [2, 3],
        "model_field": {
            "value": 4
        },
        "list_model_field": [{
            "value": 5
        }, {
            "value": 6
        }],
    })

    modela_copy = copy.deepcopy(modela)

    # Check equality of values.
    assert modela_copy.int_field == 1
    assert modela_copy.list_int_field == [2, 3]
    assert modela_copy.model_field.value == 4
    assert modela_copy.list_model_field[0].value == 5
    assert modela_copy.list_model_field[1].value == 6

    # Now check for inequality of objects.
    assert modela_copy.list_int_field is not modela.list_int_field
    assert modela_copy.model_field is not modela.model_field
    assert modela_copy.list_model_field is not modela.list_model_field
    assert modela_copy.list_model_field[0] is not modela.list_model_field[0]
    assert modela_copy.list_model_field[1] is not modela.list_model_field[1]

    # To be extra sure, modify one of the values and check that it does not
    # affect the other object.
    modela.model_field.value = 7
    assert modela_copy.model_field.value == 4


class ModelC(schema_utils.KwargsModelMixin, models.Model):
    int_field = types.IntType()
    string_field = types.StringType()


def test_kwargs():
    modelc = ModelC(int_field=3, string_field="hi")
    assert modelc.int_field == 3
    assert modelc.string_field == "hi"


def test_kwargs_dict():
    modelc = ModelC({"int_field": 3, "string_field": "hi"})
    assert modelc.int_field == 3
    assert modelc.string_field == "hi"


def test_kwargs_validate():
    modelc = ModelC({"int_field": 3, "string_field": "hi"})
    modelc.validate()
