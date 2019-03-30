import pytest
from schematics import models

from spins.invdes.problem_graph import optplan


class DummyModel(models.Model):
    pass


class DummyModel2(models.Model):
    pass


class DummyModel3(models.Model):
    pass


def dummy_creator():
    pass


def dummy_creator2():
    pass


def test_optplan_context_register_node():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "testnode", DummyModel, dummy_creator)

    assert ctx.get_node_model("testmeta", "testnode") == DummyModel
    assert ctx.get_node_creator("testmeta", "testnode") == dummy_creator


def test_optplan_context_register_node_multiple_same_meta():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, dummy_creator)
    ctx.register_node_type("testmeta", "dummy2", DummyModel2, dummy_creator2)

    assert ctx.get_node_model("testmeta", "dummy") == DummyModel
    assert ctx.get_node_model("testmeta", "dummy2") == DummyModel2

    assert ctx.get_node_creator("testmeta", "dummy") == dummy_creator
    assert ctx.get_node_creator("testmeta", "dummy2") == dummy_creator2


def test_optplan_context_register_node_multiple_different_meta():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, dummy_creator)
    ctx.register_node_type("testmeta2", "dummy", DummyModel2, dummy_creator2)

    assert ctx.get_node_model("testmeta", "dummy") == DummyModel
    assert ctx.get_node_model("testmeta2", "dummy") == DummyModel2

    assert ctx.get_node_creator("testmeta", "dummy") == dummy_creator
    assert ctx.get_node_creator("testmeta2", "dummy") == dummy_creator2


def test_optplan_context_register_node_same_name_throws_error():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, None)

    with pytest.raises(ValueError, match="registered twice"):
        ctx.register_node_type("testmeta", "dummy", DummyModel2, None)


def test_optplan_context_get_nonexistent_node():
    ctx = optplan.OptplanContext()

    assert ctx.get_node_model("testmeta", "dummy") == None


def test_optplan_context_get_node_model_dict():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, None)
    ctx.register_node_type("testmeta", "dummy2", DummyModel2, None)
    ctx.register_node_type("testmeta2", "dummy3", DummyModel3, None)

    assert ctx.get_node_model_dict("testmeta") == {
        "dummy": DummyModel,
        "dummy2": DummyModel2
    }


def test_optplan_context_get_node_model_dict_empty():
    ctx = optplan.OptplanContext()

    assert ctx.get_node_model_dict("testmeta") == {}


def test_optplan_context_stack_get_node_model():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, dummy_creator)

    ctx_stack = optplan.OptplanContextStack()
    ctx_stack.push(ctx)

    assert ctx_stack.get_node_model("testmeta", "dummy") == DummyModel
    assert ctx_stack.get_node_creator("testmeta", "dummy") == dummy_creator
    assert ctx_stack.get_node_model("testmeta", "dummy2") == None
    assert ctx_stack.get_node_creator("testmeta", "dummy2") == None


def test_optplan_context_stack_get_node_model_multiple():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, dummy_creator)

    ctx2 = optplan.OptplanContext()
    ctx2.register_node_type("testmeta", "dummy2", DummyModel2, dummy_creator2)

    ctx_stack = optplan.OptplanContextStack()
    ctx_stack.push(ctx)
    ctx_stack.push(ctx2)

    assert ctx_stack.get_node_model("testmeta", "dummy") == DummyModel
    assert ctx_stack.get_node_model("testmeta", "dummy2") == DummyModel2
    assert ctx_stack.get_node_model("testmeta", "dummy3") == None

    assert ctx_stack.get_node_creator("testmeta", "dummy") == dummy_creator
    assert ctx_stack.get_node_creator("testmeta", "dummy2") == dummy_creator2
    assert ctx_stack.get_node_creator("testmeta", "dummy3") == None


def test_optplan_context_stack_get_node_model_overwriting():
    ctx = optplan.OptplanContext()
    ctx.register_node_type("testmeta", "dummy", DummyModel, dummy_creator)

    ctx2 = optplan.OptplanContext()
    ctx2.register_node_type("testmeta", "dummy", DummyModel2, dummy_creator2)

    ctx_stack = optplan.OptplanContextStack()
    ctx_stack.push(ctx)
    ctx_stack.push(ctx2)

    assert ctx_stack.get_node_model("testmeta", "dummy") == DummyModel2
    assert ctx_stack.get_node_creator("testmeta", "dummy") == dummy_creator2


def test_optplan_context_stack_push_and_pop():
    ctx = optplan.OptplanContext()
    ctx2 = optplan.OptplanContext()
    ctx_stack = optplan.OptplanContextStack()

    assert ctx_stack.peek() == None

    ctx_stack.push(ctx)
    assert ctx_stack.peek() == ctx

    ctx_stack.push(ctx2)

    assert ctx_stack.peek() == ctx2
    assert ctx_stack.pop() == ctx2

    assert ctx_stack.peek() == ctx
    assert ctx_stack.pop() == ctx

    assert ctx_stack.peek() == None
    assert ctx_stack.pop() == None
