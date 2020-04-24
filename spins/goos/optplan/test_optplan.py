import pytest

import numpy as np

from spins import goos


@pytest.fixture
def temp_context():
    goos.GLOBAL_CONTEXT_STACK.push()
    yield
    goos.GLOBAL_CONTEXT_STACK.pop()


def test_problem_graph_node_basic(temp_context):

    class Node(goos.ProblemGraphNode):
        node_type = "test_node"

        def __init__(self, x: int, y: int, z: int = 6):
            pass

    node = Node(3, 4, z=5, name="test")

    assert node._goos_schema.to_native() == {
        "name": "test",
        "type": "test_node",
        "x": 3,
        "y": 4,
        "z": 5
    }
    assert node._goos_name == "test"

    assert Node(3, 2, name="test2")._goos_schema.to_native() == {
        "name": "test2",
        "type": "test_node",
        "x": 3,
        "y": 2,
        "z": 6
    }
    assert Node(3, y=2, name="test3")._goos_schema.to_native() == {
        "name": "test3",
        "type": "test_node",
        "x": 3,
        "y": 2,
        "z": 6
    }
    assert Node(y=2, x=3, name="test4")._goos_schema.to_native() == {
        "name": "test4",
        "type": "test_node",
        "x": 3,
        "y": 2,
        "z": 6
    }


def test_problem_graph_node_serializable_check(temp_context):

    class Node(goos.ProblemGraphNode):
        node_type = "test_node"

        def __init__(self, x: int):
            pass

    a = Node(3)
    b = Node(Node(4))
    c = Node(None)


def test_plan(tmp_path):
    plan_dir = tmp_path / "test_plan"
    plan_dir.mkdir()
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(3.0, name="x")
        y = goos.Variable(2.0, name="y")
        z = x + y

        assert z.get() == 5
        assert z.get_grad([x, y]) == [1, 1]
        assert (x + x + y + 2).get_grad([x, y]) == [2, 1]
        assert (x**2).get_grad([x]) == [6]

        x.set(4)

        assert z.get() == 5
        assert z.get(run=True) == 6
        assert z.get() == 6

        y.set(x)

        assert z.get(run=True) == 8

        with goos.OptimizationPlan():
            assert z.get() == 5

        goos.opt.scipy_minimize((x + y**2 + 1)**2 + (y + 1)**2, "CG")
        plan.run()
        plan.save(plan_dir)

        np.testing.assert_almost_equal(x.get().array, -2, decimal=4)
        np.testing.assert_almost_equal(y.get().array, -1, decimal=4)

    with goos.OptimizationPlan() as plan:
        plan.load(plan_dir)
        x = plan.get_node("x")
        y = plan.get_node("y")

        assert x.get() == 3
        assert y.get() == 2

        plan.run()

        np.testing.assert_almost_equal(x.get().array, -2, decimal=4)
        np.testing.assert_almost_equal(y.get().array, -1, decimal=4)


def test_plan_heavy(tmp_path):
    # This is the same test as `test_plan` excep that we replace some nodes
    # with heavy node implementations by directly manipulating the flags.
    plan_dir = tmp_path / "test_plan"
    plan_dir.mkdir()
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(3.0, name="x")
        y = goos.Variable(2.0, name="y")
        z = x + y
        z.parallelize()

        assert z.get() == 5
        assert z.get_grad([x, y]) == [1, 1]
        assert (x + x + y + 2).get_grad([x, y]) == [2, 1]
        assert (x**2).get_grad([x]) == [6]

        x.set(4)

        assert z.get() == 5
        assert z.get(run=True) == 6
        assert z.get() == 6

        y.set(x)

        assert z.get(run=True) == 8

        with goos.OptimizationPlan():
            assert z.get() == 5

        first_part = x + y**2 + 1
        first_part.parallelize()

        second_part = y + 1
        second_part.parallelize()

        obj = first_part**2 + second_part**2
        goos.opt.scipy_minimize(obj, "CG")
        plan.run()
        plan.save(plan_dir)

        np.testing.assert_almost_equal(x.get().array, -2, decimal=4)
        np.testing.assert_almost_equal(y.get().array, -1, decimal=4)


def test_flag_eval(temp_context):

    # Create a node that asserts that expected context.
    class AssertFlagNode(goos.ProblemGraphNode):
        node_type = "test_node"

        def __init__(self, node: goos.ProblemGraphNode, expected_context: int):
            super().__init__(node)
            self._context = expected_context

        def eval(self, inputs, context):
            assert self._context == context
            return inputs[0]

        def grad(self, inputs, grad_val, context):
            assert self._context == context
            return grad_val

    with goos.OptimizationPlan() as plan:
        x = goos.Variable(3.0)
        y = x + 2
        z = goos.Constant(4) + goos.Constant(5)

        AssertFlagNode(
            x,
            goos.EvalContext(input_flags=[
                goos.NodeFlags(
                    const_flags=goos.NumericFlow.ConstFlags(False),
                    frozen_flags=goos.NumericFlow.ConstFlags(False),
                ),
            ])).get()
        AssertFlagNode(
            y,
            goos.EvalContext(input_flags=[
                goos.NodeFlags(
                    const_flags=goos.NumericFlow.ConstFlags(False),
                    frozen_flags=goos.NumericFlow.ConstFlags(False),
                ),
            ])).get()
        AssertFlagNode(
            z,
            goos.EvalContext(input_flags=[
                goos.NodeFlags(
                    const_flags=goos.NumericFlow.ConstFlags(True),
                    frozen_flags=goos.NumericFlow.ConstFlags(True),
                ),
            ])).get()

        x.freeze()
        plan.run()

        AssertFlagNode(
            x,
            goos.EvalContext(input_flags=[
                goos.NodeFlags(
                    const_flags=goos.NumericFlow.ConstFlags(False),
                    frozen_flags=goos.NumericFlow.ConstFlags(True),
                ),
            ])).get()
        AssertFlagNode(
            y,
            goos.EvalContext(input_flags=[
                goos.NodeFlags(
                    const_flags=goos.NumericFlow.ConstFlags(False),
                    frozen_flags=goos.NumericFlow.ConstFlags(True),
                ),
            ])).get()


def test_plan_autorun():
    with goos.OptimizationPlan(autorun=True) as plan:
        x = goos.Variable(3)
        x.set(4)
        assert x.get() == 4
