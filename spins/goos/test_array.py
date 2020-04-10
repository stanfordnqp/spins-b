from typing import List

import pytest

from spins import goos


@pytest.fixture
def temp_context():
    goos.GLOBAL_CONTEXT_STACK.push()
    yield
    goos.GLOBAL_CONTEXT_STACK.pop()


def test_array_flow_op(temp_context):

    class ArrayFlowOp(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
        node_type = "array_flow_op"

        def __init__(self, nums: List[int]) -> None:
            super().__init__(flow_types=[goos.Function] * len(nums))
            self._nums = nums

        def eval(self, inputs):
            agg_flow = [goos.NumericFlow(x) for x in self._nums]
            return goos.ArrayFlow(agg_flow)

    with goos.OptimizationPlan() as plan:
        op = ArrayFlowOp([3, 4])

        assert op[0].get() == 3
        assert (op[1] + 2).get() == 6


def test_array_flow_op_naming(temp_context):

    class ArrayFlowOp(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
        node_type = "array_flow_op"

        def __init__(self, nums: List[int], names: List[str]) -> None:
            super().__init__(flow_types=[goos.Function] * len(nums),
                             flow_names=names)
            self._nums = nums

        def eval(self, inputs):
            agg_flow = [goos.NumericFlow(x) for x in self._nums]
            return goos.ArrayFlow(agg_flow)

    with goos.OptimizationPlan() as plan:
        op = ArrayFlowOp([3, 4, 5], names=["first", None, "third"], name="op")

        assert op["first"].get() == 3
        assert (op[1] + 2).get() == 6

        assert op[0]._goos_name == "op.first"
        assert op[2]._goos_name == "op.third"


def test_array_flow_op_raises_duplicate_flow_name(temp_context):

    class ArrayFlowOp(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
        node_type = "array_flow_op"

        def __init__(self, nums: List[int], names: List[str]) -> None:
            super().__init__(flow_types=[goos.Function] * len(nums),
                             flow_names=names)
            self._nums = nums

        def eval(self, inputs):
            agg_flow = [goos.NumericFlow(x) for x in self._nums]
            return goos.ArrayFlow(agg_flow)

    with pytest.raises(ValueError, match="Duplicate flow name"):
        with goos.OptimizationPlan() as plan:
            op = ArrayFlowOp([3, 4, 5], names=["first", "second", "first"])


def test_array_flow_const_flags_bool():
    flags = goos.ArrayFlow.ConstFlags(
        [goos.NumericFlow.ConstFlags(),
         goos.ShapeFlow.ConstFlags()])

    assert not flags

    flags.flow_flags[0].array = True
    assert not flags

    flags.flow_flags[1].set_all(True)
    assert flags


def test_array_flow_const_flags_set_all():
    flags = goos.ArrayFlow.ConstFlags(
        [goos.NumericFlow.ConstFlags(),
         goos.ShapeFlow.ConstFlags()])

    flags.set_all(True)
    assert flags
    assert flags.flow_flags[0]
    assert flags.flow_flags[1]


def test_array_flow_grad():
    grad1 = goos.ArrayFlow.Grad(
        [goos.NumericFlow.Grad(2),
         goos.NumericFlow.Grad(3)])
    grad2 = goos.ArrayFlow.Grad(
        [goos.NumericFlow.Grad(9),
         goos.NumericFlow.Grad(1)])

    grad1 += grad2

    assert grad1.flows_grad == [
        goos.NumericFlow.Grad(11),
        goos.NumericFlow.Grad(4)
    ]


def test_array_flow_equality():
    flow = goos.ArrayFlow([goos.NumericFlow(3), goos.NumericFlow(2)])
    flow2 = goos.ArrayFlow([goos.NumericFlow(3), goos.NumericFlow(2)])

    assert flow == flow2


def test_array_flow_inequality():
    flow = goos.ArrayFlow([goos.NumericFlow(3), goos.NumericFlow(2)])
    flow2 = goos.ArrayFlow([goos.NumericFlow(3), goos.NumericFlow(1)])

    assert flow != flow2
