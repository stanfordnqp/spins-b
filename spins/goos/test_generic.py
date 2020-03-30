from typing import List

import os

import numpy as np
import pytest

from spins import goos
from spins.goos import flows
from spins.goos import generic


@pytest.fixture
def temp_context():
    goos.GLOBAL_CONTEXT_STACK.push()
    yield
    goos.GLOBAL_CONTEXT_STACK.pop()


def test_cast(tmpdir, temp_context):
    # Create a class identical to `Constant` but is not a `goos.Function`
    # so we cannot utilize the `+` operator.
    class NewConstant(goos.ProblemGraphNode):
        node_type = "goos.new_constant"

        def __init__(self, value: np.ndarray):
            super().__init__()
            self._value = np.array(value)

        def eval(self, inputs: List[flows.NumericFlow]) -> flows.NumericFlow:
            return flows.NumericFlow(self._value)

        def grad(self, inputs: List[flows.NumericFlow],
                 grad_val: flows.NumericFlow) -> List[flows.NumericFlow]:
            return [flows.NumericFlow(np.zeros_like(self._value))]

    # Path to save optplan.
    plan_path = str(tmpdir)

    with goos.OptimizationPlan() as plan:
        node = NewConstant(3)

        # The following should fail because `NewConstant` is not a `Function`.
        with pytest.raises(TypeError):
            node + 6

        numeric_node = generic.cast(node, goos.Function) + 6
        numeric_node = goos.Sum([numeric_node], name="result")
        assert numeric_node.get() == 9

        # Check that we can save and load.
        plan.save(plan_path)

    with goos.OptimizationPlan() as plan:
        plan.load(plan_path)
        numeric_node = plan.get_node("result")
        assert numeric_node.get() == 9


def test_rename(temp_context):

    class NewConstant(goos.Function):
        node_type = "goos.new_constant"

        def __init__(self, value: np.ndarray):
            super().__init__()
            self._value = np.array(value)

        def eval(self, inputs: List[flows.NumericFlow]) -> flows.NumericFlow:
            return flows.NumericFlow(self._value)

        def grad(self, inputs: List[flows.NumericFlow],
                 grad_val: flows.NumericFlow) -> List[flows.NumericFlow]:
            return [flows.NumericFlow(np.zeros_like(self._value))]

    with goos.OptimizationPlan() as plan:
        orig_node = NewConstant(3, name="old_node")
        new_node = goos.rename(orig_node, "new_node")

        # Make sure that the name is correct.
        assert new_node._goos_name == "new_node"
        # Ensure that the node type is correct.
        assert isinstance(new_node, NewConstant)
        # Ensure that basic operations hold.
        assert (new_node + 4).get() == 7
