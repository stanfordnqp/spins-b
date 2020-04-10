import numpy as np

from spins.goos import flows


def test_const_flags():
    """Test that const flag autogeneration is correct."""

    class NewFlow(flows.Flow):
        pos: int = 0
        rot: float = 0
        priority: int = flows.constant_field(default=0)

    flags = NewFlow.ConstFlags(pos=False, rot=True)

    assert flags.pos == False
    assert flags.rot == True
    assert not hasattr(flags, "priority")


def test_set_all_const_flags():

    class NewFlow(flows.Flow):
        pos: int = 0
        rot: float = 0
        priority: int = flows.constant_field(default=0)

    flags = NewFlow.ConstFlags()
    flags.set_all(True)

    assert flags.pos == True
    assert flags.rot == True
    assert flags


def test_const_flags_bool():

    class NewFlow(flows.Flow):
        pos: int = 0
        rot: float = 0
        priority: int = flows.constant_field(default=0)

    flags = NewFlow.ConstFlags()

    flags.pos = False
    flags.rot = True

    assert not flags


def test_grad_generation():

    class NewFlow(flows.Flow):
        pos: int = 0
        rot: float = 0
        priority: int = flows.constant_field(default=0)

    grad = NewFlow.Grad()

    assert grad.pos_grad == 0
    assert grad.rot_grad == 0
    assert not hasattr(grad, "priority_grad")

    grad += NewFlow.Grad(pos_grad=1)

    assert grad.pos_grad == 1
    assert grad.rot_grad == 0


def test_flow_equality_identity():

    class NewFlow(flows.Flow):
        pos: np.ndarray = flows.np_zero_field(3)
        priority: int = flows.constant_field(default=0)

    flow = NewFlow(pos=np.array([3, 1, 2]), priority=3)

    assert flow == flow


def test_flow_equality():

    class NewFlow(flows.Flow):
        pos: np.ndarray = flows.np_zero_field(3)
        priority: int = flows.constant_field(default=0)

    flow = NewFlow(pos=np.array([3, 1, 2]), priority=3)
    flow2 = NewFlow(pos=np.array([3, 1, 2]), priority=3)

    assert flow == flow2


def test_flow_inequality():

    class NewFlow(flows.Flow):
        pos: np.ndarray = flows.np_zero_field(3)
        priority: int = flows.constant_field(default=0)

    flow = NewFlow(pos=np.array([3, 1, 2]))
    flow2 = NewFlow(pos=np.array([3, 3, 2]))

    assert flow != flow2


def test_flow_inheritance_equality():

    class NewFlow(flows.Flow):
        pos: np.ndarray = flows.np_zero_field(3)
        priority: int = flows.constant_field(default=0)

    class NewFlowSuper(NewFlow):
        rot: np.ndarray = flows.np_zero_field(3)

    flow = NewFlowSuper(pos=np.array([3, 1, 2]), rot=np.array([1, 1, 1]))
    flow2 = NewFlowSuper(pos=np.array([3, 1, 2]), rot=np.array([1, 1, 1]))

    assert flow == flow2


def test_flow_inheritance_inequality():

    class NewFlow(flows.Flow):
        pos: np.ndarray = flows.np_zero_field(3)
        priority: int = flows.constant_field(default=0)

    class NewFlowSuper(NewFlow):
        rot: np.ndarray = flows.np_zero_field(3)

    flow = NewFlowSuper(pos=np.array([3, 1, 2]), rot=np.array([1, 1, 1]))
    flow2 = NewFlowSuper(pos=np.array([3, 3, 2]), rot=np.array([1, 1, 1]))

    assert flow != flow2
