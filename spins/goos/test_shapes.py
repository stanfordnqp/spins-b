import numpy as np

from spins import goos
from spins.goos import material
from spins.goos import shapes


def test_pixelated_cont_shape():

    def init(size):
        return np.ones(size)

    var, shape = shapes.pixelated_cont_shape(
        init, [100, 100, 10], [20, 30, 10],
        var_name="var_name",
        name="shape_name",
        pos=goos.Constant([1, 2, 3]),
        material=material.Material(index=1),
        material2=material.Material(index=2))

    assert var._goos_name == "var_name"
    assert shape._goos_name == "shape_name"


def test_pixelated_cont_shape_flow_get_relative_cell_coords():
    coords = shapes.PixelatedContShapeFlow.get_relative_cell_coords(
        [100, 100, 10], [20, 40, 10])

    np.testing.assert_array_equal(coords[0], [-40, -20, 0, 20, 40])
    np.testing.assert_array_equal(coords[1], [-35, 0, 35])
    assert coords[2] == 0


def test_pixelated_cont_shape_flow_get_relative_cell_coords_decimal():
    factor = 2.25
    coords = shapes.PixelatedContShapeFlow.get_relative_cell_coords(
        np.array([100, 100, 10]) * factor,
        np.array([20, 40, 10]) * factor)

    np.testing.assert_array_equal(coords[0],
                                  np.array([-40, -20, 0, 20, 40]) * factor)
    np.testing.assert_array_equal(coords[1], np.array([-35, 0, 35]) * factor)
    assert coords[2] == 0


def test_pixelated_cont_shape_flow_get_relative_edge_coords():
    coords = shapes.PixelatedContShapeFlow.get_relative_edge_coords(
        [100, 100, 10], [20, 40, 10])

    np.testing.assert_array_equal(coords[0], [-50, -30, -10, 10, 30, 50])
    np.testing.assert_array_equal(coords[1], [-50, -20, 20, 50])
    np.testing.assert_array_equal(coords[2], [-5, 5])


def test_pixelated_cont_shape_flow_get_shape():
    extents = [100, 110, 10]
    pixel_size = [20, 40, 10]
    coords = shapes.PixelatedContShapeFlow.get_relative_cell_coords(
        extents, pixel_size)

    shape = shapes.PixelatedContShapeFlow.get_shape(extents, pixel_size)
    assert shape == [len(coords[0]), len(coords[1]), len(coords[2])]


def make_cuboid_flow_grad(index, priority=0):
    extents = [0, 0, 0]
    pos = [index, 0, 0]
    shape = goos.cuboid(extents=extents, pos=pos, priority=priority)
    flow = goos.CuboidFlow(extents=extents, pos=pos, priority=priority)
    grad = goos.CuboidFlow.Grad(pos_grad=[index, 0, 0])
    return shape, flow, grad


def test_group_shape_1_shape():
    with goos.OptimizationPlan() as plan:
        shape, shape_flow, shape_grad = make_cuboid_flow_grad(1)
        group = goos.GroupShape([shape])

        inputs = [shape_flow]

        assert group.eval(inputs) == goos.ArrayFlow([shape_flow])

        assert group.grad(inputs,
                          goos.ArrayFlow.Grad([shape_grad])) == [shape_grad]


def test_group_shape_2_shape():
    with goos.OptimizationPlan() as plan:
        shape1, flow1, grad1 = make_cuboid_flow_grad(1)
        shape2, flow2, grad2 = make_cuboid_flow_grad(2)
        group = goos.GroupShape([shape1, shape2])

        inputs = [flow1, flow2]
        assert group.eval(inputs) == goos.ArrayFlow([flow1, flow2])

        assert group.grad(inputs,
                          goos.ArrayFlow.Grad([grad1,
                                               grad2])) == [grad1, grad2]


def test_group_shape_2_shape_priority():
    with goos.OptimizationPlan() as plan:
        shape1, flow1, grad1 = make_cuboid_flow_grad(1, priority=1)
        shape2, flow2, grad2 = make_cuboid_flow_grad(2)
        group = goos.GroupShape([shape1, shape2])

        inputs = [flow1, flow2]
        assert group.eval(inputs) == goos.ArrayFlow([flow2, flow1])

        assert group.grad(inputs,
                          goos.ArrayFlow.Grad([grad2,
                                               grad1])) == [grad1, grad2]


def test_group_shape_3_shape_priority():
    with goos.OptimizationPlan() as plan:
        shape1, flow1, grad1 = make_cuboid_flow_grad(1, priority=2)
        shape2, flow2, grad2 = make_cuboid_flow_grad(2)
        shape3, flow3, grad3 = make_cuboid_flow_grad(3, priority=1)
        group = goos.GroupShape([shape1, shape2, shape3])

        inputs = [flow1, flow2, flow3]
        assert group.eval(inputs) == goos.ArrayFlow([flow2, flow3, flow1])
        assert group.grad(inputs,
                          goos.ArrayFlow.Grad([grad2, grad3, grad1
                                              ])) == [grad1, grad2, grad3]


def test_group_shape_3_shape_priority_stable_sort():
    with goos.OptimizationPlan() as plan:
        shape1, flow1, grad1 = make_cuboid_flow_grad(1, priority=2)
        shape2, flow2, grad2 = make_cuboid_flow_grad(2, priority=1)
        shape3, flow3, grad3 = make_cuboid_flow_grad(3, priority=1)
        group = goos.GroupShape([shape1, shape2, shape3])

        inputs = [flow1, flow2, flow3]
        assert group.eval(inputs) == goos.ArrayFlow([flow2, flow3, flow1])
        assert group.grad(inputs,
                          goos.ArrayFlow.Grad([grad2, grad3, grad1
                                              ])) == [grad1, grad2, grad3]


def test_group_shape_array():
    with goos.OptimizationPlan() as plan:
        shape1, flow1, grad1 = make_cuboid_flow_grad(1)
        shape2, flow2, grad2 = make_cuboid_flow_grad(2)
        shape3, flow3, grad3 = make_cuboid_flow_grad(3)
        group = goos.GroupShape([shape1, goos.GroupShape([shape2, shape3])])

        inputs = [flow1, goos.ArrayFlow([flow2, flow3])]
        assert group.eval(inputs) == goos.ArrayFlow([flow1, flow2, flow3])
        assert (group.grad(inputs, goos.ArrayFlow.Grad(
            [grad1, grad2,
             grad3])) == [grad1, goos.ArrayFlow.Grad([grad2, grad3])])


def test_group_shape_array_priority():
    with goos.OptimizationPlan() as plan:
        shape1, flow1, grad1 = make_cuboid_flow_grad(1, priority=2)
        shape2, flow2, grad2 = make_cuboid_flow_grad(2)
        shape3, flow3, grad3 = make_cuboid_flow_grad(3, priority=1)
        shape4, flow4, grad4 = make_cuboid_flow_grad(4)
        group = goos.GroupShape([
            goos.GroupShape([shape1, shape2]), shape3,
            goos.GroupShape([shape4])
        ])

        inputs = [
            goos.ArrayFlow([flow1, flow2]), flow3,
            goos.ArrayFlow([flow4])
        ]
        assert group.eval(inputs) == goos.ArrayFlow(
            [flow2, flow4, flow3, flow1])
        assert (group.grad(inputs,
                           goos.ArrayFlow.Grad(
                               [grad2, grad4, grad3, grad1])) == [
                                   goos.ArrayFlow.Grad([grad1, grad2]), grad3,
                                   goos.ArrayFlow.Grad([grad4])
                               ])
