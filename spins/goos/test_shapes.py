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
