"""Tests for float_raster.py"""
import unittest

import numpy as np

from spins.gridlock import float_raster


class TestRaster1D(unittest.TestCase):
    def test_segment(self):
        grid_x = np.arange(5)
        poly_x = np.array([1.5, 3.3])
        render = float_raster.raster_1D(poly_x, grid_x)
        np.testing.assert_array_almost_equal(
                render, np.array([0, 0.5, 1, 0.3]), 7)

    def test_segment_nonuniform(self):
        grid_x = np.array([0, 2, 2.5, 3.0, 3.1, 3.3])
        poly_x = np.array([1.5, 3.3])
        render = float_raster.raster_1D(poly_x, grid_x)
        np.testing.assert_array_almost_equal(
                render, np.array([0.25, 1, 1, 1, 1]), 7)

    def test_segment_partially_outside_grid(self):
        grid_x = np.arange(5)

        # Segment to the right of the grid.
        poly_x = np.array([5.0, 6.0])
        render = float_raster.raster_1D(poly_x, grid_x)
        np.testing.assert_array_almost_equal(
                render, np.zeros(4), 7)

        # Segment to the left of the grid.
        poly_x = np.array([-1.0, -0.5])
        render = float_raster.raster_1D(poly_x, grid_x)
        np.testing.assert_array_almost_equal(
                render, np.zeros(4), 7)

    def test_segment_overlap_pixel(self):
        grid_x = np.arange(5)
        poly_x = np.array([1, 2])
        render = float_raster.raster_1D(poly_x, grid_x)
        np.testing.assert_array_almost_equal(
                render, np.array([0, 1, 0, 0]), 7)

    def test_segment_inside_pixel(self):
        grid_x = np.arange(5)
        poly_x = np.array([1.1, 1.9])
        render = float_raster.raster_1D(poly_x, grid_x)
        np.testing.assert_array_almost_equal(
                render, np.array([0, 0.8, 0, 0]), 7)

    def test_raise_value_error_invalid_poly_x(self):
        with self.assertRaisesRegex(
                ValueError,
                "Expected `poly_x` to have exactly 2 elements, got*"):
            float_raster.raster_1D(np.array([1, 2, 4]),
                                   np.arange(5))

    def test_raise_value_error_invalid_grid_x(self):
        with self.assertRaisesRegex(
                ValueError,
                "Expected `grid_x` to have at least 2 elements, got*"):
            float_raster.raster_1D(np.array([1, 2]),
                                   np.array([0]))


class TestRaster2D(unittest.TestCase):
    def test_square(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1, 3.5, 3.5, 1],
                            [1, 1, 3.5, 3.5]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
            render,
            np.array([[0, 0, 0, 0],
                      [0, 1, 1, 0.5],
                      [0, 1, 1, 0.5],
                      [0, 0.5, 0.5, 0.25]]), 7)

    def test_square_nonuniform(self):
        grid_x = np.array([0, 1, 3, 3.5, 4.2])
        grid_y = np.array([0, 2.5, 3, 3.7])
        poly_xy = np.array([[1, 3.5, 3.5, 1],
                            [1, 1, 3.5, 3.5]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
            render,
            np.array([[0, 0, 0],
                      [0.6, 1, 5.0 / 7.0],
                      [0.6, 1, 5.0 / 7.0],
                      [0, 0, 0]]), 7)

    def test_square_partially_outside_grid(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1, 5, 5, 1],
                            [1, 1, 3.5, 3.5]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
            render,
            np.array([[0, 0, 0, 0],
                      [0, 1, 1, 0.5],
                      [0, 1, 1, 0.5],
                      [0, 1, 1, 0.5]]), 7)

    def test_square_outside_grid(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)

        # Polygon outside x-extent.
        poly_xy = np.array([[5, 7.5, 7.5, 5],
                            [1, 1, 3.5, 3.5]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(render,
                                             np.zeros((4, 4)), 7)
        # Polygon outside y-extent.
        poly_xy = np.array([[1, 3.5, 3.5, 1],
                            [5, 5, 7.5, 7.5]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(render,
                                             np.zeros((4, 4)), 7)

        # Polygon outside x-extent and y-extent.
        poly_xy = np.array([[5, 7.5, 7.5, 5],
                            [5, 5, 7.5, 7.5]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(render,
                                             np.zeros((4, 4)), 7)

    def test_square_overlap_pixel(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1, 2, 2, 1],
                            [1, 1, 2, 2]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
                render,
                np.array([[0, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]), 7)

    def test_square_inside_pixel(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1.1, 1.9, 1.9, 1.1],
                            [1.1, 1.1, 1.9, 1.9]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
                render,
                np.array([[0, 0, 0, 0],
                          [0, 0.64, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]), 7)

    def test_triangle(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1, 3, 3],
                            [1, 1, 4]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
            render,
            np.array([[0, 0, 0, 0],
                      [0, 2.0 / 3.0, 1.0 / 12.0, 0],
                      [0, 1, 11.0 / 12.0, 1.0 / 3.0],
                      [0, 0, 0, 0]]), 7)

    def test_triangle_partially_outside_grid(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1, 5, 3],
                            [1, 1, 4]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
            render,
            np.array([[0, 0, 0, 0],
                      [0, 2.0 / 3.0, 1.0 / 12.0, 0],
                      [0, 1, 11.0 / 12.0, 1.0 / 3.0],
                      [0, 1, 11.0 / 12.0, 1.0 / 3.0]]), 7)

    def test_triangle_inside_pixel(self):
        grid_x = np.arange(5)
        grid_y = np.arange(5)
        poly_xy = np.array([[1, 1.5, 1.5],
                            [1, 1, 2]])
        render = float_raster.raster_2D(poly_xy, grid_x, grid_y)
        np.testing.assert_array_almost_equal(
            render,
            np.array([[0, 0, 0, 0],
                      [0, 0.25, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]), 7)

    def test_raise_value_error_invalid_poly_xy(self):
        with self.assertRaisesRegex(
                ValueError,
                "Expected `poly_xy` to have 2 rows, got*"):
            float_raster.raster_2D(np.array([[1, 2, 4]]),
                                   np.arange(5),
                                   np.arange(5))

    def test_raise_value_error_invalid_grid_x(self):
        with self.assertRaisesRegex(
                ValueError,
                "Expected both `grid_x` and `grid_y` to have atleast 2*"):
            float_raster.raster_2D(np.array([[1, 2, 4], [1, 2, 1]]),
                                   np.array([0]),
                                   np.array([1, 2, 3]))

    def test_raise_value_error_invalid_grid_x(self):
        with self.assertRaisesRegex(
                ValueError,
                "Expected both `grid_x` and `grid_y` to have atleast 2*"):
            float_raster.raster_2D(np.array([[1, 2, 4], [1, 2, 1]]),
                                   np.array([0, 1, 2]),
                                   np.array([]))



if __name__ == "__main__":
    unittest.main()
