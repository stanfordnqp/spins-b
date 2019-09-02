"""Tests for grating_parametrization.py"""
import unittest

import numpy as np

from spins.invdes.parametrization import grating_parametrization


class TestGratingParam(unittest.TestCase):

    def test_rendering(self):
        edges = np.array([0.1, 1.1, 3.4, 4.1])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.9, 0.1, 0, 0.6, 0.1]),
                                             7)

    def test_rendering_inverted(self):
        edges = np.array([0.1, 1.1, 3.4, 4.1])
        param = grating_parametrization.GratingParam(edges, 5, inverted=True)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.1, 0.9, 1, 0.4, 0.9]),
                                             7)

    def test_rendering_unsorted_edges(self):
        edges = np.array([3.4, 1.1, 4.1, 0.1])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.9, 0.1, 0, 0.6, 0.1]),
                                             7)

    def test_rendering_edge_on_pixel_edge(self):
        edges = np.array([0.1, 1.1, 3.0, 4.1])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.9, 0.1, 0, 1.0, 0.1]),
                                             7)

    def test_rendering_edge_outside_left_grating_boundary(self):
        edges = np.array([-0.2, 0.1, 1.1, 2.4])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.1, 0.9, 0.4, 0, 0]), 7)

    def test_rendering_edge_outside_right_grating_boundary(self):
        edges = np.array([0.2, 1.1, 2.4, 6.7])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.8, 0.1, 0.6, 1, 1]), 7)

    def test_rendering_two_edges_same_pixel(self):
        edges = np.array([0.2, 0.3, 1.1, 2.4, 2.7, 4.1])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.get_structure(),
                                             np.array([0.1, 0.9, 0.7, 1, 0.1]),
                                             7)

    def test_gradient(self):
        edges = np.array([0.1, 1.1, 3.4, 4.1])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([-vec[0], vec[1], 0, -vec[2], vec[3]]), 7)

    def test_gradient_inverted(self):
        edges = np.array([0.1, 1.1, 3.4, 4.1])
        param = grating_parametrization.GratingParam(edges, 5, inverted=True)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            -param.calculate_gradient() @ vec,
            np.array([-vec[0], vec[1], 0, -vec[2], vec[3]]), 7)

    def test_gradient_unsorted_edges(self):
        edges = np.array([3.4, 1.1, 4.1, 0.1])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([-vec[0], vec[1], 0, -vec[2], vec[3]]), 7)

    def test_gradient_edge_on_pixel_edge(self):
        edges = np.array([0.1, 1.1, 3.0, 4.1])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([-vec[0], vec[1], 0, -vec[2], vec[3]]), 7)

    def test_gradient_edge_outside_left_grating_boundary(self):
        edges = np.array([-0.2, 0.1, 1.1, 2.4])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([vec[1], -vec[2], vec[3], 0, 0]), 7)

    def test_gradient_edge_outside_right_grating_boundary(self):
        edges = np.array([0.2, 1.1, 2.4, 6.7])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([-vec[0], vec[1], -vec[2], 0, 0]), 7)

    def test_gradient_edge_at_grating_boundary(self):
        edges = np.array([0, 1.1, 2.4, 5.0])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([-vec[0], vec[1], -vec[2], 0, 0]), 7)

    def test_rendering_two_edges_same_pixel(self):
        edges = np.array([0.2, 0.3, 1.1, 2.4, 2.7, 4.1])
        param = grating_parametrization.GratingParam(edges, 5)
        vec = np.array([2, 1, 2, 3, 4, 2])
        np.testing.assert_array_almost_equal(
            param.calculate_gradient() @ vec,
            np.array([vec[1] - vec[0], -vec[2], vec[3] - vec[4], 0, vec[5]]), 7)

    def test_encode(self):
        edges = np.array([0.2, 0.3, 1.1, 2.9])
        param = grating_parametrization.GratingParam(edges, 5)
        np.testing.assert_array_almost_equal(param.encode(), edges, 7)

    def test_encode_decode(self):
        edges = np.array([0.2, 0.3, 1.1, 2.9])
        param = grating_parametrization.GratingParam(edges, 5)
        edges_new = 2 * edges
        param.decode(edges_new)
        np.testing.assert_array_almost_equal(edges_new, param.encode(), 7)

    def test_raise_value_error_odd_num_edges(self):
        with self.assertRaisesRegex(
                ValueError,
                "The number of edges in the grating expected to be even*"):
            edges = np.array([1, 3, 2])
            grating_parametrization.GratingParam(edges, 5)


if __name__ == "__main__":
    unittest.main()
