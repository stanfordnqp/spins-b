""" Test grid """
import unittest

import numpy as np
import pytest

from spins.gridlock import axisvec2polarity, axisvec2axis


class TestDirection(unittest.TestCase):
    """Tests if direction vectors are converted correctly."""

    def test_axisvec2axis_and_axisvec2polarity(self):

        # x-dir
        vec = np.array([1, 0, 0])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 0)
        self.assertTrue(pol == 1)

        vec = np.array([2, 1e-8, 1e-8])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 0)
        self.assertTrue(pol == 1)

        vec = np.array([-1, 0, 0])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 0)
        self.assertTrue(pol == -1)

        # y-dir
        vec = np.array([0, 1, 0])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 1)
        self.assertTrue(pol == 1)

        vec = np.array([0, -1, 0])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 1)
        self.assertTrue(pol == -1)

        # z-dir
        vec = np.array([0, 0, 1])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 2)
        self.assertTrue(pol == 1)

        vec = np.array([0, 0, -1])
        ax = axisvec2axis(vec)
        pol = axisvec2polarity(vec)
        self.assertTrue(ax == 2)
        self.assertTrue(pol == -1)


@pytest.mark.parametrize("vec", [
    [1, 1],
    [1, 0.01],
    [0.01, 1],
    [0, 0, 0],
])
def test_axisvec2axis_no_primary_coordinate_raises_value_error(vec):
    with pytest.raises(ValueError, match="no valid primary coordinate axis"):
        axisvec2axis(vec)
