import numpy as np

from spins.invdes.parametrization import cubic_utils


def test_floor2vector_rem():
    x_vector = np.array([[1.1, 2, 3], [4.2, -3, 1]])
    x_v = [0, 1, 1.5, 3, 6]

    rem, cell_diff = cubic_utils.floor2vector_rem(x_vector, x_v)
    np.testing.assert_array_almost_equal(rem, [0.1, 1.2, 0.5, -9, 0, 0.])
    np.testing.assert_array_almost_equal(cell_diff, [0.5, 3, 1.5, -6, 3, 0.5])


def test_floor2vector():
    x_vector = np.array([[1.1, 2, 3], [4.2, -3, 1]])
    x_v = [0, 1, 1.5, 3]

    x_near, index = cubic_utils.floor2vector(x_vector, x_v)
    np.testing.assert_array_almost_equal(x_near, [1, 1.5, 1.5, 3, 1.5, 1])
    np.testing.assert_array_almost_equal(index, [1, 2, 2, -1, 2, 1])
