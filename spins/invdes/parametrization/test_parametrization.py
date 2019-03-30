""" Test the parametrization module. """

import numpy as np
import unittest

import spins.invdes.parametrization as param


class TestDirectParam(unittest.TestCase):
    """ Tests direct parametrization. """

    def test_get_structure(self):
        """ Tests that get_structure() works as intended. """
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.get_structure().tolist(), [0, 0.5, 1, 0.5])

    def test_project(self):
        """Test that project() works as intended. """
        # Test that project works for legal values.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        structure.project()
        self.assertEqual(structure.get_structure().tolist(), [0, 0.5, 1, 0.5])

        # Test that project works when values exceed range.
        structure = param.DirectParam(np.array([0, -0.5, 1.2, 0.5]))
        structure.project()
        self.assertEqual(structure.get_structure().tolist(), [0, 0, 1, 0.5])

    def test_encode(self):
        # Test that encode works.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.encode().tolist(), [0, 0.5, 1, 0.5])

    def test_decode(self):
        # Test that decode works properly.
        structure = param.DirectParam(np.array([0, 0]))
        structure.decode(np.array([1, 2, 3]))
        self.assertEqual(structure.encode().tolist(), [1, 2, 3])

    def test_to_vector(self):
        # Test that to_vector works.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.to_vector().tolist(), [0, 0.5, 1, 0.5])

    def test_from_vector(self):
        # Test that from_vector works.
        structure = param.DirectParam(np.array([0, 0, 0]))

        structure.from_vector(np.array([0.5, 0.1, 0.3]))
        self.assertEqual(structure.encode().tolist(), [0.5, 0.1, 0.3])

        # Check for projection.
        structure.from_vector(np.array([-1, 2, 3]))
        self.assertEqual(structure.encode().tolist(), [0, 1, 1])

    def test_get_bounds(self):
        structure = param.DirectParam(np.array([0]))
        self.assertEqual(structure.get_bounds(), ((0,), (1,)))

        structure = param.DirectParam(np.array([0.1, 1]))
        self.assertEqual(structure.get_bounds(), ((0, 0), (1, 1)))

        structure = param.DirectParam(np.array([0.1, 1, 2]), bounds=(2, 3))
        self.assertEqual(structure.get_bounds(), ((2, 2, 2), (3, 3, 3)))

        structure = param.DirectParam(
            np.array([0.1, 1, 2]), bounds=((1, 2, 3), (4, 5, 6)))
        self.assertEqual(structure.get_bounds(), ((1, 2, 3), (4, 5, 6)))

    def test_no_bounds(self):
        structure = param.DirectParam([1000, -1234], bounds=None)
        structure.project()
        self.assertEqual(structure.to_vector().tolist(), [1000, -1234])
        self.assertEqual(structure.get_bounds(), ((None, None), (None, None)))


class TestCubicParam(unittest.TestCase):
    """ Tests direct parametrization. """

    def test_get_structure(self):
        """ Tests that get_structure() works as intended. """
        coarse_x = np.arange(0, 501, 50)
        coarse_y = np.arange(0, 301, 50)
        fine_x = np.arange(0, 501, 10)
        fine_y = np.arange(0, 301, 10)

        shp_c = (len(coarse_x), len(coarse_y))
        shp_f = (len(fine_x), len(fine_y))
        val = np.random.random_sample(shp_c)
        init_val = val.flatten(order='F')

        # No geometry
        par = param.CubicParam(init_val, coarse_x, coarse_y, fine_x, fine_y)
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0::5, 0::5], val_k, rtol=1e-9)

        # sym 0
        par = param.CubicParam(
            init_val, coarse_x, coarse_y, fine_x, fine_y, symmetry=[1, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(
            structure[0:shp_f[0] // 2],
            np.flipud(structure[-shp_f[0] // 2 + 1:]),
            rtol=1e-9)
        # sym 1
        par = param.CubicParam(
            init_val, coarse_x, coarse_y, fine_x, fine_y, symmetry=[0, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(
            structure[:, 0:shp_f[1] // 2],
            np.fliplr(structure[:, -shp_f[1] // 2 + 1:]),
            rtol=1e-9)

        # sym 0,1
        par = param.CubicParam(
            init_val, coarse_x, coarse_y, fine_x, fine_y, symmetry=[1, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(
            structure[0:shp_c[0] // 2, 0:shp_c[1] // 2],
            np.flipud(
                np.fliplr(structure[-shp_c[0] // 2 + 1:, -shp_c[1] // 2 + 1:])),
            rtol=1e-9)

        # periodicity[1,0] periods 0
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 0],
            periods=[0, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)

        # periodicity[0,1] periods 0
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[0, 1],
            periods=[0, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)

        # periodicity[1,1] periods 0
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[0, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)

        coarse_x = np.arange(50, 451, 50)
        coarse_y = np.arange(50, 651, 50)
        fine_x = np.arange(50, 451, 10)
        fine_y = np.arange(50, 651, 10)
        shp_c = (len(coarse_x), len(coarse_y))
        shp_f = (len(fine_x), len(fine_y))
        val = np.random.random_sample(shp_c)
        init_val = val.flatten(order='F')

        # periodicity[1,1] periods [2,1]
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[2, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 2, :],
            structure[:shp_f[0] // 2, :],
            rtol=1e-9)

        # periodicity[1,1] periods [4,2]
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[4, 2])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 4 + 1, :],
            structure[(3 * shp_f[0] // 4):, :],
            rtol=1e-9)

        # periodicity[1,1] periods [2,2] symmetry [1,0]
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[2, 2],
            symmetry=[1, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 2 + 1, :],
            structure[shp_f[0] // 2:, :],
            rtol=1e-9)
        np.testing.assert_allclose(
            structure[:, :shp_f[1] // 2 + 1],
            structure[:, shp_f[1] // 2:],
            rtol=1e-9)
        np.testing.assert_allclose(
            np.flipud(structure[:shp_f[0] // 4 + 1, :shp_f[1] // 2 + 1]),
            structure[shp_f[0] // 4:shp_f[0] // 2 + 1, :shp_f[1] // 2 + 1],
            rtol=1e-9)

        # periodicity[1,1] periods [2,2] symmetry [0,1]
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[2, 2],
            symmetry=[0, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 2 + 1, :],
            structure[shp_f[0] // 2:, :],
            rtol=1e-9)
        np.testing.assert_allclose(
            structure[:, :shp_f[1] // 2 + 1],
            structure[:, shp_f[1] // 2:],
            rtol=1e-9)
        np.testing.assert_allclose(
            np.fliplr(structure[:shp_f[0] // 2 + 1, :shp_f[1] // 4 + 1]),
            structure[:shp_f[0] // 2 + 1, shp_f[1] // 4:shp_f[1] // 2 + 1],
            rtol=1e-9)

        # periodicity[0,1] periods [0,2] symmetry [0,1]
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[0, 1],
            periods=[0, 2],
            symmetry=[0, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:, :shp_f[1] // 2 + 1],
            structure[:, shp_f[1] // 2:],
            rtol=1e-9)
        np.testing.assert_allclose(
            np.fliplr(structure[:shp_f[0] // 2 + 1, :shp_f[1] // 4 + 1]),
            structure[:shp_f[0] // 2 + 1, shp_f[1] // 4:shp_f[1] // 2 + 1],
            rtol=1e-9)

    def test_project(self):
        """Test that project() works as intended. """
        # Test that project works for legal values.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        structure.project()
        self.assertEqual(structure.get_structure().tolist(), [0, 0.5, 1, 0.5])

        # Test that project works when values exceed range.
        structure = param.DirectParam(np.array([0, -0.5, 1.2, 0.5]))
        structure.project()
        self.assertEqual(structure.get_structure().tolist(), [0, 0, 1, 0.5])

    def test_encode(self):
        # Test that encode works.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.encode().tolist(), [0, 0.5, 1, 0.5])

    def test_decode(self):
        # Test that decode works properly.
        structure = param.DirectParam(np.array([0, 0]))
        structure.decode(np.array([1, 2, 3]))
        self.assertEqual(structure.encode().tolist(), [1, 2, 3])

    def test_to_vector(self):
        # Test that to_vector works.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.to_vector().tolist(), [0, 0.5, 1, 0.5])

    def test_from_vector(self):
        # Test that from_vector works.
        structure = param.DirectParam(np.array([0, 0, 0]))

        structure.from_vector(np.array([0.5, 0.1, 0.3]))
        self.assertEqual(structure.encode().tolist(), [0.5, 0.1, 0.3])

        # Check for projection.
        structure.from_vector(np.array([-1, 2, 3]))
        self.assertEqual(structure.encode().tolist(), [0, 1, 1])

    def test_get_bounds(self):
        structure = param.DirectParam(np.array([0]))
        self.assertEqual(structure.get_bounds(), ((0,), (1,)))

        structure = param.DirectParam(np.array([0.1, 1]))
        self.assertEqual(structure.get_bounds(), ((0, 0), (1, 1)))

        structure = param.DirectParam(np.array([0.1, 1, 2]), bounds=(2, 3))
        self.assertEqual(structure.get_bounds(), ((2, 2, 2), (3, 3, 3)))

        structure = param.DirectParam(
            np.array([0.1, 1, 2]), bounds=((1, 2, 3), (4, 5, 6)))
        self.assertEqual(structure.get_bounds(), ((1, 2, 3), (4, 5, 6)))

    def test_no_bounds(self):
        structure = param.DirectParam([1000, -1234], bounds=None)
        structure.project()
        self.assertEqual(structure.to_vector().tolist(), [1000, -1234])
        self.assertEqual(structure.get_bounds(), ((None, None), (None, None)))


class TestHermiteParam(unittest.TestCase):
    """ Tests direct parametrization.
    """

    def test_get_structure(self):
        """ Tests that get_structure() works as intended.
        """
        coarse_x = np.arange(0, 501, 50)
        coarse_y = np.arange(0, 301, 50)
        fine_x = np.arange(0, 501, 10)
        fine_y = np.arange(0, 301, 10)

        shp_c = (len(coarse_x), len(coarse_y))
        shp_f = (len(fine_x), len(fine_y))
        val = np.random.random_sample(shp_c)
        init_val = val.flatten(order='F')

        # No geometry
        par = param.HermiteParam(init_val, coarse_x, coarse_y, fine_x, fine_y)
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0::5, 0::5], val_k, rtol=1e-9)

        # sym 0
        par = param.HermiteParam(
            init_val, coarse_x, coarse_y, fine_x, fine_y, symmetry=[1, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        len_vec = 4 * shp_c[0] * shp_c[1]
        np.testing.assert_allclose(
            structure[0:shp_f[0] // 2],
            np.flipud(structure[-shp_f[0] // 2 + 1:]),
            rtol=1e-9)

        # sym 1
        par = param.HermiteParam(
            init_val, coarse_x, coarse_y, fine_x, fine_y, symmetry=[0, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(
            structure[:, 0:shp_f[1] // 2],
            np.fliplr(structure[:, -shp_f[1] // 2 + 1:]),
            rtol=1e-9)

        # sym 0,1
        par = param.HermiteParam(
            init_val, coarse_x, coarse_y, fine_x, fine_y, symmetry=[1, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(
            structure[0:shp_c[0] // 2, 0:shp_c[1] // 2],
            np.flipud(
                np.fliplr(structure[-shp_c[0] // 2 + 1:, -shp_c[1] // 2 + 1:])),
            rtol=1e-9)

        # periodicity[1,0] periods 0
        par = param.CubicParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 0],
            periods=[0, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)

        # periodicity[0,1] periods 0
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[0, 1],
            periods=[0, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)

        # periodicity[1,1] periods 0
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[0, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)

        coarse_x = np.arange(50, 451, 50)
        coarse_y = np.arange(50, 651, 50)
        fine_x = np.arange(50, 451, 10)
        fine_y = np.arange(50, 651, 10)
        shp_c = (len(coarse_x), len(coarse_y))
        shp_f = (len(fine_x), len(fine_y))
        val = np.random.random_sample(shp_c)
        init_val = val.flatten(order='F')

        # periodicity[1,1] periods [2,1]
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[2, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 2, :],
            structure[:shp_f[0] // 2, :],
            rtol=1e-9)

        # periodicity[1,1] periods [4,2]
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[4, 2])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 4 + 1, :],
            structure[(3 * shp_f[0] // 4):, :],
            rtol=1e-9)

        # periodicity[1,1] periods [2,2] symmetry [1,0]
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[2, 2],
            symmetry=[1, 0])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 2 + 1, :],
            structure[shp_f[0] // 2:, :],
            rtol=1e-9)
        np.testing.assert_allclose(
            structure[:, :shp_f[1] // 2 + 1],
            structure[:, shp_f[1] // 2:],
            rtol=1e-9)
        np.testing.assert_allclose(
            np.flipud(structure[:shp_f[0] // 4 + 1, :shp_f[1] // 2 + 1]),
            structure[shp_f[0] // 4:shp_f[0] // 2 + 1, :shp_f[1] // 2 + 1],
            rtol=1e-9)

        # periodicity[1,1] periods [2,2] symmetry [0,1]
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[1, 1],
            periods=[2, 2],
            symmetry=[0, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[0], structure[-1], rtol=1e-9)
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:shp_f[0] // 2 + 1, :],
            structure[shp_f[0] // 2:, :],
            rtol=1e-9)
        np.testing.assert_allclose(
            structure[:, :shp_f[1] // 2 + 1],
            structure[:, shp_f[1] // 2:],
            rtol=1e-9)
        np.testing.assert_allclose(
            np.fliplr(structure[:shp_f[0] // 2 + 1, :shp_f[1] // 4 + 1]),
            structure[:shp_f[0] // 2 + 1, shp_f[1] // 4:shp_f[1] // 2 + 1],
            rtol=1e-9)

        # periodicity[0,1] periods [0,2] symmetry [0,1]
        par = param.HermiteParam(
            init_val,
            coarse_x,
            coarse_y,
            fine_x,
            fine_y,
            periodicity=[0, 1],
            periods=[0, 2],
            symmetry=[0, 1])
        val_k = 1 / (1 + np.exp(-par.k * (2 * val - 1)))
        structure = np.reshape(par.get_structure(), shp_f, order='F')
        np.testing.assert_allclose(structure[:, 0], structure[:, -1], rtol=1e-9)
        np.testing.assert_allclose(
            structure[:, :shp_f[1] // 2 + 1],
            structure[:, shp_f[1] // 2:],
            rtol=1e-9)
        np.testing.assert_allclose(
            np.fliplr(structure[:shp_f[0] // 2 + 1, :shp_f[1] // 4 + 1]),
            structure[:shp_f[0] // 2 + 1, shp_f[1] // 4:shp_f[1] // 2 + 1],
            rtol=1e-9)

    def test_project(self):
        """Test that project() works as intended. """
        # Test that project works for legal values.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        structure.project()
        self.assertEqual(structure.get_structure().tolist(), [0, 0.5, 1, 0.5])

        # Test that project works when values exceed range.
        structure = param.DirectParam(np.array([0, -0.5, 1.2, 0.5]))
        structure.project()
        self.assertEqual(structure.get_structure().tolist(), [0, 0, 1, 0.5])

    def test_encode(self):
        # Test that encode works.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.encode().tolist(), [0, 0.5, 1, 0.5])

    def test_decode(self):
        # Test that decode works properly.
        structure = param.DirectParam(np.array([0, 0]))
        structure.decode(np.array([1, 2, 3]))
        self.assertEqual(structure.encode().tolist(), [1, 2, 3])

    def test_to_vector(self):
        # Test that to_vector works.
        structure = param.DirectParam(np.array([0, 0.5, 1, 0.5]))
        self.assertEqual(structure.to_vector().tolist(), [0, 0.5, 1, 0.5])

    def test_from_vector(self):
        # Test that from_vector works.
        structure = param.DirectParam(np.array([0, 0, 0]))

        structure.from_vector(np.array([0.5, 0.1, 0.3]))
        self.assertEqual(structure.encode().tolist(), [0.5, 0.1, 0.3])

        # Check for projection.
        structure.from_vector(np.array([-1, 2, 3]))
        self.assertEqual(structure.encode().tolist(), [0, 1, 1])

    def test_get_bounds(self):
        structure = param.DirectParam(np.array([0]))
        self.assertEqual(structure.get_bounds(), ((0,), (1,)))

        structure = param.DirectParam(np.array([0.1, 1]))
        self.assertEqual(structure.get_bounds(), ((0, 0), (1, 1)))

        structure = param.DirectParam(np.array([0.1, 1, 2]), bounds=(2, 3))
        self.assertEqual(structure.get_bounds(), ((2, 2, 2), (3, 3, 3)))

        structure = param.DirectParam(
            np.array([0.1, 1, 2]), bounds=((1, 2, 3), (4, 5, 6)))
        self.assertEqual(structure.get_bounds(), ((1, 2, 3), (4, 5, 6)))

    def test_no_bounds(self):
        structure = param.DirectParam([1000, -1234], bounds=None)
        structure.project()
        self.assertEqual(structure.to_vector().tolist(), [1000, -1234])
        self.assertEqual(structure.get_bounds(), ((None, None), (None, None)))
