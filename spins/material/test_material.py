""" Test that refractive indices match expected value. """
import unittest

import numpy as np

import spins.material as material


class TestRefractiveIndex(unittest.TestCase):

    def test_material_interface(self):
        # Test that the helper methods in Material work properly.
        mat_obj = material.Si()
        np.testing.assert_almost_equal(
            mat_obj.eps_real(1550), 12.0866167)

    def test_Air(self):
        mat_obj = material.Air()
        np.testing.assert_almost_equal(
            mat_obj.refractive_index(1550), (1, 0))

    def test_Si(self):
        mat_obj = material.Si()
        np.testing.assert_almost_equal(
            mat_obj.refractive_index(1300),
            (3.51028918, 4.421444023885752e-04))
        np.testing.assert_almost_equal(
            mat_obj.refractive_index(1550),
            (3.4765811895, 1.999931238411233e-04))

    def test_SiO2(self):
        mat_obj = material.SiO2()
        np.testing.assert_almost_equal(
            mat_obj.refractive_index(1300),
            (1.446887695758727, 0.0267956579768))
        np.testing.assert_almost_equal(
            mat_obj.refractive_index(1550),
            (1.444022172402488, 0.0286394399114))

    def test_Si3N4(self):
        mat_obj = material.Si3N4()
        np.testing.assert_almost_equal(mat_obj.eps_real(800), 4, 1)
