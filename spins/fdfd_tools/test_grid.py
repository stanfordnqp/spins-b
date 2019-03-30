""" Test grid """

import unittest
import numpy as np

from spins.fdfd_tools import grid


class TestMakeNonUniformGrid(unittest.TestCase):
    '''
    The test makes a grid with 2 box regions and then checks if dx in the center
    of the box is smaller than the requested mesh size.
    '''

    def test_make_nonuniform_grid(self):
        SimBorders = [-1000, 1000, -1000, 1000, -1000, 1000]
        dx_default = [50, 50, 50]
        Box1 = {
            'pos': [-100, 0, 0],
            'size': [100, 100, 100],
            'meshsize': [10, 10, 10]
        }
        Box2 = {
            'pos': [0, 0, 0],
            'size': [200, 200, 200],
            'meshsize': [30, 30, 30]
        }
        Boxes = [Box1, Box2]

        xs, ys, zs = grid.make_nonuniform_grid(SimBorders, dx_default, Boxes)

        find_ind = lambda x, vec: np.abs(x - vec).argmin()
        find_all_ind = lambda pos, vecs: [
            find_ind(pos[i], vecs[i]) for i in range(3)
        ]
        vecs = [xs, ys, zs]

        ind_box1 = find_all_ind(Box1['pos'], vecs)
        ind_box2 = find_all_ind(Box2['pos'], vecs)

        self.assertTrue(np.diff(xs)[ind_box1[0]] < Box1['meshsize'][0])
        self.assertTrue(np.diff(ys)[ind_box1[1]] < Box1['meshsize'][1])
        self.assertTrue(np.diff(zs)[ind_box1[2]] < Box1['meshsize'][2])

        self.assertTrue(np.diff(xs)[ind_box2[0]] < Box2['meshsize'][0])
        self.assertTrue(np.diff(ys)[ind_box2[1]] < Box2['meshsize'][1])
        self.assertTrue(np.diff(zs)[ind_box2[2]] < Box2['meshsize'][2])


def make_dxes():
    dx = 2
    shape = [3, 3, 3]
    return [[
        np.array([dx] * shape[0]),
        np.array([dx] * shape[1]),
        np.array([dx] * shape[2]),
    ] for i in range(2)]


def test_apply_scpml_no_pmls():
    dxes = make_dxes()
    dxes_new = grid.apply_scpml(dxes, None, 1)

    # Check that `dxes_new` is a copy.
    assert dxes is not dxes_new
    np.testing.assert_array_equal(dxes, dxes_new)


def test_apply_scpml_with_pmls_xdir_left():
    dxes = make_dxes()
    dxes_new = grid.apply_scpml(dxes, [2, 0, 0, 0, 0, 0], 1)
    dxes_exp = make_dxes()
    dxes_exp[0][0] = np.array([2. - 6.328125j, 2. - 0.078125j, 2. + 0.j])
    dxes_exp[1][0] = np.array([2. - 20.j, 2. - 1.25j, 2. + 0.j])
    np.testing.assert_array_equal(dxes_new, dxes_exp)


def test_apply_scpml_with_pmls_xdir_right():
    dxes = make_dxes()
    dxes_new = grid.apply_scpml(dxes, [0, 2, 0, 0, 0, 0], 1)
    dxes_exp = make_dxes()
    dxes_exp[0][0] = np.array([2. + 0.j, 2. - 0.078125j, 2. - 6.328125j])
    dxes_exp[1][0] = np.array([2. + 0.j, 2. + 0.j, 2. - 1.25j])
    np.testing.assert_array_equal(dxes_new, dxes_exp)


def test_apply_scpml_with_pmls_ydir_left():
    dxes = make_dxes()
    dxes_new = grid.apply_scpml(dxes, [0, 0, 2, 0, 0, 0], 1)
    dxes_exp = make_dxes()
    dxes_exp[0][1] = np.array([2. - 6.328125j, 2. - 0.078125j, 2. + 0.j])
    dxes_exp[1][1] = np.array([2. - 20.j, 2. - 1.25j, 2. + 0.j])
    np.testing.assert_array_equal(dxes_new, dxes_exp)


def test_apply_scpml_with_pmls_equal_all_sides():
    dxes = make_dxes()
    dxes_exp = grid.apply_scpml(dxes, [2] * 6, 1)
    dxes_new = grid.apply_scpml(dxes, 2, 1)
    np.testing.assert_array_equal(dxes_new, dxes_exp)
