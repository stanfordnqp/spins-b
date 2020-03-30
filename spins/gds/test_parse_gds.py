"""Tests GDS loading."""
import os
import numpy as np
import pytest

from spins import gds

TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")


def test_load_gds():
    with open(os.path.join(TESTDATA, "rect.gds"), "rb") as fp:
        gds_file = gds.GDSImport(fp)
    polygons = gds_file.get_polygons((100, 0))

    assert len(polygons) == 1
    np.testing.assert_almost_equal(polygons[0],
                                   [[-1, 0.7], [-5, 0.7], [-5, 0.2], [-1, 0.2]])

    boxes = gds_file.get_bounding_boxes((100, 0))
    assert len(boxes) == 1
    np.testing.assert_almost_equal(polygons[0],
                                   [[-1, 0.7], [-5, 0.7], [-5, 0.2], [-1, 0.2]])


def test_load_gds_diff_units():
    """Tests that loading GDS file with different units work.

    "rect_um.gds" is identical to "rect.gds" except for the fact that the
    precision changes from 1e-9 (nm) to 1e-6 (um) (the unit of the files
    are the same (both are um)). This means that the polygons should be 1000
    times larger than in "rect.gds".
    """
    with open(os.path.join(TESTDATA, "rect_um.gds"), "rb") as fp:
        gds_file = gds.GDSImport(fp)
    polygons = gds_file.get_polygons((100, 0))

    assert len(polygons) == 1
    np.testing.assert_almost_equal(
        polygons[0], [[-1000, 700], [-5000, 700], [-5000, 200], [-1000, 200]])


def test_standard_load():
    with open(os.path.join(TESTDATA, "tlc_test_gds1.gds"), "rb") as fp:
        gds_file = gds.GDSImport(fp)
    assert gds_file.top_level_cell.name == "CELL1"


def test_named_cell_load():
    with open(os.path.join(TESTDATA, "tlc_test_gds1.gds"), "rb") as fp:
        gds_file = gds.GDSImport(fp, "CELL2")
    assert gds_file.top_level_cell.name == "CELL2"


def test_name_not_found():
    with pytest.raises(ValueError, match="name not found"):
        with open(os.path.join(TESTDATA, "tlc_test_gds1.gds"), "rb") as fp:
            gds_file = gds.GDSImport(fp, "CELL3")


def test_same_num_polygons_load():
    with pytest.raises(ValueError, match="same number"):
        with open(os.path.join(TESTDATA, "tlc_test_gds2.gds"), "rb") as fp:
            gds_file = gds.GDSImport(fp)


def test_no_valid_cell():
    with pytest.raises(ValueError, match="valid cell"):
        with open(os.path.join(TESTDATA, "tlc_test_gds3.gds"), "rb") as fp:
            gds_file = gds.GDSImport(fp)
