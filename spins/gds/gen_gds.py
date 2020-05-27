"""Outputs a gds file when given input list of polygon coordinates."""

import os
from typing import List

import gdspy
import numpy as np

NM_PER_UM = 1000


def gen_gds(poly_coords: List[np.ndarray],
            filepath: str,
            extra_structures: List[np.ndarray] = None,
            deembed: bool = True) -> None:
    """Generate GDS given a list of polygon coordinates.

    Output GDS has units of microns and precision in nanometers.

    Args:
        poly_coords: List of polygon coordinates. Elements of list are
            n x 2 numpy.ndarrays. Assumes coordinates in nanometers.
        filepath: Name of the filepath to output gds file.
        extra_structures: List of polygons to be added to the gds that are
            not part of the levelset function.
        deembed: Boolean to control if polygon deembeding will occur or not.
    """
    poly_cell = gdspy.Cell("POLYGONS", exclude_from_current=True)

    test_points = []
    gds_polygons = []

    # Convert polygon coordinates from nanometers to microns since
    # output is in microns.
    poly_coords = [np.array(poly) / NM_PER_UM for poly in poly_coords]

    for polygon in poly_coords:
        test_points.append(tuple(polygon[0]))
        gds_polygons.append(gdspy.Polygon(polygon, 0))

    if deembed:

        containment_mx = []
        for polygon in gds_polygons:
            containment_mx.append(gdspy.inside(test_points, [polygon]))

        # Subtract by identity matrix since polygon_i trivially contains
        # polygon_i.
        containment_mx = np.array(containment_mx) - np.eye(len(gds_polygons))
        # overlap_list[i] tells how many polygons are contained in polygon i.
        overlap_list = np.sum(containment_mx, axis=1)

        overlap_num = 0
        while sum(overlap_list) != 0:

            overlap_num = overlap_num + 1

            # We loop until there are no longer any more polygons of a specific
            # overlap number before going to the next overlap number.
            #
            # The reasoning for this is as we begin to delete polygons,
            # the overlap number changes and so polygons which were of
            # overlap_num > 1, may become polygons of overlap_num = 1.
            #
            # np.where(overlap_list==overlap_num)[0].size is how we see if
            # any polygons satisfy the current overlap number.

            target_polys = np.where(overlap_list == overlap_num)[0]
            while target_polys.size != 0:

                # We iterate through each polygon in our polygon list until no
                # polygon satisfies the current overlap number.
                for i in target_polys[::-1]:
                    containment_list = np.nonzero(containment_mx[i, :])[0]
                    # Concatenate all the contained polygons to subtract out.
                    poly_list = sum([
                        gds_polygons[j].polygons for j in containment_list
                    ], [])
                    # Note that we had to turn precision from the default 1e-3
                    # to 1e-6 to avoid errors in the NOT operation.
                    gds_polygons[i] = gdspy.boolean(gds_polygons[i],
                                                    gdspy.PolygonSet(poly_list),
                                                    "not",
                                                    layer=0,
                                                    precision=1e-6)
                    for j in containment_list:
                        gds_polygons[j] = None
                        test_points[j] = None

                for k in range(len(gds_polygons) - 1, -1, -1):
                    if gds_polygons[k] is None:
                        del gds_polygons[k]
                        del test_points[k]

                containment_mx = []
                for polygon in gds_polygons:
                    containment_mx.append(gdspy.inside(test_points, [polygon]))

                containment_mx = np.array(containment_mx) - np.eye(
                    len(gds_polygons))
                overlap_list = np.sum(containment_mx, axis=1)

                target_polys = np.where(overlap_list == overlap_num)[0]

    for polygon in gds_polygons:
        poly_cell.add(polygon)

    if extra_structures is not None:
        for extra_polygon in extra_structures:
            poly_cell.add(gdspy.Polygon(extra_polygon, 0))

    gdspy.write_gds(filepath, cells=[poly_cell], unit=1.0e-6, precision=1.0e-9)
