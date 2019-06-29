"""Contains useful functions for handling grids."""
from typing import Tuple

import numpy as np

from spins import fdfd_tools
from spins import gridlock


# TODO(logansu): Refactor `gridlock.Grid` into a portion that handles just
# the grid part and another part that handles drawing. Then these functions
# can go poof.
def create_region_slices(
        edge_coords: fdfd_tools.EdgeCoords,
        center: fdfd_tools.Vec3d,
        extents: fdfd_tools.Vec3d,
) -> Tuple[slice, slice, slice]:
    """Return `slice` objects corresponding to a given rectangular region.

    `center` and `extents` describe a rectangular prism in the space described
    by a grid with edge coordinates given by `edge_coords`. This function
    returns three `slice` objects corresponding to each of the three axes that
    select out this region. For example, suppose `eps` is a
    `fdfd_tools.VecField` that lives on a grid described by `edge_coords`. Then
    `eps[calculate_slices(...)]` would correspond to the portion of `eps` that
    lies in the rectangular region.

    Args:
        center: Three element array indicating the center of the region in the
                grid's units.
        extent: Three element array indicating the size of the region in the
                grid's units.

    Returns:
        Three element tuple with the slices.
    """
    # Get min and max position.
    xyz_min = np.array(center) - np.array(extents) / 2
    xyz_max = np.array(center) + np.array(extents) / 2

    # Make a dummy grid so we can use `pos2ind`.
    grid = gridlock.Grid(edge_coords, num_grids=3)
    grid_min = [v[0] for v in grid.xyz]
    grid_max = [v[-1] for v in grid.xyz]

    # Adjust min max.
    xyz_min_clip = [max(gr, sl) for gr, sl in zip(grid_min, xyz_min)]
    xyz_max_clip = [min(gr, sl) for gr, sl in zip(grid_max, xyz_max)]

    # Get the min and max indices.
    xyz_ind_min = grid.pos2ind(xyz_min_clip, which_shifts=None).astype(int)
    xyz_ind_max = grid.pos2ind(xyz_max_clip, which_shifts=None).astype(int)

    # TODO(logansu): Revisit and consider whether `pos2ind` should allow
    # out of bounds coordinates.
    # Set slices correctly if clipped. The issue at hand here is that if
    # `xyz_max` is out of bounds, we actually want to make sure the slice
    # includes the border.
    xyz_ind_min[xyz_min_clip > xyz_min] = 0
    xyz_ind_max[xyz_max_clip < xyz_max] += 1

    # Make sure that slices are nonzero.
    for i in range(3):
        xyz_ind_max[i] = max(xyz_ind_max[i], xyz_ind_min[i] + 1)

    # TODO(logansu): Return tuple instead of array. Only tuple slices work
    # with np.ndarrays.
    return [slice(mi, ma) for mi, ma in zip(xyz_ind_min, xyz_ind_max)]
