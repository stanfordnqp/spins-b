from typing import List, Tuple

import logging

import numpy as np

from spins import goos
from spins import fdfd_tools
from spins import gridlock

logger = logging.getLogger(__name__)


class MeshModel(goos.Model):
    pass


@goos.polymorphic_model()
class UniformMesh(MeshModel):
    """Defines a uniform mesh.

    Attributes:
        type: Must be "uniform".
        dx: Unit cell distance for EM grid (nm).
    """
    type = goos.ModelNameType("uniform")
    dx = goos.types.FloatType()


class Symmetry:
    """Declares symmetries for `SimulationSpace`."""
    NONE = 0
    PEC = 1
    PMC = 2


class SimulationSpace(goos.Model):
    """Defines a simulation space.

    A simulation space contains information regarding the permittivity
    distributions but not the fields, i.e. no information regarding sources
    and wavelengths.

    Attributes:
        name: Name to identify the simulation space. Must be unique.
        mesh: Meshing information. This describes how the simulation region
            should be meshed.
        sim_region: Rectangular prism simulation domain.
        reflection_symmetry: Three element list with symmetry values in every axis
                            - 0: no symmetry
                            - 1: electric anti-symmetry around the center
                            - 2: electric symmetry around the center
    """
    type = goos.ModelNameType("simulation_space")
    mesh = goos.types.PolyModelType(MeshModel)
    sim_region = goos.types.ModelType(goos.Box3d)
    pml_thickness = goos.types.ListType(goos.types.IntType(),
                                        min_size=6,
                                        max_size=6)
    reflection_symmetry = goos.types.ListType(goos.types.IntType(),
                                              min_size=3,
                                              max_size=3)


def create_edge_coords(
        sim_region: goos.Box3d,
        dx: float,
        reflection_symmetry: List[int] = None) -> fdfd_tools.EdgeCoords:
    """Creates the edge coordinates of the grid for a uniform grid.

    Args:
        sim_region: The box defining the simulation region.
        dx: The grid spacing.

    Returns:
        Tuple where each element corresponds to one axis and contains an array
        that has the coordinates for the grid along that axis.
    """
    if reflection_symmetry is None:
        reflection_symmetry = [0, 0, 0]

    extents_raw = np.array(sim_region.extents)
    # Fill in `dx` for any extents that are smaller than `dx`.
    extents = np.maximum(extents_raw, dx)

    # Give warning and modify simulation extent, if it will produce an odd dxes
    # length in a symmetric axis
    for i, ext in enumerate(extents):
        if reflection_symmetry[i]:
            if not (ext / (2 * dx)).is_integer():
                extents[i] = np.floor(ext / (2 * dx)) * 2 * dx
                logger.warning(
                    "Symmetry requires simulation extents to be an integer "
                    "multiple of `2 * dx`, the simulation extents for {} "
                    "direction has been changed to {}".format(
                        "xyz"[i], extents[i]))

    xyz_min = np.array(sim_region.center) - np.array(extents) / 2
    xyz_max = np.array(sim_region.center) + np.array(extents) / 2

    edge_coords = []
    for i in range(3):
        edge_coords.append(np.arange(xyz_min[i], xyz_max[i] + dx / 2, dx))

    # Recenter around the center.
    edge_coords = [
        e - (e[0] + e[-1]) / 2 + c
        for e, c in zip(edge_coords, sim_region.center)
    ]

    return edge_coords


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
