"""This module defines types for the package."""
from typing import Callable, List, Tuple

import numpy as np

# Define useful types.
# Define a 1D field in 3D space where each point takes on a single value.
ScalarField = np.ndarray
# Defines a 3D vector field, i.e. one that has three components for every
# point in space.
VecField = Tuple[ScalarField, ScalarField, ScalarField]
# Defines a 3D complex vector.
Vec3d = Tuple[complex, complex, complex]
# Defines the grid spacing array. This is an array of two tuples, each of which
# corresponds to the primary and secondary grid spacing. Each grid spacing
# is another tuple of three linear arrays corresponding to actual grid spacing.
GridSpacing = Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[
    np.ndarray, np.ndarray, np.ndarray]]
# List of PML layers for each side of the simulation domain. Order is
# x-, x+, y-, y+, z-, z+.
PmlLayers = Tuple[int, int, int, int, int, int]
# Represents the edge coordinates of a simulation grid. Each element
# of the tuple cooresponds to a single axis and holds a 1D array corresponding
# to the coordinates of the grid.
EdgeCoords = Tuple[np.ndarray, np.ndarray, np.ndarray]

# TODO(logansu): Remove old-style types.
dx_lists_t = List[List[np.ndarray]]
vfield_t = np.ndarray
field_t = VecField
