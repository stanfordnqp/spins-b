"""Helper functions for handling vectors that represent a spatial direction."""
from typing import List

from enum import Enum
import numpy as np


class Direction(Enum):
    """
    Enum for axis->integer mapping
    """
    x = 0
    y = 1
    z = 2


def axisvec2polarity(vector: np.ndarray) -> int:
    """Return the polarity along the vector's primary coordinate axis.

     Args:
         vector: The direction vector.

     Returns:
         The polarity of the vector, which is either 1 (for positive direction)
         and -1 (for negative direction).
    """
    if isinstance(vector, List):
        vec = np.array(vector)
    else:
        vec = vector

    axis = axisvec2axis(vec)

    return np.sign(vec[axis])


def axisvec2axis(vector: np.ndarray) -> int:
    """Return the vector's primary coordinate axis.

     Args:
         vector: The direction vector.

     Returns:
         axis: Direction axis.

     Raises:
         ValueError: If the vector is not axis-aligned.
    """
    if isinstance(vector, List):
        vec = np.array(vector)
    else:
        vec = vector

    norm = np.linalg.norm(vec)
    delta = 1e-6 * norm

    # Check that only one element of vector is larger than delta.
    if sum(abs(vec) > delta) != 1:
        raise ValueError(
            "Vector has no valid primary coordinate axis, got: {}".format(vec))

    axis = np.argwhere(abs(vec) > delta).flatten()[0]

    return axis
