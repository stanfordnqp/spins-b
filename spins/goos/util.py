"""This module contains small miscellaneous utility functions."""
from typing import Optional, Tuple, Union

import logging
import os

import numpy as np

from spins import goos

from spins.invdes.problem_graph.workspace import get_latest_log_file
from spins.invdes.problem_graph.workspace import get_latest_log_step


def setup_logging(save_folder: str, log_level: int = logging.INFO):
    """Setup logging.

    This will setup logging to stdout as well as to a `spins.log` file within
    `save_folder`.

    Args:
        save_folder: Folder to save logs.
        log_level: Default logging level.

    Returns:
        The log file handler.
    """
    os.makedirs(save_folder, exist_ok=True)

    # Setup logging.
    logging.basicConfig(format=goos.LOG_FORMAT)

    # Now also log to file.
    log_file_handler = logging.FileHandler(
        os.path.join(save_folder, "spins.log"))
    log_file_handler.setFormatter(logging.Formatter(goos.LOG_FORMAT))
    # Add handler to root logger.
    logging.getLogger("").addHandler(log_file_handler)

    logging.getLogger("").setLevel(log_level)
    # Disable requests logging because it is very verbose.
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("connectionpool").setLevel(logging.ERROR)

    return log_file_handler


def visualize_fields(
        fields: Union[np.ndarray, goos.NumericFlow],
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
) -> None:
    """Plots all the field components.

    This function is meant as a quick utility to draw vector fields when
    debugging.

    Args:
        fields: A numpy array with dimensions `(3, num_x, num_y, num_z)`.
        x: If set, the field array is sliced along the given x-index.
        y: If set, the field array is sliced along the given y-index.
        z: If set, the field array is sliced along the given z-index.
    """
    import matplotlib.pyplot as plt

    if isinstance(fields, goos.Flow):
        fields = fields.array

    slicer = make_slice(x=x, y=y, z=z, shape=fields.shape)

    plt.figure(figsize=(10, 5))
    for i, comp_name in zip(range(3), "xyz"):
        plt.subplot(2, 3, i + 1)
        plt.imshow(np.real(fields[i][slicer].squeeze()))
        plt.colorbar()
        plt.title("Re[E{}]".format(comp_name))

        plt.subplot(2, 3, i + 4)
        plt.imshow(np.imag(fields[i][slicer].squeeze()))
        plt.colorbar()
        plt.title("Im[E{}]".format(comp_name))
    plt.show()


def visualize_eps(
        eps: Union[np.ndarray, goos.NumericFlow],
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
) -> None:
    """Plots permittivity distribution.

    This function is meant as a quick utility to draw epsilon when debugging.

    Args:
        eps: A numpy array with dimensions `(num_x, num_y, num_z)`.
        x: If set, the field array is sliced along the given x-index.
        y: If set, the field array is sliced along the given y-index.
        z: If set, the field array is sliced along the given z-index.
    """
    import matplotlib.pyplot as plt

    if isinstance(eps, goos.Flow):
        eps = eps.array

    slicer = make_slice(x=x, y=y, z=z, shape=eps.shape)

    plt.imshow(np.real(eps[slicer].squeeze()))
    plt.colorbar()
    plt.title("Re[Eps]")
    plt.show()


def make_slice(
        x: Optional[Union[int, str]] = None,
        y: Optional[Union[int, str]] = None,
        z: Optional[Union[int, str]] = None,
        shape: Tuple[int, int, int] = None,
) -> Tuple:
    """Creates a 3D `slice` object.

    Depending on the coordinate set in the arguments (x, y, or z), a slice
    object will be created that extracts the numpy array at the given index.

    Example:
    ```python

    arr[make_slice(x=3)] == arr[3, :, :]
    arr[make_slice(y=3)] == arr[:, 3, :]
    arr[make_slice(x=2,z=3)] == arr[2, :, 3]
    ```

    Args:
        x: If set, the field array is sliced along the given x-index.
        y: If set, the field array is sliced along the given y-index.
        z: If set, the field array is sliced along the given z-index.

    Returns:
        A tuple of slice objects that can be applied on a numpy array.
    """
    slicer = [slice(None), slice(None), slice(None)]

    if x == "center":
        x = shape[0] // 2
    if y == "center":
        y = shape[1] // 2
    if z == "center":
        z = shape[2] // 2

    if x:
        slicer[0] = slice(x, x + 1)
    if y:
        slicer[1] = slice(y, y + 1)
    if z:
        slicer[2] = slice(z, z + 1)
    return tuple(slicer)
