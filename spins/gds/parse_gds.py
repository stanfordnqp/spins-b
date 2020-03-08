"""Module for importing and accessing objects in GDSII file format."""
import copy
from typing import BinaryIO, Dict, NamedTuple, List, Optional, Tuple

import gdspy
import numpy as np


class Box(
        NamedTuple("Box", [("x_minmax", List[float]),
                           ("y_minmax", List[float])])):
    """Defines a box.

    Attributes:
        x_minmax: The min and max coordinates for the x coordinate.
        y_minmax: The min and max coordinates for the y coordinate.
    """

    @property
    def center(self) -> Tuple[float, float]:
        """Gets the center coordinate of the box."""
        return (self.x_minmax[0] + self.x_minmax[1]) / 2, (
            self.y_minmax[0] + self.y_minmax[1]) / 2

    @property
    def extents(self) -> Tuple[float, float]:
        """Computes the extents of the box."""
        return (self.x_minmax[1] - self.x_minmax[0],
                self.y_minmax[1] - self.y_minmax[0])


class GDSImport:
    """Imports information about polygon contents of a GDSII file."""

    def __init__(self, gds_file: BinaryIO,
                 cell_name: Optional[str] = None) -> None:
        """Initializes the object with layer information from target GDSII file.

        Args:
            gds_file: ``gds_file`` is not the name of the string but rather the
                context manager created by ``with open as gds_file.``
            cell_name: Optional argument to specify the name of the cell that
                will be the top level cell. Otherwise, the cell which contains
                the most polygons will be used.
        """
        self.gds_filename = gds_file.name
        # TODO(logansu): Change units to nanometers.
        self.gdsii = gdspy.GdsLibrary(units=1e-6, precision=1e-9)

        # Convert whatever units are in the GDS file to um/nm.
        self.gdsii.read_gds(gds_file, units="convert")

        self.top_level_cell = self._get_top_level_cell(cell_name)
        self.layers = self._extract_layers()

    def _extract_layers(self) -> Dict[Tuple[int, int], gdspy.PolygonSet]:
        """Extracts layers from the top level cell.

        The layers are then stored in a dictionary whose keys are in the form a
        tuple (layer, datatype).

        Note that we take the layer/datatype name to be that of the first value
        of the list. This doesn't seem to be a hack since when creating a
        ``PolygonSet object``, you only specify a single number for the layer or
        datatype.

        Returns:
            Dictionary whose keys corespond to the tuples (layer, datatype) and
            elements are ``PolygonSets`` which hold all the polygons contained
            in the layer.
        """

        layers = {}
        for poly in self.top_level_cell.get_polygonsets():
            key = (poly.layers[0], poly.datatypes[0])
            if key not in layers:
                layers[key] = []
            layers[key] += poly.polygons

        return layers

    def _get_top_level_cell(self, cell_name) -> gdspy.Cell:
        """Extracts the top level cell.

        We choose the top level cell to be the cell which contains the most
        number of polygons.

        To this end, we simply go over all the top level cells produced through
        the gdspy function call ``gdsii_library.top_level`` and choose the one
        with the most polygons to call THE (singular) top level cell. We do
        ignore cell whose names begin with "$$$" as they are hidden cells.

        Args:
            cell_name: Optional arguement to directly specify which cell will
                be the top level cell.

        Returns:
            The cell which contains the most number of polygons.
        """
        # Deep copy so we don't permanently flatten the cells in our class
        top_level_cells = copy.deepcopy(self.gdsii.top_level())

        if cell_name is None:
            max_num_polys = 0
            for cell in top_level_cells:
                # Ignore cells that are hidden
                if cell.name.startswith("$$$"):
                    continue
                else:
                    num_polys = 0
                    cell.flatten()

                    polygon_sets = cell.get_polygonsets()
                    for polygon_set in polygon_sets:
                        num_polys += len(polygon_set.polygons)

                    if max_num_polys == num_polys:
                        raise ValueError(
                            "Multiple top level cells with same number of "
                            "polygons. Unable to uniquely identify top level"
                            " cell. Please specify cell by name instead.")
                    if max_num_polys < num_polys:
                        max_num_polys = num_polys
                        max_poly_tlc = cell

            if max_num_polys > 0:
                return max_poly_tlc
            else:
                raise ValueError(
                    "No valid cell found. Please specify cell by name instead.")
        else:
            for cell in top_level_cells:
                if cell.name == cell_name:
                    return cell.flatten()
            raise ValueError("Cell name not found, got {}.".format(cell_name))

    def get_bounding_box(self, polygon_coords):
        """Returns a NamedTuple which describes the bounding box of a polygon.

        Function takes in a list of tuples representing the 2D coordinates of a
        polygon and outputs a namedtuple which provides information on the
        x-limits, y-limits, and center coordinate of the bounding box of the
        polygon.

        Args:
            polygon_coords: List of tuples containing the coordinates of a
                polygon.

        Returns:
            ``box``: Named tuple which contains information on the x lower and
                upper bounds, ``box.x_minmax``, y lower and upper bounds,
                ``box.y_minmax``, and the coordinates of the center,
                ``box.center``. These quanties are in the form of Lists.
        """
        np_poly_coords = np.array(polygon_coords)

        x_max = max(np_poly_coords[:, 0]) * 1000
        x_min = min(np_poly_coords[:, 0]) * 1000

        y_max = max(np_poly_coords[:, 1]) * 1000
        y_min = min(np_poly_coords[:, 1]) * 1000

        box = Box(x_minmax=[x_min, x_max], y_minmax=[y_min, y_max])

        return box

    def get_bounding_boxes(self, layer):
        """Returns a list containing the bounding box for polygons in layer.

        Args:
            ``layer``: Layer which the bounding boxes of polygons is wanted.
                Layer specification is done by passing a tuple containing
                ``(layer numer, datatype)``.

        Returns:
            ``boxes``: A list containing Box NamedTuples. See documentation for
                ``get_bounding_box`` for more details.

        """

        boxes = []
        if layer not in self.layers:
            return []

        for polygon in self.layers[layer]:
            boxes.append(self.get_bounding_box(polygon))

        return boxes

    def get_polygons(self, layer):
        """Returns a list of all polygons in the specified layer.

        Args:
            ``layer``: Layer which the bounding boxes of polygons is wanted.
                Layer specification is done by passing a tuple containing
                ``(layer numer, datatype)``.

        Returns:
            ``List[List]``: Returns a list containing lists of coordinates for
                each polygon in the specified ``layer``.

        Raises:
            ValueError: If layer cannot be found.
        """
        if layer not in self.layers:
            return []
        return self.layers[layer]
