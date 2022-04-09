from typing import List, Tuple, Callable

import gdspy
import numpy as np
from numpy import diff, floor, ceil, zeros, hstack, newaxis

import pickle
import warnings
import copy

from spins.gridlock.float_raster import raster_1D, raster_2D
from spins.gridlock import GridError, Direction, GridType
from spins.gridlock._helpers import is_scalar


class Grid:
    """
    Simulation grid generator intended for electromagnetic simulations.
    Can be used to generate non-uniform rectangular grids (the entire grid
    is generated based on the coordinates of the boundary points). Also does
    straightforward natural <-> grid unit conversion.

    The Grid object must be specified with shifts that generate the primary grid and a
    complementary grid from grid specified by the coordinates of the boundary points. In the
    context of EM simulations, the primary grid is the E-field grid and the complementary
    grid is the H-field grid. More formally, the primary grid should have vertices at the
    body-centers of the complementary grid and vice-versa. This relationship between the
    primary and complementary grid is assumed while aliasing the drawn structures onto the grid

    Objects on the grid can be drawn via the draw_ functions (e.g. draw_cuboid, draw_cylinder,
    draw_slab). Once all the objects have been drawn on the grid, the render() function can be
    called to raster the drawn objects on the grid. It is assumed that the object drawn latest is
    the correct object and should replace any of the older objects being drawn in case of an intersection
    with the older objects.

    self.grids[i][a,b,c] contains the value of epsilon for the cell located at
          (xyz[0][a]+dxyz[0][a]*shifts[i, 0],
           xyz[1][b]+dxyz[1][b]*shifts[i, 1],
           xyz[2][c]+dxyz[2][c]*shifts[i, 2]).
    You can get raw edge coordinates (exyz),
                   center coordinates (xyz),
                           cell sizes (dxyz),
    from the properties named as above, or get them for a given grid by using the
    self.shifted_*xyz(which_shifts) functions.

    It is tricky to determine the size of the right-most cell after shifting,
    since its right boundary should shift by shifts[i][a] * dxyz[a][dxyz[a].size],
    where the dxyz element refers to a cell that does not exist.
    Because of this, we either assume this 'ghost' cell is the same size as the last
    real cell, or, if self.periodic[a] is set to True, the same size as the first cell.
    """

    # Intended for use as static constants
    Yee_Shifts_E = 0.5 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  dtype=float)  # type: np.ndarray
    Yee_Shifts_H = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                                  dtype=float)  # type: np.ndarray

    @property
    def dxyz(self) -> List[np.ndarray]:
        """
        Cell sizes for each axis, no shifts applied

        :return: List of 3 ndarrays of cell sizes
        """
        return [diff(self.exyz[a]) for a in range(3)]

    @property
    def xyz(self) -> List[np.ndarray]:
        """
        Cell centers for each axis, no shifts applied

        :return: List of 3 ndarrays of cell edges
        """
        return [self.exyz[a][:-1] + self.dxyz[a] / 2.0 for a in range(3)]

    @property
    def shape(self) -> np.ndarray:
        """
        The number of cells in x, y, and z

        :return: ndarray [x_centers.size, y_centers.size, z_centers.size]
        """
        # Substract one because we keep track of edges.
        return np.array([coord.size - 1 for coord in self.exyz], dtype=int)

    @property
    def dxyz_with_ghost(self) -> List[np.ndarray]:
        """
        Gives dxyz with an additional 'ghost' cell at the end, whose value depends
         on whether or not the axis has periodic boundary conditions. See main description
         above to learn why this is necessary.

         If periodic, final edge shifts same amount as first
         Otherwise, final edge shifts same amount as second-to-last

        :return: list of [dxs, dys, dzs] with each element same length as elements of self.xyz
        """
        el = [0 if p else -1 for p in self.periodic]
        return [
            hstack((self.dxyz[a], self.dxyz[a][e]))
            for a, e in zip(range(3), el)
        ]

    @property
    def center(self) -> np.ndarray:
        """
        Center position of the entire grid, no shifts applied
        :return: ndarray [x_center, y_center, z_center]
        """
        # center is just average of first and last xyz, which is just the average of the
        #  first two and last two exyz
        centers = [(self.exyz[a][0] + self.exyz[a][1] + self.exyz[a][-2] +
                    self.exyz[a][-1]) / 4.0 for a in range(3)]
        return np.array(centers, dtype=float)

    @property
    def dxyz_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the minimum and maximum cell size for each axis, as a tuple of two 3-element
         ndarrays. No shifts are applied, so these are extreme bounds on these values (as a
         weighted average is performed when shifting).

        :return: List of 2 ndarrays, d_min=[min(dx), min(dy), min(dz)] and d_max=[...]
        """
        d_min = np.array([min(self.dxyz[a]) for a in range(3)], dtype=float)
        d_max = np.array([max(self.dxyz[a]) for a in range(3)], dtype=float)
        return d_min, d_max

    def shifted_exyz(self,
                     which_shifts: int or None,
                     which_grid: GridType = GridType.PRIM) -> List[np.ndarray]:
        """
        Returns edges for which_shifts.

        :param which_shifts: Which grid (which shifts) to use, or None for unshifted
        :param which_grid: GridType.PRIM for the primary grid and GRIDTYPE.COMP for the complementary grid
        :return: List of 3 ndarrays of cell edges
        """
        if which_shifts is None:
            return self.exyz
        dxyz = self.dxyz_with_ghost
        if which_grid.value == 0:
            shifts = self.shifts[which_shifts, :]
            sexyz = [self.exyz[a] + dxyz[a] * shifts[a] for a in range(3)]
        else:
            shifts = self.comp_shifts[which_shifts, :]
            # Adding ghost cell to the beginning if the complementary grid
            sexyz = [
                np.append(self.exyz[a][0] - dxyz[a][-1] * shifts[a],
                          self.exyz[a] + dxyz[a] * shifts[a]) for a in range(3)
            ]
            # Removing the ghost cell if the compelementary grid is not shifted in a particular direction
            sexyz = [
                sexyz[a][1:] if shifts[a] == 0 else sexyz[a] for a in range(3)
            ]
        return sexyz

    def shifted_dxyz(self,
                     which_shifts: int or None,
                     which_grid: GridType = GridType.PRIM) -> List[np.ndarray]:
        """
        Returns cell sizes for which_shifts.

        :param which_shifts: Which grid (which shifts) to use, or None for unshifted
        :param which_grid: GridType.PRIM for the primary grid and GridType.COMP for complementary grid
        :return: List of 3 ndarrays of cell sizes
        """
        if which_shifts is None:
            return self.dxyz

        dxyz = self.dxyz_with_ghost
        if which_grid.value == 0:
            shifts = self.shifts[which_shifts, :]
            sdxyz = [(dxyz[a][:-1] * (1 - shifts[a]) + dxyz[a][1:] * shifts[a])
                     for a in range(3)]
        else:
            shifts = self.comp_shifts[which_shifts, :]
            # Adding ghost cell to the beginning of the complementary grid
            sdxyz = [
                np.append(
                    dxyz[a][-1] * (1 - shifts[a]) + dxyz[a][0] * shifts[a],
                    dxyz[a][:-1] * (1 - shifts[a]) + dxyz[a][1:] * shifts[a])
                for a in range(3)
            ]
            # Removing the ghost cell if the complementary grid is not shifted in a particular direction
            sdxyz = [
                sdxyz[a][1:] if shifts[a] == 0 else sdxyz[a] for a in range(3)
            ]
        return sdxyz

    def shifted_xyz(self,
                    which_shifts: int or None,
                    which_grid: GridType = GridType.PRIM) -> List[np.ndarray]:
        """
        Returns cell centers for which_shifts.

        :param which_shifts: Which grid (which shifts) to use, or None for unshifted
        :which_grid: GridType.PRIM for the primary grid and GridType.COMP for the complementary grid
        :return: List of 3 ndarrays of cell centers
        """
        if which_shifts is None:
            return self.xyz
        exyz = self.shifted_exyz(which_shifts, which_grid)
        dxyz = self.shifted_dxyz(which_shifts, which_grid)
        return [exyz[a][:-1] + dxyz[a] / 2.0 for a in range(3)]

    def autoshifted_dxyz(self):
        """
        Return cell widths, with each dimension shifted by the corresponding shifts.

        :return: [grid.shifted_dxyz(which_shifts=a)[a] for a in range(3)]
        """

        return [
            self.shifted_dxyz((a + 1) % 3, GridType.COMP)[a][:-1]
            for a in range(3)
        ]

    def ind2pos(self,
                ind: np.ndarray or List,
                which_shifts: int or None,
                which_grid: GridType = GridType.PRIM,
                round_ind: bool = True,
                check_bounds: bool = True) -> np.ndarray:
        """
        Returns the natural position corresponding to the specified indices.
         The resulting position is clipped to the bounds of the grid
        (to cell centers if round_ind=True, or cell outer edges if round_ind=False)

        :param ind: Indices of the position. Can be fractional. (3-element ndarray or list)
        :param which_shifts: which grid number (shifts) to use
        :param round_ind: Whether to round ind to the nearest integer position before indexing
                (default True)
        :param check_bounds: Whether to raise an GridError if the provided ind is outside of
                the grid, as defined above (centers if round_ind, else edges) (default True)
        :return: 3-element ndarray specifying the natural position
        :raises: GridError
        """

        if which_shifts is not None and which_shifts >= self.shifts.shape[0]:
            raise GridError('Invalid shifts')

        ind = np.array(ind, dtype=float)

        if check_bounds:
            if round_ind:
                low_bound = 0.0
                high_bound = -1
            else:
                low_bound = -0.5
                high_bound = -0.5
            if (ind < low_bound).any() or (ind > self.shape - high_bound).any():
                raise GridError('Position outside of grid: {}'.format(ind))

        if round_ind:
            rind = np.clip(np.round(ind), 0, self.shape - 1)
            sxyz = self.shifted_xyz(which_shifts, which_grid)
            position = [sxyz[a][rind[a]].astype(int) for a in range(3)]
        else:
            sexyz = self.shifted_exyz(which_shifts, which_grid)
            position = [
                np.interp(ind[a],
                          np.arange(sexyz[a].size) - 0.5, sexyz[a])
                for a in range(3)
            ]
        return np.array(position, dtype=float)

    def pos2ind(self,
                r: np.ndarray or List,
                which_shifts: int or None,
                which_grid: GridType = GridType.PRIM,
                round_ind: bool = True,
                check_bounds: bool = True) -> np.ndarray:
        """
        Returns the indices corresponding to the specified natural position.
             The resulting position is clipped to within the outer centers of the grid.

        :param r: Natural position that we will convert into indices (3-element ndarray or list)
        :param which_shifts: which grid number (shifts) to use
        :param round_ind: Whether to round the returned indices to the nearest integers.
        :param check_bounds: Whether to throw an GridError if r is outside the grid edges
        :return: 3-element ndarray specifying the indices
        :raises: GridError
        """
        r = np.squeeze(r)
        if r.size != 3:
            raise GridError('r must be 3-element vector: {}'.format(r))

        if (which_shifts is not None) and (which_shifts >=
                                           self.shifts.shape[0]):
            raise GridError('')

        sexyz = self.shifted_exyz(which_shifts, which_grid)

        if check_bounds:
            for a in range(3):
                if self.shape[a] > 1 and (r[a] < sexyz[a][0] or
                                          r[a] > sexyz[a][-1]):
                    raise GridError('Position[{}] outside of grid!'.format(a))

        grid_pos = zeros((3,))
        for a in range(3):
            xi = np.digitize(r[a],
                             sexyz[a]) - 1  # Figure out which cell we're in
            xi_clipped = np.clip(
                xi, 0, sexyz[a].size - 2)  # Clip back into grid bounds

            # No need to interpolate if round_ind is true or we were outside the grid
            if round_ind or xi != xi_clipped:
                grid_pos[a] = xi_clipped
            else:
                # Interpolate
                x = self.shifted_exyz(which_shifts, which_grid)[a][xi]
                dx = self.shifted_dxyz(which_shifts, which_grid)[a][xi]
                f = (r[a] - x) / dx
                # Clip to centers
                grid_pos[a] = np.clip(xi + f, 0, sexyz[a].size - 1)
        return grid_pos

    def coord2ind(self,
                  r: float,
                  axis: Direction,
                  which_shifts: int,
                  which_grid: GridType = GridType.PRIM,
                  round_ind: bool = True,
                  check_bounds: bool = True):
        '''
        Converts a single coordinate to index
        '''
        point_3D = np.array(
            [r if a == axis.value else self.center[a] for a in range(3)])
        ind_3D = self.pos2ind(point_3D, which_shifts, which_grid, round_ind,
                              check_bounds)
        return ind_3D[axis.value]

    def __init__(self,
                 pixel_edge_coordinates: List[np.ndarray],
                 ext_dir: Direction = Direction.z,
                 shifts: np.ndarray or List = Yee_Shifts_E,
                 comp_shifts: np.ndarray or List = Yee_Shifts_H,
                 initial: float or np.ndarray or List[float] or
                 List[np.ndarray] = (1.0,) * 3,
                 num_grids: int = None,
                 periodic: bool or List[bool] = False):

        # Backgrdound permittivity and fraction of background permittivity in the grid
        self.grids_bg = []  # type: List[np.ndarray]
        self.frac_bg = []  # type: List[np.ndarray]

        # [[x0 y0 z0], [x1 y1 z1], ...] offsets for primary grid 0,1,...
        self.exyz = [np.unique(pixel_edge_coordinates[i]) for i in range(3)]
        for i in range(3):
            if len(self.exyz[i]) != len(pixel_edge_coordinates[i]):
                warnings.warn(
                    'Dimension {} had duplicate edge coordinates'.format(i))

        if is_scalar(periodic):
            self.periodic = [periodic] * 3
        else:
            self.periodic = [False] * 3

        self.shifts = np.array(shifts, dtype=float)
        self.comp_shifts = np.array(comp_shifts, dtype=float)
        if self.shifts.shape[1] != 3:
            GridError(
                'Misshapen shifts on the primary grid; second axis size should be 3,'
                ' shape is {}'.format(self.shifts.shape))
        if self.comp_shifts.shape[1] != 3:
            GridError(
                'Misshapen shifts on the complementary grid: second axis size should be 3,'
                ' shape is {}'.format(self.comp_shifts.shape))
        if self.comp_shifts.shape[0] != self.shifts.shape[0]:
            GridError(
                'Inconsistent number of shifts in the primary and complementary grid'
            )
        if not ((self.shifts >= 0).all() and (self.comp_shifts >= 0).all()):
            GridError(
                'Shifts are required to be non-negative for both primary and complementary grid'
            )

        num_shifts = self.shifts.shape[0]
        if num_grids is None:
            num_grids = num_shifts
        elif num_grids > num_shifts:
            raise GridError(
                'Number of grids exceeds number of shifts (%u)' % num_shifts)

        grids_shape = hstack((num_grids, self.shape))
        if is_scalar(initial):
            self.grids_bg = np.full(grids_shape, initial, dtype=complex)
        else:
            if len(initial) < num_grids:
                raise GridError('Too few initial grids specified!')

            self.grids_bg = [None] * num_grids
            for i in range(num_grids):
                if is_scalar(initial[i]):
                    if initial[i] is not None:
                        self.grids_bg[i] = np.full(
                            self.shape, initial[i], dtype=complex)
                else:
                    if not np.array_equal(initial[i].shape, self.shape):
                        raise GridError(
                            'Initial grid sizes must match given coordinates')
                    self.grids_bg[i] = initial[i]

        if isinstance(ext_dir, Direction):
            self.ext_dir = ext_dir.value
        elif is_scalar(ext_dir):
            if ext_dir in range(3):
                self.ext_dir = ext_dir
            else:
                raise GridError('Invalid extrusion direction')
        else:
            raise GridError('Invalid extrusion direction')

        self.grids = np.full(
            grids_shape, 0.0,
            dtype=complex)  # contains the rendering of objects on the grid
        self.frac_bg = np.full(
            grids_shape, 1.0,
            dtype=complex)  # contains the fraction of background permittivity
        self.planar_dir = np.delete(range(3), self.ext_dir)
        self.list_polygons = [
        ]  # List of polygons corresponding to each block specified by user
        self.layer_polygons = [
        ]  # List of polygons after bifurcating extrusion direction into layers
        self.reduced_layer_polygons = [
        ]  # List of polygons after removing intersections

        self.list_z = [
        ]  # List of z coordinates of the different blocks specified by user
        self.layer_z = [
        ]  # List of z coordinates of the different distinct layers

    @staticmethod
    def load(filename: str) -> 'Grid':
        """
        Load a grid from a file

        :param filename: Filename to load from.
        """
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        g = Grid([[-1, 1]] * 3)
        g.__dict__.update(tmp_dict)
        return g

    def save(self, filename: str):
        """
        Save to file.

        :param filename: Filename to save to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=2)

    def copy(self):
        """
        Return a deep copy of the grid.

        :return: Deep copy of the grid.
        """
        return copy.deepcopy(self)

    def draw_polygon(self, center: np.ndarray, polygon: np.ndarray,
                     thickness: float, eps: float or List[float]):
        """
        Draws a polygon with coordinates in polygon and thickness
        Note on order of coordinates in polygon -
        If ext_dir = x, then polygon has coordinates of form (y,z)
           ext_dir = y, then polygon has coordinates of form (x,y)
           ext_dir = z, then polygon has coordinates of form (x,y)
        """

        center = np.array(center)
        polygon = np.array(polygon)
        # Validating input arguments
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise GridError(
                'Invalid format for specifying polygon - must be a Nx2 array')

        if polygon.shape[0] <= 2:
            raise GridError(
                'Malformed Polygon - must contain more than two points')

        if center.ndim != 1 or center.size != 3:
            raise GridError('Invalid format for the polygon center')

        if (not is_scalar(thickness)) or thickness <= 0:
            raise GridError('Invalid thickness')

        if is_scalar(eps):
            eps = np.ones(self.shifts.shape[0]) * eps
        elif eps.ndim != 1 and eps.size != self.shifts.shape[0]:
            raise GridErro(
                'Invalid permittivity - must be scalar or vector of length equalling number of grids'
            )
        # Translating polygon by its center
        polygon_translated = polygon + np.tile(center[self.planar_dir],
                                               (polygon.shape[0], 1))
        self.list_polygons.append((polygon_translated, eps))

        # Adding the z-coordinates of the z-coordinates of the added layers
        self.list_z.append([
            center[self.ext_dir] - 0.5 * thickness,
            center[self.ext_dir] + 0.5 * thickness
        ])

    def draw_cuboid(self, center: np.ndarray, extent: np.ndarray, eps: float or
                    List[float]):
        """
        Draw a cuboid with permittivity epsilon
        """

        center = np.array(center)
        extent = np.array(extent)
        # Validating input parameters
        if center.ndim != 1 or center.size != 3:
            raise GridError('Invalid center coordinate')

        if extent.ndim != 1 or extent.size != 3:
            raise GridError('Invalid cuboid lengths')

        if is_scalar(eps):
            eps = np.ones(self.shifts.shape[0]) * eps
        if eps.ndim != 1 or eps.size != self.shifts.shape[0]:
            raise GridError(
                'Invalid permittivity - must be scalar or vector of length equalling number of grids'
            )

        # Calculating the polygon corresponding to the drawn cuboid
        polygon = 0.5 * np.array(
            [[-extent[self.planar_dir[0]], extent[self.planar_dir[1]]],
             [extent[self.planar_dir[0]], extent[self.planar_dir[1]]],
             [extent[self.planar_dir[0]], -extent[self.planar_dir[1]]],
             [-extent[self.planar_dir[0]], -extent[self.planar_dir[1]]]],
            dtype=float)

        thickness = extent[self.ext_dir]

        # Drawing polygon
        self.draw_polygon(center, polygon, thickness, eps)

    def draw_cylinder(self, center: np.ndarray, radius: float, thickness: float,
                      num_points: int, eps: float or np.ndarray):
        """
        Draw a cylinder with permittivity epsilon. By default, the axis of the cylinder
        is assumed to be along the extrusion direction

        """
        center = np.array(center)
        # Validating input parameters
        if center.ndim != 1 or center.size != 3:
            raise GridError('Invalid center coordinate')

        if is_scalar(eps):
            eps = np.ones(self.shifts.shape[0]) * eps

        if eps.ndim != 1 or eps.size != self.shifts.shape[0]:
            raise GridError(
                'Invalid permittvity - must be scalar or vector of length equalling number of grids'
            )

        if not is_scalar(thickness):
            raise GridError('Invalid thickness')

        if not is_scalar(num_points):
            raise GridError('Invalid number of points on the cylinder')

        # Approximating the drawn cylinder with a polygon with number of vertices = num_points
        theta = np.linspace(0, 2.0 * np.pi, num_points)
        x = radius * np.sin(theta)
        y = radius * np.cos(theta)
        polygon = np.vstack((x, y)).T

        # Drawing polygon
        self.draw_polygon(center, polygon, thickness, eps)

    def draw_slab(self, dir_slab: Direction or float, center: float,
                  thickness: float, eps: float or np.ndarray):
        """
        Draw a slab
        """

        # Validating input arguments
        if isinstance(dir_slab, Direction):
            dir_slab = dir_slab.value
        elif not is_scalar(dir_slab):
            raise GridError('Invalid slab direction')
        elif not dir_slab in range(3):
            raise GridError('Invalid slab direction')

        if not is_scalar(center):
            raise GridError('Invalid slab center')

        if is_scalar(eps):
            eps = np.ones(self.shifts.shape[0]) * eps

        if eps.ndim != 1 or eps.size != self.shifts.shape[0]:
            raise GridError(
                'Invalid permittivity - must be a scalar or vector of length equalling number of grids'
            )

        dir_slab_par = np.delete(range(3), dir_slab)
        cuboid_cen = np.array(
            [self.center[a] if a != dir_slab else center for a in range(3)])
        cuboid_extent = np.array([2*np.abs(self.exyz[a][-1]-self.exyz[a][0]) if a !=dir_slab \
                                  else thickness for a in range(3)])
        self.draw_cuboid(cuboid_cen, cuboid_extent, eps)

    def fill_cuboid(self, fill_dir: Direction, fill_pol: int,
                    surf_center: np.ndarray, surf_extent: np.ndarray,
                    eps: float or np.ndarray):
        '''
        INPUTS:
        1. surf_extent - array of size 2 corresponding to the extent of the surface. If the fill direction
        is x, then the two elements correspond to y,z, if it is y then x,z and if it is z then x,y
        '''
        surf_center = np.array(surf_center)
        surf_extent = np.array(surf_extent)

        # Validating input arguments
        if isinstance(fill_dir, Direction):
            fill_dir = fill_dir.value
        elif not is_scalar(fill_dir):
            raise GridError('Invalid slab direction')
        elif not dir_slab in range(3):
            raise GridError('Invalid slab direction')

        if not is_scalar(fill_pol):
            raise GridError('Invalid polarity')
        if not fill_pol in [-1, 1]:
            raise GridError('Invalid polarity')

        if surf_center.ndim != 1 or surf_center.size != 3:
            raise GridError('Invalid surface center')

        if surf_extent.ndim != 1 or surf_extent.size != 2:
            raise GridError('Invalid surface extent')

        edge_lim = self.exyz[fill_dir][0] if fill_pol == -1 else self.exyz[
            fill_dir][-1]
        cuboid_extent = np.insert(surf_extent, fill_dir,
                                  2 * np.abs(edge_lim - surf_center[fill_dir]))


        cuboid_center = np.array([surf_center[a] if a != fill_dir else \
                                     (surf_center[a]+0.5*fill_pol*cuboid_extent[a]) for a in range(3)])

        self.draw_cuboid(cuboid_center, cuboid_extent, eps)

    def fill_slab(self, fill_dir: Direction, fill_pol: int, surf_center: float,
                  eps: float or np.ndarray):

        # Validating input arguments
        if isinstance(fill_dir, Direction):
            fill_dir = fill_dir.value
        elif not is_scalar(fill_dir):
            raise GridError('Invalid slab direction')
        elif not dir_slab in range(3):
            raise GridError('Invalid slab direction')

        if not is_scalar(fill_pol):
            raise GridError('Invalid polarity')
        if not fill_pol in [-1, 1]:
            raise GridError('Invalid polarity')

        if not is_scalar(surf_center):
            raise GridError('Invalid surface center')

        edge_lim = self.exyz[fill_dir][0] if fill_pol == -1 else self.exyz[
            fill_dir][-1]
        slab_thickness = 2 * np.abs(edge_lim - surf_center)
        slab_center = surf_center + 0.5 * fill_pol * slab_thickness
        self.draw_slab(fill_dir, slab_center, slab_thickness, eps)

    def compute_layers(self):
        """
        Function to break the structure into different layers

        OUTPUT: Takes the set of polygons, which may be drawn at displaced z coordinates
        and breaks them into layers which can then be seperately be rendered
        """

        # Calculating the layer coordinates
        self.layer_z = np.sort(np.unique(np.array(self.list_z).flatten('F')))
        self.layer_polygons = [[] for i in range(self.layer_z.size - 1)]

        # Assigning polynomials into layers
        for i in range(len(self.list_polygons)):
            ind_bottom = np.searchsorted(self.layer_z, self.list_z[i][0])
            ind_top = np.searchsorted(self.layer_z, self.list_z[i][1])
            for k in range(ind_bottom, ind_top):
                self.layer_polygons[k].append(self.list_polygons[i])

    def remove_intersection(self):
        """
        Function to remove polygon intersections
        We assume that the material drawn at the end is the desired material

        OUTPUT: Converts the set of objects specified by the user into another
        set of objects which do NOT intersect with each other
        """

        def check_bounding_box(polygon_1, polygon_2):
            '''
            Helper function to perform a simple check if the bounding box of
            polygon_1 and polygon_2 do not intersect

            This is mainly to avoid computing intersections if the two polygons
            are very far from each other, in order to speed up the reduction process
            '''
            r1_max = np.max(polygon_1, axis=0)
            r2_max = np.max(polygon_2, axis=0)
            r1_min = np.min(polygon_1, axis=0)
            r2_min = np.min(polygon_2, axis=0)

            if r1_max[0] < r2_min[0] or r2_max[0] < r2_min[0]:
                return False
            elif r1_max[1] < r2_min[1] or r2_max[1] < r2_min[1]:
                return False
            else:
                return True

        def compute_intersection(polygon_1, polygon_2):
            '''
            Wrapper function around the gdspy module to take as input
            two polygons and return polygon_1-polygon_2
            Explicit NOT operation is only performed if the bounding boxes
            of the two polygons do not intersect.
            '''
            if check_bounding_box(polygon_1, polygon_2):
                gds_poly1 = gdspy.Polygon(polygon_1, 0)
                gds_poly2 = gdspy.Polygon(polygon_2, 0)
                gds_poly = gdspy.fast_boolean(
                    gds_poly1, gds_poly2, 'not', layer=1)
                if gds_poly is None:
                    return []
                else:
                    return gds_poly.polygons

            else:
                return [polygon_1]

        num_layers = len(self.layer_polygons)
        self.reduced_layer_polygons = []
        for layer_i_polygons in self.layer_polygons:
            # In each layer we remove the polygons added later from the
            # polygons added earlier
            if layer_i_polygons:
                red_layer_i_polygons = [layer_i_polygons[0]]
                num_polygons = len(layer_i_polygons)
                for n in range(1, num_polygons):
                    temp_layer_i_polygons = []
                    for red_layer_i_polygon in red_layer_i_polygons:
                        polygon_inter = compute_intersection(
                            red_layer_i_polygon[0], layer_i_polygons[n][0])
                        if polygon_inter:
                            polygon_inter_eps = [(polygon,
                                                  red_layer_i_polygon[1])
                                                 for polygon in polygon_inter]
                            temp_layer_i_polygons = temp_layer_i_polygons + polygon_inter_eps

                    temp_layer_i_polygons.append(layer_i_polygons[n])
                    red_layer_i_polygons = copy.deepcopy(temp_layer_i_polygons)
            else:
                red_layer_i_polygons = []
            self.reduced_layer_polygons.append(
                copy.deepcopy(red_layer_i_polygons))

    def render_polygon(self, polygon: np.ndarray, z_extent: np.ndarray,
                       eps: np.ndarray):
        """
        Function to render grid with contribution due to polygon 'polygon'.

        INPUTS:
        polygon - list of (x,y) vertices of the polygon being rendered
        z_extent - extent (z_1, z_2) along the extrusion direction of the polygon being rendered
        eps - permittivity of the polygon being rendered

        OUTPUTS:
        updates self.grids with the properly aliased polygon permittivity
        reduces self.frac_bg by the fraction of space occupied by the polygon 'polygon'
        """

        # Validating input arguments
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise GridError(
                'Invalid format for specifying polygon - must be a Nx2 array')
        if polygon.shape[0] <= 2:
            raise GridError(
                'Malformed Polygon - must contain more than two points')
        if z_extent.ndim != 1 or z_extent.size != 2:
            raise GridError(
                'Invalid format for specifying z-extent - must be a vector of length 2'
            )

        def to_3D(vector: List or np.ndarray,
                  z: float = 0.5 * (z_extent[0] + z_extent[1])):
            return np.insert(vector, self.ext_dir, z)

        def get_zi(z: float, which_shifts: float):
            pos_3D = to_3D([0, 0], z)
            grid_coords = self.pos2ind(
                pos_3D,
                which_shifts,
                which_grid=GridType.PRIM,
                check_bounds=False)
            return grid_coords[self.ext_dir]

        # Calculating slice affected by polygon
        pbd_min = polygon.min(axis=0)
        pbd_max = polygon.max(axis=0)
        z_min = z_extent.min()
        z_max = z_extent.max()

        for n, grid in enumerate(self.grids):
            '''
            Computing the in-plane pixel values
            '''
            # Shape of the complementary grid
            comp_shape = np.array([self.shape[a]+2 if self.comp_shifts[n][a] == 0 else self.shape[a]+1 \
                          for a in range(3)])

            # Calculating the indices of the maximum and minimum polygon coordinate
            ind_xy_min = self.pos2ind(
                to_3D(pbd_min),
                which_shifts=n,
                which_grid=GridType.PRIM,
                round_ind=True,
                check_bounds=False)

            ind_xy_max = self.pos2ind(
                to_3D(pbd_max),
                which_shifts=n,
                which_grid=GridType.PRIM,
                round_ind=True,
                check_bounds=False)

            # Calculating the points on the grid that are affected by the drawn polygons
            corner_xy_min = ind_xy_min[self.planar_dir].astype(int)
            corner_xy_max = np.minimum(
                ind_xy_max[self.planar_dir] + 1,
                self.shape[self.planar_dir] - 1).astype(int)

            # Calculating the points of the complementary grid that need to be passed
            comp_corner_xy_min = corner_xy_min.astype(int)
            comp_corner_xy_max = np.minimum(
                corner_xy_max + 1, comp_shape[self.planar_dir] - 1).astype(int)

            # Setting up slices
            edge_slice_xy = [
                np.s_[j:f + 1]
                for j, f in zip(comp_corner_xy_min, comp_corner_xy_max)
            ]

            # Calling the rastering function
            aa_x, aa_y = (self.shifted_exyz(which_shifts = n, which_grid = GridType.COMP)[a][s] \
                          for a,s in zip(self.planar_dir, edge_slice_xy))
            w_xy = raster_2D(polygon.T, aa_x, aa_y)
            '''
            Computing the pixel value along the surface normal
            '''
            # Calculating the indices of the start and stop point
            ind_z_min = get_zi(z_min, which_shifts=n)
            ind_z_max = get_zi(z_max, which_shifts=n)
            corner_z_min = ind_z_min.astype(int)
            corner_z_max = np.minimum(ind_z_max + 1,
                                      self.shape[self.ext_dir] - 1).astype(int)
            comp_corner_z_min = corner_z_min.astype(int)
            comp_corner_z_max = np.minimum(
                corner_z_max + 1, comp_shape[self.ext_dir] - 1).astype(int)

            edge_slice_z = np.s_[comp_corner_z_min:comp_corner_z_max + 1]
            aa_z = self.shifted_exyz(
                which_shifts=n,
                which_grid=GridType.COMP)[self.ext_dir][edge_slice_z]
            w_z = raster_1D(z_extent, aa_z)

            # Combining the extrusion and planar area calculation
            w = (w_xy[:, :, np.newaxis] * w_z).transpose(
                np.insert([0, 1], self.ext_dir, (2,)))

            # Adding to the grid
            center_slice = [None for a in range(3)]
            center_slice[self.ext_dir] = np.s_[corner_z_min:corner_z_max + 1]
            for i in range(2):
                center_slice[self.planar_dir[i]] = np.s_[corner_xy_min[i]:
                                                         corner_xy_max[i] + 1]

            # Updating permittivity
            self.grids[n][tuple(center_slice)] += eps[n] * w
            self.frac_bg[n][tuple(center_slice)] -= w

    def clear(self):
        '''
        Function to clear the existing polygons in the grid object
        Following the clear command, new structures can be added to the grid
        object and subsequently rendered
        '''
        self.list_polygons = [
        ]  # List of polygons corresponding to each block specified by user
        self.layer_polygons = [
        ]  # List of polygons after bifurcating extrusion direction into layers
        self.reduced_layer_polygons = [
        ]  # List of polygons after removing intersections

        self.list_z = [
        ]  # List of z coordinates of the different blocks specified by user
        self.layer_z = [
        ]  # List of z coordinates of the different distinct layers

    def render(self, disable_intersection: bool = False):
        """
        Function to render the added polygons to the specified grid

        INPUTS:
        1. disable_intersection - set this flag to True if you are
        sure that the polygons that you draw do not intersect with each other.
        The intersection removal process will not be performed, and direct rastering of
        the polygons onto the grid will be performed. Note that one polygon completely
        being inside the other counts as an intersection. This might speed up the
        drawing functions if you are drawing a large number of polygons (for e.g. in a
        photonic crystal)

        OUTPUTS: Renders all the drawn polygons onto the grid
        There are three steps to rendering the grid
        1. Computing the layers along the z-direction
        2. Simplify polygon intersection at each layer (done only if disable_intersection is False)
        3. Use the rastering functions to compute areas and add back background permittivities

        NOTE: The rendering function CAN be called more than once - if you draw a bunch of objects, render,
        visualize and do some operations on the resulting grid, and want to edit the grid by
        adding more objects, you can continue to add the polygons on the same object and
        render again.
        """

        # Begin by setting the grid.grids to 0 and frac_bg to 1 -
        # This handles the case if it is not the first call to the render function
        self.frac_bg = np.ones_like(self.grids_bg)
        self.grids = np.zeros_like(self.grids_bg)

        # Computing layers involved in the problem
        self.compute_layers()

        # Removing intersections involved in the problem
        if disable_intersection:
            self.reduced_layer_polygons = self.layer_polygons
        else:
            self.remove_intersection()

        # Now all the layers and polygons should not intersect with each other and can be aliased on the grids
        for i, polygons in enumerate(self.reduced_layer_polygons):
            for j, polygon in enumerate(self.reduced_layer_polygons[i]):
                # Iterating over each layer and rendering each polygon
                if polygon is not None:
                    self.render_polygon(
                        polygon[0],
                        z_extent=np.array(
                            [self.layer_z[i], self.layer_z[i + 1]]),
                        eps=polygon[1])

        # Finally, adding the background permittivity
        for i in range(0, self.shifts.shape[0]):
            self.grids[i] = self.grids[i] + self.grids_bg[i] * self.frac_bg[i]

    def get_slice(self,
                  surface_normal: Direction or int,
                  center: float,
                  which_shifts: int = 0,
                  sample_period: int = 1) -> np.ndarray:
        """
            Retrieve a slice of a grid.
            Interpolates if given a position between two planes.

            :param surface_normal: Axis normal to the plane we're displaying. Can be a Direction or
             integer in range(3)
            :param center: Scalar specifying position along surface_normal axis.
            :param which_shifts: Which grid to display. Default is the first grid (0).
            :param sample_period: Period for down-sampling the image. Default 1 (disabled)
            :return Array containing the portion of the grid.
        """
        if not is_scalar(center) and np.isreal(center):
            raise GridError('center must be a real scalar')

        sp = round(sample_period)
        if sp <= 0:
            raise GridError('sample_period must be positive')

        if not is_scalar(which_shifts) or which_shifts < 0:
            raise GridError('Invalid which_shifts')

        # Turn surface_normal into its integer representation
        if isinstance(surface_normal, Direction):
            surface_normal = surface_normal.value
        if surface_normal not in range(3):
            raise GridError('Invalid surface_normal direction')

        surface = np.delete(range(3), surface_normal)

        # Extract indices and weights of planes
        center3 = np.insert([0, 0], surface_normal, (center,))
        center_index = self.pos2ind(
            center3, which_shifts, round_ind=False,
            check_bounds=False)[surface_normal]
        centers = np.unique([floor(center_index),
                             ceil(center_index)]).astype(int)
        if len(centers) == 2:
            fpart = center_index - floor(center_index)
            w = [1 - fpart, fpart]  # longer distance -> less weight
        else:
            w = [1]

        c_min, c_max = (self.xyz[surface_normal][i] for i in [0, -1])
        if center < c_min or center > c_max:
            raise GridError(
                'Coordinate of selected plane must be within simulation domain')

        # Extract grid values from planes above and below visualized slice
        sliced_grid = zeros(self.shape[surface])
        for ci, weight in zip(centers, w):
            s = tuple(
                ci if a == surface_normal else np.s_[::sp] for a in range(3))
            sliced_grid += weight * self.grids[which_shifts][tuple(s)]

        # Remove extra dimensions
        sliced_grid = np.squeeze(sliced_grid)

        return sliced_grid

    def visualize_slice(self,
                        surface_normal: Direction or int,
                        center: float,
                        which_shifts: int = 0,
                        sample_period: int = 1,
                        finalize: bool = True):
        """
        Visualize a slice of a grid.
        Interpolates if given a position between two planes.

        :param surface_normal: Axis normal to the plane we're displaying. Can be a Direction or
         integer in range(3)
        :param center: Scalar specifying position along surface_normal axis.
        :param which_shifts: Which grid to display. Default is the first grid (0).
        :param sample_period: Period for down-sampling the image. Default 1 (disabled)
        :param finalize: Whether to call pyplot.show() after constructing the plot. Default True
        """
        from matplotlib import pyplot

        # Set surface normal to its integer value
        if isinstance(surface_normal, Direction):
            surface_normal = surface_normal.value

        grid_slice = self.get_slice(
            surface_normal=surface_normal,
            center=center,
            which_shifts=which_shifts,
            sample_period=sample_period)

        surface = np.delete(range(3), surface_normal)

        x, y = (self.shifted_exyz(which_shifts)[a] for a in surface)
        xmesh, ymesh = np.meshgrid(x, y, indexing='ij')
        x_label, y_label = ('xyz' [a] for a in surface)
        if (len(grid_slice.shape) == 1):
            grid_slice = np.transpose(np.array([grid_slice]))

        pyplot.figure()
        pyplot.pcolormesh(xmesh, ymesh, grid_slice)
        pyplot.colorbar()
        pyplot.gca().set_aspect('equal', adjustable='box')
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        if finalize:
            pyplot.show()

    def visualize_isosurface(self,
                             level: float = None,
                             which_shifts: int = 0,
                             sample_period: int = 1,
                             show_edges: bool = True,
                             finalize: bool = True):
        """
        Draw an isosurface plot of the device.

        :param level: Value at which to find isosurface. Default (None) uses mean value in grid.
        :param which_shifts: Which grid to display. Default is the first grid (0).
        :param sample_period: Period for down-sampling the image. Default 1 (disabled)
        :param show_edges: Whether to draw triangle edges. Default True
        :param finalize: Whether to call pyplot.show() after constructing the plot. Default True
        """
        from matplotlib import pyplot
        import skimage.measure
        # Claims to be unused, but needed for subplot(projection='3d')
        from mpl_toolkits.mplot3d import Axes3D

        # Get data from self.grids
        grid = self.grids[which_shifts][::sample_period, ::sample_period, ::
                                        sample_period]
        if level is None:
            level = grid.mean()

        # Find isosurface with marching cubes
        verts, faces = skimage.measure.marching_cubes(grid, level)

        # Convert vertices from index to position
        pos_verts = np.array([
            self.ind2pos(verts[i, :], which_shifts, round_ind=False)
            for i in range(verts.shape[0])
        ],
                             dtype=float)
        xs, ys, zs = (pos_verts[:, a] for a in range(3))

        # Draw the plot
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        if show_edges:
            ax.plot_trisurf(xs, ys, faces, zs)
        else:
            ax.plot_trisurf(xs, ys, faces, zs, edgecolor='none')

        # Add a fake plot of a cube to force the axes to be equal lengths
        max_range = np.array(
            [xs.max() - xs.min(),
             ys.max() - ys.min(),
             zs.max() - zs.min()],
            dtype=float).max()
        mg = np.mgrid[-1:2:2, -1:2:2, -1:2:2]
        xbs = 0.5 * max_range * mg[0].flatten() + 0.5 * (xs.max() + xs.min())
        ybs = 0.5 * max_range * mg[1].flatten() + 0.5 * (ys.max() + ys.min())
        zbs = 0.5 * max_range * mg[2].flatten() + 0.5 * (zs.max() + zs.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(xbs, ybs, zbs):
            ax.plot([xb], [yb], [zb], 'w')

        if finalize:
            pyplot.show()
