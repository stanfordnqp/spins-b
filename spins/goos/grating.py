"""Defines parametrizations for 1D grating couplers.

The most proper grating parametrization is the `BarcodeGrating` as it defines
the grating rigorously using boxes. However, because of the gradient
calculation is not yet implemented, it cannot be used to optimize a grating.

The `PixelatedGrating` renders the grating onto a grid using interpolation
so may be less accurate in terms of simulation quality. However, gradients are
defined, so it can be used to design a 1D grating structure. `PixelatedGrating`
supports fully-etched gratings, partially-etched gratings, and gratings with
multiple possible etch steps.

The `GratingL2D` action can be used to convert a grating parametrized by
`PixelatedContShape` into a `PixelatedGrating`. The function to
`discretize_to_pixelated_grating` is a wrapper for conveniently discretizing
and created a `PixelatedGrating` object. Both minimum and maximum feature sizes
can be imposed. Min/max features can depend on the etch-depth level.
"""
from typing import Callable, List, Optional, Tuple, Union

import copy
import numbers

import numpy as np
import scipy.linalg

from spins import goos
from spins.invdes.parametrization import Parametrization
from spins.invdes.parametrization.grating_parametrization import GratingParam


class PixelatedGrating(goos.Shape):
    """Serves an alternative for `BarcodeGrating`.

    `BarcodeGrating` generates a series of a boxes and properly differentiating
    through boxes that touch each other is a bit tricky. Here, we take a
    shortcut by utilizing our old barcode grating code that simply draws
    directly onto a grid.

    The grating can either be defined in terms of the widths of each grating
    segment or based on the locations of the edges. Which one to use is set by
    the `use_edge_locs` flags (`True` for edge locations). It seems that using
    widths as the parametrization leads to better optimization results.

    Each grating has an associated height fraction array that corresponds to
    the possible etch depths (heights) of each grating segment. Each grating
    segment, in turn, has an associated height index, which indexes into the
    height fraction array.
    """
    node_type = "goos.shape.pixelated_grating"

    def __init__(
            self,
            widths_or_edge_locs: goos.Function,
            height_index: goos.Function,
            height_fracs: np.ndarray,
            pos: np.ndarray,
            extents: np.ndarray,
            material: goos.material.Material,
            material2: goos.material.Material,
            grating_dir: int,
            grating_dir_spacing: float,
            etch_dir: int = 2,
            etch_dir_divs: int = 1,
            etch_polarity: int = 1,
            use_edge_locs: bool = False,
    ) -> None:
        """Creates a new pixelated grating.

        Args:
            widths_or_edge_locs: Either the widths of each grating segment
                or the edge locations. Set `use_edge_locs` according to what
                parametrization is used.
            height_index: Array with the same length as `widths_or_edge_locs`
                that indexes into the `height_fracs` array. If `use_edge_locs`
                is `False`, grating segment `i` has height
                `height_fracs[height_index[i]]`; else, grating segment between
                `edge_locs[i]` and `edge_locs[i + 1]` has height
                `height_fracs[height_index[i]]`.
            height_fracs: List of possible height fractions. This must be an
                increasing array.
            pos: Position of the center of the grating.
            extents: Extents of the grating. Any part of the grating outside
                the extents are cut off.
            material: Material of the "holes" of the grating.
            material2: Material of the actual grating.
            grating_dir: Direction of the grating variation. 0 for x, 1 for y,
                and 2 for z.
            grating_dir_spacing: Pixel size of the discrete grating along the
                grating direction.
            etch_dir: Direction of the etch (height) variation of the grating.
            etch_dir_divs: The pixel size of the discrete grating along the
                etch direction is given by `extents[etch_dir] / etch_dir_divs`.
            etch_polarity: Direction of the etch. Either +/- 1.
            use_edge_locs: Whether to use width or edge location
                parametrization.

        Raises:
            ValueError: If `height_fracs` is an not an increasing array.
        """
        super().__init__([widths_or_edge_locs, height_index])

        if np.any(np.diff(height_fracs) < 0):
            raise ValueError("`height_fracs` needs to be an increasing array.")

        self._use_edge_locs = use_edge_locs

        self._mat = goos.material.get_material(material)
        self._mat2 = goos.material.get_material(material2)
        self._pos = np.array(pos)
        self._extents = np.array(extents)
        self._dir = grating_dir
        self._etch_dir = etch_dir
        self._etch_polarity = etch_polarity
        # Calculate the last direction. Because we know that the three axis
        # directions are 0, 1, and 2, we must have that
        # `dir + etch_dir + extrude_dir = 3`.
        self._extrude_dir = 3 - (grating_dir + etch_dir)

        # Calculate the size of each pixel.
        self._pixel_size = np.array(self._extents, dtype=float)
        self._pixel_size[self._dir] = grating_dir_spacing
        self._pixel_size[self._etch_dir] = extents[etch_dir] / etch_dir_divs

        self._value_shape = goos.PixelatedContShapeFlow.get_shape(
            extents, self._pixel_size)

        self._param_dims = [
            self._value_shape[self._dir], self._value_shape[self._etch_dir]
        ]
        self._height_fracs = height_fracs

        # TODO(logansu): Decide if it's worth it to use `GratingParam`.
        if False and np.all(height_fracs == [0, 1]) and etch_dir_divs == 1:
            self._use_grating_param = True
        else:
            self._use_grating_param = False

            etch_fracs = 1 - np.flip(height_fracs)
            # Passing in dummy initial values.
            initial_value = [0, 1]
            levels = [0, 0]
            self._param = MultiEtchEdgeParametrization(self._param_dims,
                                                       etch_fracs,
                                                       initial_value, levels)

    def eval(self, inputs: List[goos.NumericFlow]) -> goos.ShapeFlow:
        if not self._use_edge_locs:
            edge_locs = np.cumsum(inputs[0].array)
        else:
            edge_locs = inputs[0].array

        if self._use_grating_param:
            # TODO(logansu): Handle inverted flag.
            # We have to instantiate the parametrization every time because
            # it requires knowing the number of elements.

            array = np.r_[edge_locs, self._extents[self._dir]]
            self._param = GratingParam(array / self._pixel_size[self._dir],
                                       self._param_dims[0],
                                       inverted=False)
        else:
            # `MultiEtchParametrization` assumes that we start and end the grating
            # at the highest level. We handle this by prepending and appending a
            # grating edge at the very beginning and very end with the highest
            # possible etch level.
            edge_locs = np.r_[0, edge_locs, self._extents[self._dir]]
            self._param.levels = np.r_[len(self._height_fracs), inputs[1].array,
                                       len(self._height_fracs)]
            self._param.from_vector(edge_locs / self._pixel_size[self._dir])
        values = np.reshape(self._param.get_structure(),
                            self._param_dims,
                            order="F")
        values = np.moveaxis(values[:, :, np.newaxis], [0, 1, 2],
                             [self._dir, self._etch_dir, self._extrude_dir])

        if self._etch_polarity < 0:
            values = np.flip(values, axis=self._etch_dir)
        return goos.PixelatedContShapeFlow(pos=self._pos,
                                           rot=np.zeros(3),
                                           array=values,
                                           material=self._mat,
                                           material2=self._mat2,
                                           pixel_size=self._pixel_size,
                                           extents=self._extents)

    def grad(
        self, input_vals: List[goos.NumericFlow],
        grad_val: goos.PixelatedContShapeFlow.Grad
    ) -> List[goos.NumericFlow.Grad]:
        grad = grad_val.array_grad
        if self._etch_polarity < 0:
            grad = np.flip(grad, axis=self._etch_dir)

        grad = np.moveaxis(grad, [self._dir, self._etch_dir, self._extrude_dir],
                           [0, 1, 2])
        grad = np.array(
            grad.flatten(order="F")
            @ self._param.calculate_gradient()) / self._pixel_size[self._dir]
        if not self._use_grating_param:
            # Remove the first and last "artificial" edges (see `eval`).
            grad = grad[1:-1]
        else:
            grad = grad[:-1]

        if not self._use_edge_locs:
            grad = np.flip(np.cumsum(np.flip(grad)))

        return [goos.NumericFlow.Grad(grad), None]


class GratingL2D(goos.Action):
    node_type = "goos.grating.grating_l2d"

    def __init__(
        self,
        cont_var: goos.Function,
        disc_widths_or_edge_locs: goos.Variable,
        disc_levels: goos.Variable,
        pixel_size: float,
        depths: np.ndarray,
        start_depth_ind: int,
        end_depth_ind: int,
        min_features: np.ndarray,
        max_features: np.ndarray = None,
        divisions: int = 5,
        use_edge_locs: bool = False,
        discr_min_features: np.ndarray = None,
        discr_max_features: np.ndarray = None,
    ):
        """Discretizes a grating.

        Args:
            cont_var: Variable containing single array of values of the
                continuous pixelated grating.
            disc_widths_or_edge_locs: Either variable for storing the grating
                widths or edge locations.
            disc_levels: Variable for storing grating height levels.
            pixel_size: Size of each pixel in `cont_var`.
            depths: Increasing array of grating depths.
            start_depth_ind: Index of the starting depth (leftmost).
            end_depth_ind: Index of the ending depth (rightmost).
            min_features: List of minimum feature sizes for each etch depth,
                or a single scalar for uniform minimum features.
            max_features: List of maximum feature sizes for each etch depth,
                or a single scalar for uniform maximum features.
            divisions: Number of divisions to split each pixel in the continuous
                grating. More divisions increase the accuracy of the solution.
            use_edge_locs: Flag that determins whether
                `disc_widths_or_edge_locs` corresponds to widths or edge
                locaations.
            discr_min_features: Minimum features used during the discretization
                process (rather than the features of the grating). Defaults
                to `min_features` if `None`.
            discr_max_features: Maximum features used during the discretization
                process (rather than the features of the grating). Defaults
                to `max_features` if `None`.
        """
        super().__init__([cont_var, disc_widths_or_edge_locs, disc_levels])

        self._use_edge_locs = use_edge_locs

        if pixel_size > np.min(min_features):
            raise ValueError("Minimum feature must be larger than pixel size.")

        self._cont_var = cont_var
        self._disc_widths = disc_widths_or_edge_locs
        self._disc_levels = disc_levels

        self._pixel_size = pixel_size
        self._depths = depths
        self._min_features = min_features
        self._max_features = max_features

        if discr_min_features is None:
            discr_min_features = min_features
        if discr_max_features is None:
            discr_max_features = max_features

        self._discr_min_features = np.array(discr_min_features) / pixel_size
        self._discr_max_features = discr_max_features
        if self._discr_max_features is not None:
            self._discr_max_features = np.array(
                self._discr_max_features) / pixel_size

        self._start_depth_ind = start_depth_ind
        self._end_depth_ind = end_depth_ind
        self._divisions = divisions

    def run(self, plan: goos.OptimizationPlan) -> None:
        target = plan.eval_node(self._cont_var).array.squeeze()

        edges, levels = _get_general_edge_loc_dp(target, self._depths,
                                                 self._discr_min_features,
                                                 self._discr_max_features,
                                                 self._start_depth_ind,
                                                 self._end_depth_ind,
                                                 self._divisions)
        edges = np.array(edges) * self._pixel_size

        if not self._use_edge_locs:
            widths = np.r_[edges[0], edges[1:] - edges[:-1]]
            plan.set_var_value(self._disc_widths, widths)

            if np.isscalar(self._min_features):
                min_feats = self._min_features
            else:
                min_feats = np.array([self._min_features[i] for i in levels])

            if self._max_features is None:
                max_feats = np.inf
            elif np.isscalar(self._max_features):
                max_feats = self._max_features
            else:
                max_feats = np.array([self._max_features[i] for i in levels])

            ones = np.ones_like(widths)
            plan.set_var_bounds(self._disc_widths,
                                [ones * min_feats, ones * max_feats])
        else:
            plan.set_var_value(self._disc_widths, edges)
            ones = np.ones_like(edges)
            plan.set_var_bounds(self._disc_widths,
                                [ones * -np.inf, ones * np.inf])

        plan.set_var_value(self._disc_levels, levels)


def discretize_to_pixelated_grating(cont_var: goos.Variable,
                                    height_fracs: np.ndarray,
                                    pixel_size: float,
                                    start_height_ind: int,
                                    end_height_ind: int,
                                    min_features: np.ndarray,
                                    max_features: np.ndarray = None,
                                    divisions: int = 5,
                                    use_edge_locs: bool = False,
                                    discr_min_features: np.ndarray = None,
                                    discr_max_features: np.ndarray = None,
                                    **kwargs):

    widths_or_edge_locs = goos.Variable(0)
    height_index = goos.Variable(0)
    grating = PixelatedGrating(widths_or_edge_locs,
                               height_index,
                               height_fracs,
                               use_edge_locs=use_edge_locs,
                               **kwargs)

    action = GratingL2D(cont_var,
                        widths_or_edge_locs,
                        height_index,
                        pixel_size,
                        height_fracs,
                        start_height_ind,
                        end_height_ind,
                        min_features,
                        max_features,
                        divisions,
                        use_edge_locs=use_edge_locs,
                        discr_min_features=discr_min_features,
                        discr_max_features=discr_max_features)
    goos.get_default_plan().add_action(action)
    height_index.freeze()

    return widths_or_edge_locs, height_index, grating


class MultiEtchEdgeParametrization(Parametrization):

    def __init__(self,
                 dims,
                 etch_fracs,
                 initial_value: np.ndarray,
                 levels: np.ndarray,
                 min_feature=0,
                 lower_bound=0,
                 upper_bound=1,
                 bounds=None) -> None:
        self.etch_depths = dims[1] * np.array(etch_fracs)
        self.dims = dims
        self.min_feature = min_feature
        self.vector = np.array(initial_value)
        self.levels = levels
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
        # Expand upper and lower bounds into arrays.
        if isinstance(self.lower_bound, numbers.Number):
            self.lower_bound = (self.lower_bound,) * len(self.vector)
        if isinstance(self.upper_bound, numbers.Number):
            self.upper_bound = (self.upper_bound,) * len(self.vector)
        self.project()

    def build_rasterized_1D(self, vector) -> np.ndarray:
        # Rasterize the structure onto a 1D grid.
        vec = np.ones(self.dims[0])
        # Iterate over the "holes".
        for i in range(0, len(vector), 2):
            hole_start = vector[i]
            hole_end = vector[i + 1]
            # Find the middle region where we send all values to zero.
            start_ind_full = int(hole_start) + 1
            end_ind_full = int(hole_end)
            vec[start_ind_full:
                end_ind_full] = 0  # Handle the first and last pixels.
            start_ind_frac = int(hole_start)
            end_ind_frac = int(hole_end)
            if start_ind_frac != end_ind_frac:
                vec[start_ind_frac] -= 1 - (hole_start - int(hole_start))
                vec[end_ind_frac] -= hole_end - int(hole_end)
            else:
                # Starting and ending in same pixel.
                vec[start_ind_frac] -= hole_end - hole_start
        return vec

    def get_level_vecs(self):
        # Separate out into edge parametrizations for each etch level.
        level_vecs = [[] for i in range(len(self.etch_depths))]
        last_level = len(level_vecs)
        for i in range(len(self.vector)):
            cur_level = int(self.levels[i])
            min_level = min(cur_level, last_level)
            max_level = max(cur_level, last_level)
            for k in range(min_level, max_level):
                level_vecs[k].append(self.vector[i])
            last_level = cur_level
        return reversed(level_vecs)

    def get_structure(self) -> np.ndarray:
        # Compute the "continuous" version of the structure.
        level_vecs = self.get_level_vecs()
        raster_vecs = [self.build_rasterized_1D(vec) for vec in level_vecs]

        layer_pixels = np.diff(np.r_[0, self.etch_depths])
        # Generate the structure.
        total_z = np.zeros(self.dims)
        z_ind = 0
        partial_pixels = 0
        for i, raster_vec in enumerate(raster_vecs):
            pixels_left = layer_pixels[i]
            if i != 0:
                if np.abs(partial_pixels) > 0.001:
                    total_z[:, z_ind] += raster_vec * (1 - partial_pixels)
                    z_ind += 1
                    pixels_left -= (1 - partial_pixels)
            full_pixels = int(pixels_left)
            partial_pixels = pixels_left - full_pixels
            if full_pixels > 0:
                total_z[:, z_ind:z_ind + full_pixels] = np.hstack(
                    (raster_vec[:, np.newaxis],) * full_pixels)
                z_ind += full_pixels
            if np.abs(partial_pixels) > 0.001:
                total_z[:, z_ind] += raster_vec * partial_pixels
        # Add final layers of ones.
        if np.abs(partial_pixels) > 0.001:
            total_z[:, z_ind] += 1 - partial_pixels
            z_ind += 1
        total_z[:, z_ind:self.dims[1]] = 1
        return total_z.flatten(order='F')

    def get_bounds(self):
        return None

    def project(self) -> None:
        # Do not let vector go all the way up to self.dims[0] to avoid edge cases.
        self.vector = np.clip(self.vector, 0, self.dims[0] - 0.00001)

    def build_constraints(self):
        # Build matrix that when multiplied by a vector gives you the difference
        # of the entries.
        A = scipy.linalg.circulant([-1, 1] + [0] * (len(self.vector) - 2)).T
        # Get rid of last row since that corresponds to differencing the first
        # element by the last.
        A = A[:-1, :]
        # Constraint on the difference between entries.
        diff_constraint = {
            'type': 'ineq',
            'fun': lambda x: A @ x - np.ones(len(x) - 1) * self.min_feature,
            'jac': lambda x: A
        }
        ident = np.identity(len(self.to_vector()))
        # Constraint on lower bound (x >= 0)
        lower_bound_constraint = {
            'type': 'ineq',
            'fun': lambda x: x,
            'jac': lambda x: ident
        }
        # Constraint on upper bound (x <= self.dims[0])
        upper_bound_constraint = {
            'type': 'ineq',
            'fun': lambda x: self.dims[0] - x,
            'jac': lambda x: -ident
        }
        constraints = [
            diff_constraint, lower_bound_constraint, upper_bound_constraint
        ]
        return constraints

    def calculate_gradient(self):
        # Brute force gradient computation.
        eps = 1e-5
        param_copy = copy.deepcopy(self)
        vec = param_copy.to_vector()
        grad = np.zeros((np.prod(self.dims), len(vec)))
        for i in range(len(vec)):
            orig = vec[i]

            vec[i] = orig + eps
            param_copy.from_vector(vec)
            dsf = param_copy.get_structure()
            vec[i] = orig - eps
            param_copy.from_vector(vec)
            dsb = param_copy.get_structure()

            grad[:, i] = (dsf - dsb) / (2 * eps)

            vec[i] = orig
        return grad

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        self.vector = vector

    def serialize(self):
        data = super().serialize()
        data.update({'etch_depths': self.etch_depths, 'levels': self.levels})
        return data

    def deserialize(self, data):
        if 'etch_depths' in data:
            self.etch_depths = data['etch_depths']
            self.levels = data['levels']
        super().deserialize(data)


def _get_general_edge_loc_dp(x,
                             depths,
                             min_features,
                             max_features=None,
                             start_depth_ind=0,
                             end_depth_ind=0,
                             divisions=5):
    func = lambda a, b, k: (a - b)**2

    x = np.array(x)
    hx = depths
    num_depths = len(depths)

    max_k = divisions * len(x)

    # Compute min and max features.
    if np.isscalar(min_features):
        min_features = [min_features] * num_depths
    if max_features is None:
        max_features = 2**30
    if np.isscalar(max_features):
        max_features = [max_features] * num_depths
    # Min and max features in grid units.
    import math
    d_min = [math.ceil(i * divisions) for i in min_features]
    d_max = [math.floor(i * divisions) for i in max_features]

    # Compute values if grating were completely flat.
    level_values = []
    for k in range(len(hx)):
        vals = [0] * (len(x) + 1)
        for i in range(1, len(x) + 1):
            vals[i] = vals[i - 1] + func(x[i - 1], hx[k], i - 1)
        level_values.append(vals)

    A = [[[2**30] * num_depths
          for i in range(num_depths)]
         for j in range(max_k + 1)]
    for j in range(num_depths):
        A[0][start_depth_ind][j] = 0

    # Keep track of which choice was made in DP for backtracking.
    choices = [[[(-1, -1)
                 for k in range(num_depths)]
                for i in range(num_depths)]
               for j in range(max_k + 1)]

    # Iterate over grating edge locations.
    for k in range(0, max_k + 1):
        k_int = k // divisions
        k_frac = (k % divisions) / divisions
        # Iterate over levels.
        for this_level in range(num_depths):
            for next_level in range(num_depths):
                if this_level == next_level:
                    continue
                next_pixel_val = hx[this_level] * k_frac + hx[next_level] * (
                    1 - k_frac)

                min_val = 2**30
                # Possibility 1: First edge location is here.
                if this_level == start_depth_ind and k <= d_max[start_depth_ind]:
                    min_val = level_values[this_level][k_int]
                    if k_int < len(x):
                        min_val += func(x[k_int], next_pixel_val,
                                        k_int) * k_frac
                min_choice = (-1, -1)
                # Possibility 2: This is not the first edge location.
                # Iterate over last grating edge.
                for i in range(max(0, k - d_max[this_level]),
                               k - d_min[this_level] + 1):
                    i_int = i // divisions
                    i_frac = (i % divisions) / divisions
                    # Iterate over the last level.
                    for prev_level in range(num_depths):
                        if prev_level == this_level:
                            continue
                        last_pixel_val = hx[this_level] * (
                            1 - i_frac) + hx[prev_level] * i_frac
                        val = (A[i][prev_level][this_level] +
                               func(x[i_int], last_pixel_val, i_int) *
                               (1 - i_frac) + level_values[this_level][k_int] -
                               level_values[this_level][i_int + 1])
                        if k_int < len(x):
                            val += func(x[k_int], next_pixel_val,
                                        k_int) * k_frac
                        if val < min_val:
                            min_val = val
                            min_choice = (i, prev_level)
                A[k][this_level][next_level] = min_val
                choices[k][this_level][next_level] = min_choice

    # Backfill values.
    for k in range(0, max_k):
        k_int = k // divisions
        k_frac = (k % divisions) / divisions
        for level in range(num_depths):
            if level == end_depth_ind:
                A[k][level][end_depth_ind] = 2**30
                continue
            last_pixel_val = hx[end_depth_ind] * (1 -
                                                  k_frac) + hx[level] * k_frac
            A[k][level][end_depth_ind] += (
                func(x[k_int], last_pixel_val, k_int) * (1 - k_frac) +
                level_values[end_depth_ind][-1] -
                level_values[end_depth_ind][k_int + 1])

    # Find best structure.
    best_struct_ind = 0
    best_level_ind = 0
    best_val = 2**30
    # Start looking only in the portion that does not violate maximum
    # feature constraint.
    for k in range(max(0, max_k - d_max[end_depth_ind]), max_k + 1):
        for level in range(num_depths):
            if A[k][level][end_depth_ind] < best_val:
                best_val = A[k][level][end_depth_ind]
                best_struct_ind = k
                best_level_ind = level

    # Backtrack to find correct edges.
    levels = []
    edge_locs = []
    struct_ind = best_struct_ind
    prev_level_ind = end_depth_ind
    level_ind = best_level_ind
    while level_ind != -1:
        edge_locs.append(struct_ind / divisions)
        levels.append(level_ind)
        struct_ind, next_level_ind = choices[struct_ind][level_ind][
            prev_level_ind]
        prev_level_ind = level_ind
        level_ind = next_level_ind
    edge_locs.reverse()
    levels.reverse()
    levels = levels[1:] + [end_depth_ind]

    final_edge_locs = [edge_locs[0]]
    final_levels = [levels[0]]
    # Filter out extra levels.
    for i in range(1, len(edge_locs)):
        if final_levels[-1] == levels[i]:
            continue
        final_edge_locs.append(edge_locs[i])
        final_levels.append(levels[i])
    return final_edge_locs, final_levels


class EdgeLocConstraint(goos.Function):
    node_type = "goos.grating.edge_loc_constraint"

    def __init__(self, edge_locs: goos.Function, height_index: goos.Function,
                 min_features: np.ndarray, grating_len: float) -> None:
        super().__init__([edge_locs, height_index])

        self._min_features = min_features
        self._grating_len = grating_len
        # Cache the difference matrix.
        self._diff_mat = None

    def eval(self, inputs: List[goos.NumericFlow]) -> goos.NumericFlow:
        vector = inputs[0].array
        index = inputs[1].array

        if np.isscalar(self._min_features):
            min_feats = self._min_features
        else:
            min_feats = np.array([self._min_features[i] for i in index[:-1]])
        val = min_feats - (vector[1:] - vector[:-1])
        # Append constraints for first and last edge.
        val = np.r_[val, -vector[0], vector[-1] - self._grating_len]

        return goos.NumericFlow(val)

    def grad(self, inputs: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad):
        vector = inputs[0].array

        if self._diff_mat is None or self._diff_mat.shape[1] != len(vector):
            # Build matrix that when multiplied by a vector gives you the
            # difference of the entries.
            A = scipy.linalg.circulant([-1, 1] + [0] * (len(vector) - 2)).T
            # Get rid of last row since that corresponds to differencing first
            # element by the last.
            A = A[:-1, :]

            self._diff_mat = -A

            # Append constraints for first and last edge.
            first = np.zeros(len(vector))
            first[0] = 1
            last = np.zeros(len(vector))
            last[-1] = 1
            self._diff_mat = np.vstack([self._diff_mat, first, last])

        return [
            goos.NumericFlow.Grad(grad_val.array_grad * self._diff_mat), None
        ]


class BarcodeGrating(goos.Shape):
    node_type = "goos.shape.barcode_grating"

    def __init__(
            self,
            edge_locs: goos.Function,
            thickness: goos.Function,
            pos: np.ndarray,
            extents: np.ndarray,
            material: goos.material.Material,
            grating_dir: int,
            etch_dir: int = 2,
    ) -> None:
        super().__init__([edge_locs, thickness])

        self._mat = goos.material.get_material(material)
        self._pos = np.array(pos)
        self._extents = np.array(extents)
        self._dir = grating_dir
        self._etch_dir = etch_dir

    def eval(self, inputs: List[goos.NumericFlow]) -> goos.ArrayFlow:
        grating = []
        edge_locs = inputs[0].array * self._extents[self._dir]
        heights = inputs[1].array
        if len(heights) == 1:
            heights = np.array([heights[0]] * len(edge_locs))
        for left_pos, right_pos, height in zip(edge_locs[:-1], edge_locs[1:],
                                               heights):
            pos = np.array(self._pos, dtype=float)
            extents = np.array(self._extents, dtype=float)

            pos[self._dir] = (self._pos[self._dir] -
                              self._extents[self._dir] / 2 +
                              ((left_pos + right_pos) / 2))
            extents[self._dir] = right_pos - left_pos
            extents[self._etch_dir] *= height
            grating.append(
                goos.CuboidFlow(pos=pos, extents=extents, material=self._mat))
        return goos.ArrayFlow(grating)
