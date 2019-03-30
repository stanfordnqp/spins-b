"""This module contains transformations related to grating parametrization."""
import copy
from typing import List

import numpy as np

from spins.invdes import parametrization
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


class GratingEdgeDiscretization:
    """Defines a transformation that maps a `DirectParam` into a `GratingParam`.
    """

    def __init__(self, param: parametrization.DirectParam, min_feature: float,
                 dx: float) -> None:
        """Defines a new transformation.

        Args:
            param: Continuous parametrization to use.
            min_feature: Minimum feature size in nanometers.
            dx: Grid spacing in nanometers. Used to convert `min_feature` from
                nanometers into pixels.

        Raises:
            ValueError: If minimum feature is smaller than individual pixel.
        """
        # Calculate in terms of pixels.
        self._min_feature = min_feature / dx
        if self._min_feature < 1:
            raise ValueError("Minimum feature size must be larger than one "
                             "grid spacing, got {}".format(min_feature))

        self._param = param

    def __call__(self, param: parametrization.GratingParam,
                 unused_event_data) -> None:
        """Run the discretization.

        Args:
            param: The `GratingParam` to set.
        """
        edge_loc = _get_edge_loc_dp(self._param.to_vector(), self._min_feature)

        # Post-process `edge_loc` because `_get_edge_loc_dp` returns edge
        # locations assuming that the first edge is a falling edge whereas
        # `GratingParam` assumes the first edge is a rising edge.
        # To remedy, prepend a rising edge at the left boundary and append a
        # falling edge at the end, EXCEPT if there is already an edge
        # already there.
        if edge_loc[0] == 0:
            edge_loc = edge_loc[1:]
        else:
            edge_loc = [0] + edge_loc

        grating_len = len(self._param.to_vector())
        if edge_loc[-1] == grating_len:
            edge_loc = edge_loc[:-1]
        else:
            edge_loc = edge_loc + [grating_len]

        param.from_vector(edge_loc)


@optplan.register_transformation(optplan.GratingEdgeFitTransformation)
def create_grating_edge_fit(
        params: optplan.GratingEdgeFitTransformation,
        work: workspace.Workspace) -> GratingEdgeDiscretization:
    simspace = work.get_object(params.parametrization.simulation_space)
    return GratingEdgeDiscretization(
        work.get_object(params.parametrization), params.min_feature,
        simspace.dx)


def _get_edge_loc_dp(x: List[float], min_feature: float = 0) -> np.ndarray:
    """Perform discretization at the end of continuous.

    This function transforms a vector from a `DirectParam` representing the
    continuous phase of a grating optimization to a `GratingParam` where the
    vector represents the location of edges in the grating. This is achieved by
    solving the minimization problem:

    min_y ||R(y) - x||^2

    where `R(y)` "renders" the grating `y` onto a grid with spacing given by
    `x` (equivalent to calling `get_structure` on `GratingParam` with vector
    `y`). This is a non-convex problem but it can be solved approximately
    using combinatorial optimization. See Su et al. Opt. Express (2017)
    Fully-automated optimization of grating couplers for details.

    Note that this only works if `min_feature` is larger than one pixel.

    Args:
        x: A list of permittivity values for each pixel.
        min_feature: Minimum distance between grating edges in terms of pixels.

    Returns:
        List of grating edges.
    """
    # TODO(logansu): Technically, this will not obey the fabrication constraint
    # imposed at the boundaries requiring that the pixel be more than
    # `2 * min_feature` away, but this is probably rare.

    # TODO(logansu): Properly comment. Also consider switching to the
    # multilevel discretization algorithm.

    # Optimized version of get_edge_loc_dp
    func = lambda a, b: (a - b)**2

    # Maximum possible value for structure.
    max_val = len(x)

    divisions = 5
    max_k = divisions * len(x) + 1

    # base_value = value we would have if no grating exists.
    x = np.array(x)
    zero_value = np.cumsum(func(x, 0))
    one_value = np.cumsum(func(x, 1))
    zero_value = [0] + zero_value.tolist()
    one_value = [0] + one_value.tolist()
    dp = [max_val] * max_k

    # min_feature size in terms of grid units.
    d = int(min_feature * divisions)
    struct_dp = [[] for i in range(d)]

    # k = this right edge
    dp[0] = 0
    for k in range(d, max_k):
        k_int = k // divisions
        k_frac = (k % divisions) / divisions
        # Possibility 1: Edge starts at 0 and goes until k.
        if k_int < len(x):
            val = zero_value[k_int] + func(x[k_int], 1 - k_frac) * k_frac
        else:
            val = max_val
        new_edges = [0, k / divisions]
        best_i_ind = 0
        # Possibility 2: Last edge started at i.
        j_value = max_val
        best_j_ind = -1
        # i = prev right edge
        for i in reversed(range(k - 2 * d + 1)):
            i_int = i // divisions
            i_frac = (i % divisions) / divisions
            base_value = (func(x[i_int], 1 - i_frac) * (1 - i_frac) -
                          one_value[i_int + 1] + zero_value[k_int])
            if k_int < len(x):
                base_value += func(x[k_int], 1 - k_frac) * k_frac

            # j = this left edge
            j_int = (i + d) // divisions
            j_frac = ((i + d) % divisions) / divisions
            new_j_value = (one_value[j_int] + func(x[j_int], j_frac) -
                           zero_value[j_int + 1])

            if new_j_value < j_value:
                j_value = new_j_value
                best_j_ind = i + d

            new_value = j_value + base_value
            if dp[i] + new_value < val:
                val = dp[i] + new_value
                new_edges = [best_j_ind / divisions, k / divisions]
                best_i_ind = i
        dp[k] = val
        struct_dp.append(struct_dp[best_i_ind] + new_edges)

    # Backfill dp values.
    for k in range(d, max_k - 1):
        k_int = k // divisions
        k_frac = (k % divisions) / divisions
        dp[k] += (one_value[-1] - one_value[k_int + 1] +
                  func(x[k_int], 1 - k_frac) * (1 - k_frac))

    struct_best_ind = np.argmin(dp[d:]) + d
    edge_loc = struct_dp[struct_best_ind]

    return edge_loc
