"""Module for rasterizing polygons onto a non-uniform grid.

The rasterization is done with float-precision anti-aliasing on a non-uniform
rectangular grid. This module provides functions to perform rastering on both
1D and 2D grids:
    1. `raster_1D`: This function rasters a line-segment on a 1D grid.
    2. `raster_2D`: This function rasters a polygon on a 2D grid.
"""
import numpy as np
from scipy import sparse
from typing import Tuple


def raster_1D(poly_x: np.ndarray,
              grid_x: np.ndarray) -> np.ndarray:
    """Function to raster a line segment `poly_x` onto a grid given by `grid_x`.

    This function is of use while extruding the drawn structures - `raster_2D`
    is used for rendering the 2D pattern corresponding to the structure, and
    `raster_1D` is used for rendering the 1D dependence on the extrusion
    direction.

    Args:
        poly_x: 1D array of size 2 containing the end points of the segment
            being rastered on the grid.
        grid_x: 1D array of the edge coordinates on the grid being rastered.
            Note that the rastering is done on the centers of the grid.

    Returns:
        A one-dimensional numpy array with number of points being equal to the
        the size of `grid_x` minus 1 which contains pixels corresponding to
        rendering the line segment onto the 1D grid.

    Raises:
        ValueError: If `poly_x` is not a numpy array with only two elements or
            if `grid_x` has less than 2 elements.
    """
    if poly_x.size != 2:
        raise ValueError("Expected `poly_x` to have exactly 2 elements, got "
                         "{} instead.".format(poly_x.size))
    if grid_x.size < 2:
        raise ValueError("Expected `grid_x` to have at least 2 elements, got "
                         "{} instead.".format(grid_x.size))

    # Get the dimensions of the grid.
    dim = grid_x.size

    # The rastering function assumes that `poly_x` is sorted - if not, then
    # sort `poly_x`.
    if poly_x[0] > poly_x[1]:
        poly_x = np.array([poly_x[1], poly_x[0]])

    # If the segment specified by `poly_x` lies outside the grid specified by
    # `grid_x`, return 0.
    if poly_x[1] <= grid_x[0] or poly_x[0] >= grid_x[-1]:
        return np.zeros(dim - 1)

    # Computing the indices of the two end points of the segment.
    poly_ind = np.clip(np.digitize(poly_x, grid_x) - 1, 0, dim - 2)

    # Handling the interior points.
    pixel_area = np.zeros(dim - 1)
    pixel_area[poly_ind[0] + 1:poly_ind[1]] = 1.0

    # Computing fraction of the segment in the bins on the edges.
    if poly_x[0] < grid_x[0]:
        start_frac = 1.0
    else:
        start_frac = (grid_x[poly_ind[0] + 1] - poly_x[0]) / (
            grid_x[poly_ind[0] + 1] - grid_x[poly_ind[0]])

    if poly_x[1] > grid_x[-1]:
        end_frac = 1.0
    else:
        end_frac = (poly_x[1] - grid_x[poly_ind[1]]) / (
            grid_x[poly_ind[1] + 1] - grid_x[poly_ind[1]])

    # Handling the end points.
    if poly_ind[0] == poly_ind[1]:
        # Case where the entire line segment is in the same pixel.
        pixel_area[poly_ind[0]] = start_frac + end_frac - 1.0
    else:
        # Case where the start and end is in different pixels.
        pixel_area[poly_ind[0]] = start_frac
        pixel_area[poly_ind[1]] = end_frac

    return pixel_area


def _compute_intersections(
        x_coords_a: Tuple[np.ndarray, np.ndarray],
        y_coords_a: Tuple[np.ndarray, np.ndarray],
        x_coords_b: Tuple[np.ndarray, np.ndarray],
        y_coords_b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the intersection between two sets of line segments.

    Given two sets of line segments, set A and set B, this function computes the
    intersection between each segment in set A with each segment in set B. This
    function is useful when computing the intersection of a polygon with a grid.
    The basic mathematical formulation used here is parametrizing a point on a
    line by its distance from the initial coordinate of the segment. Consider a
    segment specified by `(xi, yi)` and `(xf, yf)`, and a point `(x, y)` on the
    specified by `f`, its distance from `(xi, yi)` as a fraction of the total
    length `L` of the segment:
                                   (x, y)
            (xi, yi) o---------------o------------------o (xf, yf)
                         l = f * L
                      <------------->
                                      L
                      <--------------------------------->
    Note that if `f > 1`, then `(x, y)` lies beyond `(xf, yf)` and if `f < 0`,
    then the segment lies before `(xi, yi)`. The coordinates `(x, y)` are then
    given by:
                          x = xi + f * (xf - xi)
                          y = yi + f * (yf - yi)
    This parametrization can be used to compute the intersection of two
    segments. Consider two segments specified by `(xi_a, yi_a)`, `(xf_a, yf_a)`
    and `(xi_b, yi_b)`, `(xf_b, yf_b)` respectively. Let these two segments
    intersect at `(x, y)` which corresponds to fraction `f_a` on the first
    segment and `f_b` on the second segment. Then:
           x = xi_a + f_a * (xf_a - xi_a) = xi_b + f_b * (xf_b - xi_b)
           y = yi_a + f_a * (yf_a - yi_a) = yi_b + f_b * (yf_b - yi_b)
    which can solved simultaneously to obtain:
                    f_a = N_a / D, f_b = N_b / D
    where:
       N_a = (xi_b - xi_a) * (yf_b - yi_b) - (xf_b - xi_b) * (yi_b - yi_a)
       N_b = (xi_b - xi_a) * (yf_a - yi_a) - (xf_a - xi_a) * (yi_b - yi_a)
       D = (yf_b - yi_b) * (xf_a - xi_a) - (yf_a - yi_a) * (xf_b - xi_b)

    NOTE 1: Once `f_a` and `f_b` have been computed, it can be deduced whether
    the point of intersection `(x, y)` lies inside either of the segments. E.g.
    it lies inside the first segment if `f_a` is between `0` and `1`, else it
    lies outside. Additionally, note that:
        1. If the two segments are parallel, but not colinear (i.e. they
           don't lie on the same line), then `D = 0`, and `f_a` and `f_b` will
           evaluate to infinity.
        2. If the two segments are colinear, then `D = N_a = N_b = 0`, and
           `f_a` and `f_b` will evaluate to `nan`.

    NOTE 2: In the current implementation, we treat `colinear` segments as
    non intersecting (even if they overlap). This is because the intended use
    of this function is to compute the intersection of the segments of a polygon
    with the grid lines, and to introduce these intersections as new vertices.
    If one of the polygon segments overlaps with a gridline, then no new
    vertices need to be introduced, and hence such cases are treated as non
    intersecting.

    Args:
        x_coords_a: The initial and final x coordinates of the segments in the
            set A, specified as a tuple of two 1D arrays.
        y_coords_a: The initial and final y coordinates of the segments in the
            set A, specified as a tuple of two 1D arrays.
        x_coords_b: The initial and final x coordinates of the segments in the
            set B, specified as a tuple of two 1D arrays.
        y_coords_b: The initial and final y coordinates of the segments in the
            set B, specified as a tuple of two 1D arrays.

    Returns: A tuple of three numpy arrays which, in order, are given by -
        * An adjacency matrix - this is a 2D numpy array of booleans. The
          `(i, j)` element of this matrix indicates whether the `ith` segment of
          set A intersects with the `jth` segment of set B. Note that if the
          segments are parallel to each other, then they can either not
          intersect at all, touch each other at an end point or overlap with
          each other - in all these cases, the adjaceny matrix stores a `False`
          for the intersection.
        * A coordinate matrix - this is a 2D numpy array in which the `(i, j)`
          element of the matrix indicates where the line corresponding to the
          `ith` segment of set A intersects the line corresponding to the `jth`
          segment of set B. The coordinates `(x, y)` of the point of
          intersection are stored as a complex number `x + iy` for speed in
          future computations. Also note that this computes the intersection
          between the lines and not the segments i.e. even if `ith` segment
          does not intersect the `jth` segment (due to their finite lengths),
          this matrix will store the interesection that would have happened if
          the segments were extended to infinity. Note that -
            - If two lines are parallel to each other and don't intersect, then
              the coordinate stored corresponding to their intersection is
              `inf`.
            - If two lines are parallel to each other and do intersect (then
              they overlap with each other), then the coordinate stored
              corresponding to their intersection is `nan`.
        * A normalized signed distance matrix - this is a 2D numpy array in
          which the `(i, j)` indicates how far is the point of intersection
          between the `ith` segment in set A and the `jth` segment in set B is
          from the initial coordinate of the `ith` segment in set A. This
          distance is specified as a fraction (or multiple) of the length of the
          `ith` segment in set A. Note that this distance can be less than 0
          or greater than 1 specifying that the point of intersection is outside
          the `ith` segment in set A.
    """
    # Compute the differences `xi_b - xi_a` and `yi_b - yi_a` between each
    # segment in set A and set B. Note that in order to vectorize this
    # computation, we reshape `xi_a` (`yi_a`) into a column vector, so that on
    # subtracting the vector `xi_b` (`yi_b`), the subtraction broadcasts,
    # resulting in the computation of differences between all possible
    # combinations of segments in set A and set B.
    diff_xib_xia = x_coords_b[0] - x_coords_a[0][:, np.newaxis]
    diff_yib_yia = y_coords_b[0] - y_coords_a[0][:, np.newaxis]

    # Compute the differences `xf_a - xi_a`, `yf_a - yi_a`, `xf_b - xi_b`,
    # `yf_b - yf_a`. Note that we reshape `xf_a - xi_a` and `yf_a - yi_a` into
    # a column vector and `xf_b - xi_b` and `yf_b - yi_b` into a row vector so
    # that it is possible to broadcast multiplication with `xi_b - xi_a` and
    # `yi_b - yi_a` (which are two dimensional arrays).
    diff_xfa_xia = (x_coords_a[1] - x_coords_a[0])[:, np.newaxis]
    diff_yfa_yia = (y_coords_a[1] - y_coords_a[0])[:, np.newaxis]
    diff_xfb_xib = (x_coords_b[1] - x_coords_b[0])[np.newaxis, :]
    diff_yfb_yib = (y_coords_b[1] - y_coords_b[0])[np.newaxis, :]

    # Compute `N_a`, `N_b` and `D` - these quantities are required to compute
    # the intersections between the two sets of segments.
    numerator_a = diff_xib_xia * diff_yfb_yib - diff_yib_yia * diff_xfb_xib
    numerator_b = diff_xib_xia * diff_yfa_yia - diff_yib_yia * diff_xfa_xia
    denominator = diff_yfb_yib * diff_xfa_xia - diff_xfb_xib * diff_yfa_yia

    # Compute the intersection. Note that, as described in the docstring, when
    # two segments are colinear or parallel, the quantities `D`, `N_a` and `N_b`
    # might be 0 - we perform computations that are otherwise invalid so as to
    # avoid handling the colinear or parallel case explicitly.
    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute `f_a` and `f_b` of the points of intersection between the
        # segments. Note that `f_a` also corresponds to the normalized
        # signed distance of the point of intersection from the initial
        # coordinate of the segments in set A.
        frac_a = numerator_a / denominator
        frac_b = numerator_b / denominator

        # Find the adjacency matrix of the intersecting lines. This is done by
        # checking if both `frac_a` and `frac_b` are withing 0 and 1. Note that
        # if two segments are parallel or colinear, then they are treated as
        # non intersecting.
        adj_mat = np.logical_and.reduce((frac_a >= 0, frac_a <= 1,
                                         frac_b >= 0, frac_b <= 1))

        # Compute the intersection coordinates. Note that the intersection
        # coordinates are stored as a complex number with the `x` coordinate
        # being the real part and `y` coordinate being the imaginary part.
        inter_x_coords = x_coords_a[0][:, np.newaxis] + diff_xfa_xia * frac_a
        inter_y_coords = y_coords_a[0][:, np.newaxis] + diff_yfa_yia * frac_a
        inter_coords = inter_x_coords + 1.0j * inter_y_coords

    return adj_mat, inter_coords, frac_a


def _expand_polygon_vertices(poly_xy: np.ndarray,
                             grid_x: np.ndarray,
                             grid_y: np.ndarray) -> np.ndarray:
    """Expands the array of polygon vertices by adding grid intersections.

    An important step in rastering a polygon onto a grid is to compute the
    intersection of the polygon segments with the grid. These intersections are
    then added as vertices of the polygon to the original set of vertices that
    it was specified by. This oversampling allows each segement of the polygon
    to be associated with one grid pixel.

    Args:
        poly_xy: The array of vertices that the polygon is specified by. This is
            a `2 x N` array, where `N` is the number of vertices of the polygon.
        grid_x: The edge x-coordinates of the grid pixels.
        grid_y: The edge y-coordinates of the grid pixels.

    Returns:
        An array of oversampled vertices corresponding to the polygon in
        question (which contains the specified vertices as well as the
        intersection of the polygon segments with the grid lines).
    """
    # Number of vertices used for specifying the polygon.
    num_poly_vertices = poly_xy.shape[1]

    # The first step is to compute the intersection of the polygon segments with
    # the gridlines. This intersection can be computed using
    # `_compute_intersections`.
    # Setup the initial and final coordinates of the polygon segments. Note that
    # the initial coordinates are just the coordinates in `poly_xy`, while the
    # final coordinates are `poly_xy` circularly shifted by 1. They are stored
    # in the tuples `x_coords_poly` and `y_coords_poly` (with the first and
    # second element of the tuple being the initial and final coordinates).
    poly_xy_cshift = np.roll(poly_xy, -1, axis=1)
    x_coords_poly = (poly_xy[0], poly_xy_cshift[0])
    y_coords_poly = (poly_xy[1], poly_xy_cshift[1])

    # Setup the initial and final coordinates of the gridlines.
    # We first compute the gridlines that are not outside the bounding box of
    # the polygon.
    # Compute the bounding box of the polygon.
    min_coords = poly_xy.min(axis=1)
    max_coords = poly_xy.max(axis=1)
    # Check which gridlines lie within the bounding box.
    keep_x = np.logical_and(grid_x >= min_coords[0], grid_x <= max_coords[0])
    keep_y = np.logical_and(grid_y >= min_coords[1], grid_y <= max_coords[1])
    # Setup a local grid - this is the portion of the grid which is within the
    # bounding box of the polygon. Moreover, the bounding box is itself added
    # as gridlines to this local grid. Note that the local grid can possibly
    # fall outside the grid specified by `grid_x` and `grid_y` if there are
    # polygon vertices that fall outisde this grid (since the bounding box will
    # then fall outside this grid).
    local_grid_x = np.unique(np.hstack((min_coords[0],
                                        grid_x[keep_x],
                                        max_coords[0])))
    local_grid_y = np.unique(np.hstack((min_coords[1],
                                        grid_y[keep_y],
                                        max_coords[1])))
    # Finally, setup the coordinates of the gridlines. Note that we need to
    # account for both the horizontal and vertical gridlines. In particular,
    # suppose that the local grid has x-edge coordinates `[x1, x2 ... xn]` and
    # y-edge coordinates `[y1, y2 ... ym]`, then there will be a total of `m`
    # horizontal gridlines and `n` vertical gridlines. The coordinates of the
    # gridlines will be:
    #       Initial x: [x1, x1 ...m times... x1, x1, x2 ............. xn]
    #       Final x:   [xn, xn ...m times... xn, x1, x2 ............. xn]
    #       Initial y: [y1, y2 ............. ym, y1, y1 ...n times... y1]
    #       Final y:   [y1, y2 ............. ym, yn, yn ...n times... yn]
    # where the first `m` coordinates correspond to the `m` horizontal gridlines
    # and the next `n` coordinates correspond to the `n` vertical gridlines.
    # Setup a tuple of the initial and final x coordinates (stored as 1D
    # arrays).
    x_coords_grid = (np.hstack((np.full_like(local_grid_y, local_grid_x[0]),
                                local_grid_x)),
                     np.hstack((np.full_like(local_grid_y, local_grid_x[-1]),
                                local_grid_x)))
    # Setup a tuple of the initial and final y coordinates (stored as 1D
    # arrays).
    y_coords_grid = (np.hstack((local_grid_y,
                                np.full_like(local_grid_x, local_grid_y[0]))),
                     np.hstack((local_grid_y,
                                np.full_like(local_grid_x, local_grid_y[-1]))))

    # Perform the intersection computation. Note that this computation returns
    # an adjacency matrix indicating which segments of the polygon intersect
    # which gridlines, the points of intersection and their normalized signed
    # distance from the polygon vertices. Refer to `_compute_intersections` for
    # more details.
    adj_matrix, inter_coords, norm_dist = _compute_intersections(
        x_coords_poly, y_coords_poly, x_coords_grid, y_coords_grid)

    # Some of the computed intersection can conceivably fall outside the grid
    # specified by `grid_x` and `grid_y`, since the `local_grid` need not be
    # contained in this grid. We therefore clip the intersection coordinates to
    # be within the grid.
    inter_coords_clip = (
            np.real(inter_coords).clip(grid_x[0], grid_x[-1]) +
            1.0j * np.imag(inter_coords).clip(grid_y[0], grid_y[-1]))

    # Add the computed intersection as polygon vertices. Note that for an
    # accurate area calculation, it is important to add these vertices in the
    # correct order i.e. if the polygon segment `(xi, yi)` and `(xf, yf)`
    # intersects with the grid at `(x1, y1)`, `(x2, y2)` .... `(xn, yn)`, then
    # these points have to inserted in polygon segment in ascending order of
    # their signed distance from `(xi, yi)`. This order can be deduced by
    # sorting each row of `norm_dist`, since the `ith` row of `norm_dist` will
    # be the normalized signed distance of the intersection of the `ith` polygon
    # segment with all the gridlines from the initial coordinate of that
    # segment.
    insertion_order = norm_dist.argsort(axis=1)
    # Insert the intersection points in the right order. Note that the the
    # result of this insertion is a 2D matrix, with each row beginning with
    # a polygon vertex followed by all the intersections that occured with the
    # the segment with the polygon vertex with its initial point. Note that at
    # this point, the vertices that do not intersect with the segment have not
    # been removed, but are simply sorted as per their signed distance. Note
    # that to sort the intersection coordinate, we use the following numpy
    # indexing property - if `A` is `n x m` 2D numpy array, and `ind_row` and
    # `ind_col` are two `n x m` 2D numpy arrays of indices, then
    #                   A[(ind_row, ind_col)]
    # has `(i, j)` element as `ind_row(i), ind_col(j)` element of `A`. Moreover,
    # if `ind_row` (`ind_col`) is only a 1D array of size `n` (`m`), then
    # `ind_row` (`ind_col`) is treated as a broadcasted 2D array of shape
    # `n x m` obtained by repeating it along the rows (columns).
    poly_xy_with_inter = np.hstack(
            (poly_xy[0, :, np.newaxis] + 1.0j * poly_xy[1, :, np.newaxis],
             inter_coords_clip[(np.arange(num_poly_vertices)[:, np.newaxis],
                                insertion_order)]))
    # Compute a 2D matrix of booleans which indicate the vertices in
    # `poly_xy_with_inter` that are to be retained. Since we are retaining the
    # vertices corresponding to the original polygon, each row begins with a 1
    # (since each row of `poly_xy_with_inter` begins with a vertex of the
    # original polygon), followed by the corresponding row of `adj_matrix`
    # sorted as per `insertion_order`.
    retained_vertices = np.hstack(
            (np.ones((num_poly_vertices, 1), dtype=bool),
             adj_matrix[(np.arange(num_poly_vertices)[:, np.newaxis],
                        insertion_order)]))
    # Calculate the final polygon vertices by indexing `poly_xy_with_inter`
    # with `retained_vertices`. Note that the following numpy indexing
    # property is used - if `A` is a `n x m` 2D numpy array, and `ind` is a
    # `n x m` 2D boolean numpy array, then `A[ind]` is a 1D numpy array which
    # only pickes out the elements of `A` where `ind` has a `True`. Note that
    # the flattening (conversion to 1D) is done in a row-major format.
    vertices = poly_xy_with_inter[retained_vertices]
    # Remove all the vertices that fall outside the grid specified by `grid_x`
    # and `grid_y`.
    vertices_inside = np.logical_and.reduce((np.real(vertices) <= grid_x[-1],
                                             np.real(vertices) >= grid_x[0],
                                             np.imag(vertices) <= grid_y[-1],
                                             np.imag(vertices) >= grid_y[0]))
    vertices = vertices[vertices_inside]
    # Finally, remove all the consecutive duplicate vertices.
    vertices_consec_unique = np.ediff1d(vertices, to_begin=1 + 1j).astype(bool)
    vertices = vertices[vertices_consec_unique]

    return vertices


def raster_2D(poly_xy: np.ndarray,
              grid_x: np.ndarray,
              grid_y: np.ndarray) -> np.ndarray:
    """Draws a polygon onto a 2D grid of pixels.

    Pixel values equal to the fraction of the pixel area covered by the polygon.
    This implementation is written for accuracy and works with double precision,
    in contrast to most other implementations which are written for speed and
    usually only allow for 256 (and often fewer) possible pixel values without
    performing (very slow) super-sampling.

    Args:
        poly_xy: `2 x N` ndarray containing x,y coordinates for each point in
            the polygon.
        grid_x: x-coordinates for the edges of each pixel specified as a 1D
            array.
        grid_y: y-coordinates for the edges of each pixel specified as a 1D
            array.

    Returns:
        2D ndarray with pixel values in the range [0, 1] containing the
        anti-aliased polygon. Note that the size of the array is
        `[grid_x.size - 1, grid_y.size - 1]`.

    Raises:
        ValueError: If `poly_xy` doesn't have exactly two rows or if `grid_x`
            or `grid_y` have a size less than 2.
    """
    if poly_xy.shape[0] != 2:
        raise ValueError("Expected `poly_xy` to have 2 rows, got {} instead.".
                         format(poly_xy.shape[0]))
    if grid_x.size < 2 or grid_y.size < 2:
        raise ValueError("Expected both `grid_x` and `grid_y` to have atleast 2"
                         " elements, got sizes of {} and {} respectively."
                         .format(grid_x.size, grid_y.size))

    # Oversample the polygon by including its intersection with the grid as
    # new vertices.
    vertices = _expand_polygon_vertices(poly_xy, grid_x, grid_y)

    # If the shape fell completely outside our area, just return a blank grid.
    if vertices.size == 0:
        return zeros((grid_x.size - 1, grid_y.size - 1))
    # Calculate segment cover, area, and corresponding pixel's subscripts.
    poly = np.hstack((vertices, vertices[0]))
    endpoint_avg = (poly[:-1] + poly[1:]) * 0.5

    # Remove segments along the right and top edges (they correspond to outside
    # pixels, but couldn't be removed until now because poly_xy stores points,
    # not segments, and the edge points are needed when creating endpoint_avg).
    non_edge = np.logical_and(
        np.real(endpoint_avg) < grid_x[-1],
        np.imag(endpoint_avg) < grid_y[-1])

    endpoint_final = endpoint_avg[non_edge]
    x_sub = np.digitize(np.real(endpoint_final), grid_x) - 1
    y_sub = np.digitize(np.imag(endpoint_final), grid_y) - 1

    cover = np.diff(np.imag(poly), axis=0)[non_edge] / np.diff(grid_y)[y_sub]
    area = (np.real(endpoint_final) -
            grid_x[x_sub]) * cover / np.diff(grid_x)[x_sub]

    # Use coo_matrix(...).toarray() to efficiently convert from (x, y, v) pairs
    # to ndarrays. We can use v = (-area + 1j * cover) followed with calls to
    # np.real() and np.imag() to improve performance (otherwise we'd have to
    # call coo_matrix() twice. It's really inefficient because it involves lots
    # of random memory access, unlike real() and imag()).
    poly_grid = sparse.coo_matrix(
        (-area + 1j * cover, (x_sub, y_sub)),
        shape=(grid_x.size - 1, grid_y.size - 1)).toarray()
    result_grid = np.real(poly_grid) + np.imag(poly_grid).cumsum(axis=0)
    return np.abs(result_grid)
