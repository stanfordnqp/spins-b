import numpy as np
import scipy as sp
import scipy.sparse as sparse
from typing import Tuple

# Define small number to avoid 0 in the denominator.
EPS = 1e-12


def CubicMatrices(x_vector: np.array,
                  y_vector: np.array,
                  xq_vector: np.array,
                  yq_vector: np.array,
                  periodicity: np.array = np.array([0, 0]),
                  derivatives: bool = False,
                  deriv_scaling: np.array = np.array(
                      [1, 1e3, 1e6])) -> (np.array, np.array, np.array):
    '''
      CubicMatrices generates matrixes to calculate a cubic interpolation

      Args:
      x_vector: x vector of the fine grid
      y_vector: y vector of the fine grid
      xq_vector: xq vector of the rough grid
      yq_vector: yq vector of the rough grid
      periodicity: 2 element vector indicating if x and/or y is periodic

      Return:
      Phi2f: matrix that interpolates the rough grid to the fine grid
      Phi2fx: matrix that interpolates the rough grid to the fine grid and
      derives to x
      Phi2fy: matrix that interpolates the rough grid to the fine grid and
      derives to y
    '''

    # Evaluate input parameters.
    if np.any([x_vector[0] < xq_vector[0], y_vector[0] < yq_vector[0], \
               x_vector[-1] > xq_vector[-1], y_vector[-1] > yq_vector[-1]]):
        raise ValueError(
            'The fine grid has to be smaller or equal to the rough grid')

    # Make the matrix containing the offset information of x and y to xq and yq.
    x_rem, dx_rem = floor2vector_rem(x_vector, xq_vector)
    y_rem, dy_rem = floor2vector_rem(y_vector, yq_vector)

    x_rem, y_rem = np.meshgrid(x_rem, y_rem, indexing="ij")
    dx_rem, dy_rem = np.meshgrid(dx_rem, dy_rem, indexing="ij")

    x_rem = x_rem.flatten(order="F")
    y_rem = y_rem.flatten(order="F")
    dx_rem = dx_rem.flatten(order="F")
    dy_rem = dy_rem.flatten(order="F")


    # Make the X, Y grid.
    x_grid, y_grid = np.meshgrid(x_vector, y_vector, indexing="ij")

    xy_diag, xy_x_diag, xy_y_diag = MakeXYcubic(x_rem, y_rem)
    dxdy_inv = MakeDXDY_inv(dx_rem, dy_rem)
    dxdy_corr_inv = MakeDXDYcorr_inv(dx_rem, dy_rem)

    # Make A^(-1) matrix according to https://en.wikipedia.org/wiki/Bicubic_interpolation.
    a_cubic = dxdy_inv @ MakeAcubic(np.size(x_grid)) @ dxdy_corr_inv

    # Matrix that connects x and y to Phi at the closest xq end yq and the derivatives.
    phi2f_cubic = Phi2fii(x_vector, y_vector, xq_vector, yq_vector, periodicity,
                          derivatives, deriv_scaling)

    # Multiply the matrices to get the interpolation matrix and the derivatives
    # in x and y.
    phi2f = xy_diag * a_cubic * phi2f_cubic
    phi2fx = xy_x_diag * a_cubic * phi2f_cubic
    phi2fy = xy_y_diag * a_cubic * phi2f_cubic

    return (phi2f, phi2fx, phi2fy)


# Support functions
####################
def floor2vector(arr: np.array,
                 bins: np.array) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the element closest to `bins`.

    `arr` is flattened first.

    Args:
        arr: Array of elements to bin.
        bins: List of bin coordinates.

    Returns:
        A tuple `(arr_nearest, indices)` where `arr_nearest[i]` is an element of
        `bins` that is the largest element no greater than `arr[i]`.
        `indices[i]` corresponds to the index of the bin.
    """
    arr = arr.flatten(order="F")

    indices = np.digitize(arr, bins)
    indices[indices >= len(bins)] = len(bins) - 1

    arr_nearest = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        arr_nearest[i] = bins[indices[i] - 1]

    return arr_nearest, indices - 1


def floor2vector_rem(arr: np.array,
                     bins: np.array) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the remainder of the element closest to `bins`.

    `arr` is flattened first.

    Args:
        arr: Array of elements to bin.
        bins: List of bin coordinates.

    Returns:
        A tuple `(rem, cell_diff)` where `rem[i]` is the difference between
        `arr[i]` and the largest element in `bins` that is no greater than
        `arr[i]`. `cell_diff` is the difference between adjacent elements
        in `bins[i]` corresponding to the bin that `arr[i]` is in.
    """
    arr = arr.flatten(order="F")

    indices = np.digitize(arr, bins)
    indices[indices >= len(bins)] = len(bins) - 1

    rem = np.zeros_like(arr, dtype=float)
    cell_diff = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        rem[i] = arr[i] - bins[indices[i] - 1]
        cell_diff[i] = bins[indices[i]] - bins[indices[i] - 1]

    return rem, cell_diff


def MakeXYcubic(x_vector: np.array, y_vector: np.array) -> (sparse.coo.coo_matrix, \
                                                            sparse.coo.coo_matrix,\
                                                            sparse.coo.coo_matrix):
    '''
      generates the xy vector needed for the cubic interpolation matrix and it derivatives.
      The xy vector concatenates [1, x, x**2, x**3, y, x*y, x**2*y, x**3*y, y**2, x*y**2,
      x**2*y**2, x**3*y**2, y**3, x*y**3, x**2*y**3, x**3*y**3]

      Args:
        x_vector: x values in the 2d fine grid f.
        y_vector: y values in the 2d fine grid f.

      return:
        xy_diag: diagonal matrix with xy values
        xy_x_diag: diagonal matrix with the xy x-derivative
        xy_y_diag: diagonal matrix with the xy y-derivative
    '''

    n_p = np.size(x_vector)

    x = np.ones([4, n_p])
    y = np.ones([4, n_p])
    x_x = np.zeros([4, n_p])
    y_y = np.zeros([4, n_p])

    x[1, :] = x_vector
    x[2, :] = x_vector**2
    x[3, :] = x_vector**3
    y[1, :] = y_vector
    y[2, :] = y_vector**2
    y[3, :] = y_vector**3
    x_x[1, :] = np.ones(np.shape(x_vector))
    x_x[2, :] = 2 * x_vector
    x_x[3, :] = 3 * x_vector**2
    y_y[1, :] = np.ones(np.shape(y_vector))
    y_y[2, :] = 2 * y_vector
    y_y[3, :] = 3 * y_vector**2

    # Prepare the coordinates and values of the sparse matrices.
    diag_indices = np.arange(0, n_p)
    ind_i = []
    ind_j = []
    xy_diag_v = []
    xy_x_diag_v = []
    xy_y_diag_v = []

    for i in range(0, 4):
        for j in range(0, 4):
            xy = x[i] * y[j]
            xy_x = x_x[i] * y[j]
            xy_y = x[i] * y_y[j]
            xy_diag_v = np.append(xy_diag_v, xy)
            xy_x_diag_v = np.append(xy_x_diag_v, xy_x)
            xy_y_diag_v = np.append(xy_y_diag_v, xy_y)
            ind_i = np.append(ind_i, diag_indices)
            ind_j = np.append(ind_j, diag_indices + (i + 4 * j) * n_p)

    # Make spare matrix.
    xy_diag = sparse.csc_matrix((xy_diag_v, (ind_i, ind_j)),
                                shape=(n_p, 16 * n_p))
    xy_x_diag = sparse.csc_matrix((xy_x_diag_v, (ind_i, ind_j)),
                                  shape=(n_p, 16 * n_p))
    xy_y_diag = sparse.csc_matrix((xy_y_diag_v, (ind_i, ind_j)),
                                  shape=(n_p, 16 * n_p))

    return (xy_diag, xy_x_diag, xy_y_diag)


def MakeDXDY_inv(dx_vector: np.array,
                 dy_vector: np.array) -> (sparse.coo.coo_matrix):
    '''
      Generates the 1/dxdy vector needed to take the derivative of the xy vector of the cubic interpolation.
      The xy vector concatenates [1, x, x**2, x**3, y, x*y, x**2*y, x**3*y, y**2, x*y**2,
      x**2*y**2, x**3*y**2, y**3, x*y**3, x**2*y**3, x**3*y**3]
      The 1/dxdy vector concatenates [1, x**-1, x**-2, x**-3, y**-1, x**-1*y**-1, x**-2*y**-1,
      x**-3*y**-1, y**-2, x**-1*y**-2, x**-2*y**-2, x**-3*y**-2, y**-3, x**-1*y**-3, x**-2*y**-3,
      x**-3*y**-3]

      Args:
        dx_vector: dx values of a 2d grid.
        dy_vector: dy values of a 2d grid.

      return:
        dxdy_inv_diag: diagonal matrix with xy values
    '''

    n_p = np.size(dx_vector)

    diff_x = np.ones([4, n_p])
    diff_y = np.ones([4, n_p])

    diff_x[1, :] = dx_vector**-1
    diff_x[2, :] = dx_vector**-2
    diff_x[3, :] = dx_vector**-3
    diff_y[1, :] = dy_vector**-1
    diff_y[2, :] = dy_vector**-2
    diff_y[3, :] = dy_vector**-3

    diff_x_diff_y = np.array([])

    for i in range(0, 4):
        for j in range(0, 4):
            diff_x_diff_y = np.append(diff_x_diff_y, diff_y[i] * diff_x[j])

    return sparse.diags(diff_x_diff_y, 0, shape=(16 * n_p, 16 * n_p))


def MakeDXDYcorr_inv(dx_vector: np.array,
                     dy_vector: np.array) -> (sparse.coo.coo_matrix):
    '''
      Generates diagonal matrix with the dxdy values associated to the input values of the
      A-matrix product of the cubic interpolation.
      Since the input is [p00, p10, p01, p11, px00, px10, ..., pxy01, pxy11],
      the dxdy vector with be [1, 1, 1, 1, dx, dx, ..., dx*dy, dx*dy]

      Args:
        dx_vector: dx values of a 2d grid.
        dy_vector: dy values of a 2d grid.

      return:
        dxdycorr_diag: diagonal matrix with xy values

    '''
    n_p = len(dx_vector)
    diff_x_diff_y = np.array([])
    diff_x_diff_y = np.append(diff_x_diff_y, np.ones(4 * n_p))
    diff_x_diff_y = np.append(diff_x_diff_y, np.array(4 * dx_vector.tolist()))
    diff_x_diff_y = np.append(diff_x_diff_y, np.array(4 * dy_vector.tolist()))
    diff_x_diff_y = np.append(diff_x_diff_y,
                              np.array(4 * (dx_vector * dy_vector).tolist()))

    return sparse.diags(diff_x_diff_y, 0, shape=(16 * n_p, 16 * n_p))


def MakeXYcubic_secondDerivative(
        x_vector: np.array, y_vector: np.array
) -> Tuple[sparse.csr.csr_matrix, sparse.csr.csr_matrix, sparse.csr.csr_matrix,
           sparse.csr.csr_matrix, sparse.csr.csr_matrix, sparse.csr.csr_matrix]:
    n_p = np.size(x_vector)

    x = np.ones([4, n_p])
    y = np.ones([4, n_p])
    x_x = np.zeros([4, n_p])
    y_y = np.zeros([4, n_p])

    x_xx = np.zeros([4, n_p])
    y_yy = np.zeros([4, n_p])

    x[1, :] = x_vector
    x[2, :] = x_vector**2
    x[3, :] = x_vector**3
    y[1, :] = y_vector
    y[2, :] = y_vector**2
    y[3, :] = y_vector**3
    x_x[1, :] = np.ones(np.shape(x_vector))
    x_x[2, :] = 2 * x_vector
    x_x[3, :] = 3 * x_vector**2
    y_y[1, :] = np.ones(np.shape(y_vector))
    y_y[2, :] = 2 * y_vector
    y_y[3, :] = 3 * y_vector**2
    x_xx[1, :] = np.zeros(np.shape(x_vector))
    x_xx[2, :] = 2 * np.ones(np.shape(x_vector))
    x_xx[3, :] = 6 * x_vector
    y_yy[1, :] = np.zeros(np.shape(y_vector))
    y_yy[2, :] = 2 * np.ones(np.shape(y_vector))
    y_yy[3, :] = 6 * y_vector

    # Prepare the coordinates and values of the sparse matrices.
    diag_indices = np.arange(0, n_p)
    ind_i = []
    ind_j = []
    xy_diag_v = []
    xy_x_diag_v = []
    xy_y_diag_v = []
    xy_xy_diag_v = []
    xy_xx_diag_v = []
    xy_yy_diag_v = []

    for i in range(0, 4):
        for j in range(0, 4):
            xy = x[i] * y[j]
            xy_x = x_x[i] * y[j]
            xy_y = x[i] * y_y[j]
            xy_xy = x_x[i] * y_y[j]
            xy_xx = x_xx[i] * y[j]
            xy_yy = x[i] * y_yy[j]
            xy_diag_v = np.append(xy_diag_v, xy)
            xy_x_diag_v = np.append(xy_x_diag_v, xy_x)
            xy_y_diag_v = np.append(xy_y_diag_v, xy_y)
            xy_xy_diag_v = np.append(xy_xy_diag_v, xy_xy)
            xy_xx_diag_v = np.append(xy_xx_diag_v, xy_xx)
            xy_yy_diag_v = np.append(xy_yy_diag_v, xy_yy)
            ind_i = np.append(ind_i, diag_indices)
            ind_j = np.append(ind_j, diag_indices + (i + 4 * j) * n_p)

    # Make spare matrix.
    xy_diag = sparse.csr_matrix((xy_diag_v, (ind_i, ind_j)),
                                shape=(n_p, 16 * n_p))
    xy_x_diag = sparse.csr_matrix((xy_x_diag_v, (ind_i, ind_j)),
                                  shape=(n_p, 16 * n_p))
    xy_y_diag = sparse.csr_matrix((xy_y_diag_v, (ind_i, ind_j)),
                                  shape=(n_p, 16 * n_p))
    xy_xy_diag = sparse.csr_matrix((xy_xy_diag_v, (ind_i, ind_j)),
                                   shape=(n_p, 16 * n_p))
    xy_xx_diag = sparse.csr_matrix((xy_xx_diag_v, (ind_i, ind_j)),
                                   shape=(n_p, 16 * n_p))
    xy_yy_diag = sparse.csr_matrix((xy_yy_diag_v, (ind_i, ind_j)),
                                   shape=(n_p, 16 * n_p))

    return (xy_diag, xy_x_diag, xy_y_diag, xy_xy_diag, xy_xx_diag, xy_yy_diag)


def MakeAcubic(n_p: float) -> sparse.coo.coo_matrix:
    """
      MakeAcubic is a function needed to make the cubic interpolation matices.
      The interpolation matrix is a multiplication of the a-matrix and the
      position matrix based on the offset of the points to the rough grid.
      MakeAcubic calculated the A matrix
    """

    # A matrix for a single point (wikipedia: cubic interpolation).
    a_single = np.array([ \
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],\
                 [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],\
                 [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],\
                 [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],\
                 [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],\
                 [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],\
                 [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]\
                     ])

    a_single = sp.sparse.csc_matrix(a_single)
    return sp.sparse.kron(
        a_single, sp.sparse.identity(n_p, format="csc"), format="csc")


def Phi2fii(
        x_vector: np.array,
        y_vector: np.array,
        xq_vector: np.array,
        yq_vector: np.array,
        periodicity: np.array = np.array([0, 0]),
        derivatives: bool = False,
        deriv_scaling: np.array = np.array([1, 1e3, 1e6]),
) -> sparse.coo.coo_matrix:
    '''
      Phi2f is the matrix that generates the f vector wheb multiplied with
      the values on the rough grid. The f vector is [f00 f10 f01 f11 ... fxy11]
      (see wiki of cubic interpolation)
    '''

    n_q = np.size(xq_vector) * np.size(yq_vector)
    shape_q = np.array([np.size(xq_vector), np.size(yq_vector)])
    n_z = np.size(x_vector) * np.size(y_vector)
    shape_z = np.array([np.size(x_vector), np.size(y_vector)])
    shape_f = np.array([n_z, n_q])

    # Make fine grid.
    x_i = np.arange(0, np.size(x_vector))
    y_i = np.arange(0, np.size(y_vector))
    x_grid, y_grid = np.meshgrid(x_vector, y_vector)
    x_i_grid, y_i_grid = np.meshgrid(x_i, y_i)
    sub = np.ravel_multi_index(np.array([np.reshape(x_i_grid, n_z, order="F"),\
                                           np.reshape(y_i_grid, n_z, order="F")]),\
                               shape_z, order="F")

    # Find the fii(phi) index for ever x y index.
    _, indfx = floor2vector(x_vector, xq_vector)
    _, indfy = floor2vector(y_vector, yq_vector)
    indfx, indfy = np.meshgrid(indfx, indfy)
    indfx = indfx.flatten(order="F")
    indfy = indfy.flatten(order="F")

    subfi_00 = np.ravel_multi_index(np.array([indfx, indfy]).astype(int),\
                                    shape_q, order="F")
    subfi_10 = np.ravel_multi_index(np.array([indfx+1, indfy]).astype(int),\
                                    shape_q, order="F")
    subfi_01 = np.ravel_multi_index(np.array([indfx, indfy+1]).astype(int),\
                                    shape_q, order="F")
    subfi_11 = np.ravel_multi_index(np.array([indfx+1, indfy+1]).astype(int),\
                                    shape_q, order="F")

    # Make the diff matrices.
    diff_x = makeDmatrix_quad(xq_vector, yq_vector, 0, periodicity)
    diff_y = makeDmatrix_quad(xq_vector, yq_vector, 1, periodicity)
    #Dx = makeDxmatrix( shape_q )
    #Dy = makeDymatrix( shape_q )

    # Make matrix to duplicate the boundaries according to periodicity
    periodicity_matrix = sparse.eye(
        (len(xq_vector) - periodicity[0]) * (len(yq_vector) - periodicity[1]))
    if periodicity[0] == 1:
        shp = (len(xq_vector),
               len(yq_vector[0:len(yq_vector) - periodicity[1]]))
        periodicity_matrix = duplicate_boundary_data(shp,
                                                     0)[0] @ periodicity_matrix
    if periodicity[1] == 1:
        shp = (len(xq_vector), len(yq_vector))
        periodicity_matrix = duplicate_boundary_data(shp,
                                                     1)[0] @ periodicity_matrix

    # Make the different Phi2fii matrices.
    phi2f00 = sparse.csc_matrix((np.ones(np.size(sub)), (sub, subfi_00)), shape_f)\
                @ periodicity_matrix
    phi2f10 = sparse.csc_matrix((np.ones(np.size(sub)), (sub, subfi_10)), shape_f)\
                @ periodicity_matrix
    phi2f01 = sparse.csc_matrix((np.ones(np.size(sub)), (sub, subfi_01)), shape_f)\
                @ periodicity_matrix
    phi2f11 = sparse.csc_matrix((np.ones(np.size(sub)), (sub, subfi_11)), shape_f)\
                @ periodicity_matrix
    phi2fx00 = phi2f00 * diff_x
    phi2fx10 = phi2f10 * diff_x
    phi2fx01 = phi2f01 * diff_x
    phi2fx11 = phi2f11 * diff_x
    phi2fy00 = phi2f00 * diff_y
    phi2fy10 = phi2f10 * diff_y
    phi2fy01 = phi2f01 * diff_y
    phi2fy11 = phi2f11 * diff_y
    phi2fxy00 = phi2f00 * diff_x * diff_y
    phi2fxy10 = phi2f10 * diff_x * diff_y
    phi2fxy01 = phi2f01 * diff_x * diff_y
    phi2fxy11 = phi2f11 * diff_x * diff_y

    if derivatives:
        p2f = sparse.vstack((phi2f00, phi2f10, phi2f01, phi2f11))
        return sparse.csc_matrix(
            sparse.bmat([[deriv_scaling[0] * p2f, None, None, None],
                         [None, deriv_scaling[1]**-1 * p2f, None, None],
                         [None, None, deriv_scaling[1]**-1 * p2f, None],
                         [None, None, None, deriv_scaling[2]**-1 * p2f]]))

    return sparse.csc_matrix(
        sparse.vstack((phi2f00, phi2f10, phi2f01, phi2f11, phi2fx00, phi2fx10,
                       phi2fx01, phi2fx11, phi2fy00, phi2fy10, phi2fy01,
                       phi2fy11, phi2fxy00, phi2fxy10, phi2fxy01, phi2fxy11)))


def idxdydxy_matrix(xq_vector,
                    yq_vector,
                    periodicity: np.array = np.zeros(2),
                    deriv_scaling: np.array = np.array(
                        [1, 1e3, 1e6])) -> sparse.csc.csc_matrix:
    '''
     generates matrix that turns a parametrization vector f in a parametrization
     vector [f, fx, fy, fxy]
    '''

    diff_x = makeDmatrix_quad(xq_vector, yq_vector, 0, periodicity)
    diff_y = makeDmatrix_quad(xq_vector, yq_vector, 1, periodicity)
    eye = sparse.eye(
        len(xq_vector) * len(yq_vector),
        len(xq_vector) * len(yq_vector))

    return sparse.vstack(
        (deriv_scaling[0] * eye, deriv_scaling[1] * diff_x,
         deriv_scaling[1] * diff_y, deriv_scaling[2] * diff_x @ diff_y))


def makeDxmatrix(m_size: np.array) -> sparse.csc.csc_matrix:
    n_m = np.prod(m_size)
    ind_i = []
    ind_j = []
    values = []
    for i in range(0, m_size[0]):
        for j in range(0, m_size[1]):
            ind = np.ravel_multi_index([i, j], m_size, order="F")
            if i == m_size[0] - 1:
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i - 1, j], m_size, order="F")])
                values.extend([-1])
                ind_i.extend([ind])
                ind_j.extend([np.ravel_multi_index([i, j], m_size, order="F")])
                values.extend([1])
            elif i == 0:
                ind_i.extend([ind])
                ind_j.extend([np.ravel_multi_index([i, j], m_size, order="F")])
                values.extend([-1])
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i + 1, j], m_size, order="F")])
                values.extend([1])
            else:
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i - 1, j], m_size, order="F")])
                values.extend([-1 / 2])
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i + 1, j], m_size, order="F")])
                values.extend([1 / 2])

    return sparse.csc_matrix((values, (ind_i, ind_j)), shape=(n_m, n_m))


def makeDymatrix(m_size: np.array) -> sparse.csc.csc_matrix:
    n_m = np.prod(m_size)
    ind_i = []
    ind_j = []
    values = []
    for i in range(0, m_size[0]):
        for j in range(0, m_size[1]):
            ind = np.ravel_multi_index([i, j], m_size, order="F")
            if j == m_size[1] - 1:
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i, j - 1], m_size, order="F")])
                values.extend([-1])
                ind_i.extend([ind])
                ind_j.extend([np.ravel_multi_index([i, j], m_size, order="F")])
                values.extend([1])
            elif j == 0:
                ind_i.extend([ind])
                ind_j.extend([np.ravel_multi_index([i, j], m_size, order="F")])
                values.extend([-1])
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i, j + 1], m_size, order="F")])
                values.extend([1])
            else:
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i, j - 1], m_size, order="F")])
                values.extend([-1 / 2])
                ind_i.extend([ind])
                ind_j.extend(
                    [np.ravel_multi_index([i, j + 1], m_size, order="F")])
                values.extend([1 / 2])

    return sparse.csc_matrix((values, (ind_i, ind_j)), shape=(n_m, n_m))


def makeDmatrix_quad(x_vector: np.array,
                     y_vector: np.array,
                     axis: int,
                     periodicity: np.array = np.array(
                         [0, 0])) -> sparse.csc.csc_matrix:
    '''
    make a sparse derivative matrix in the x- or y-direction taking into account an ununiform grid

    it uses 3 points to fit a quadratic function and takes first derivative at the point of
    interest. For point in the center it uses x(i-1), x(i), x(i+1), while at the boundaries it used
    either x(i), x(i+1), x(i+2) or x(i-2), x(i-1), x(i)

    if periodic is one the last value in the in-array of the axis is considered to be the same as
    first. The in-array value is used for the dx value.
    For example if axis is 0 and periodic is 1 then the matrix have the size
    (len(x_vector)-1)*len(y_vector)
    '''
    d_zeros = [0, 0, 0]
    periodic = periodicity[axis]
    n_x = len(x_vector)
    n_y = len(y_vector)
    if periodicity[0]:
        n_x -= 1
        x_vector = np.append(x_vector,
                             x_vector[0] - (x_vector[-1] - x_vector[-2]))
    if periodicity[1]:
        n_y -= 1
        y_vector = np.append(y_vector,
                             y_vector[0] - (y_vector[-1] - y_vector[-2]))

    per_i = np.arange(0, n_x + 2)
    per_i[n_x] = 0
    per_i[n_x + 1] = n_x - 1
    per_j = np.arange(0, n_y + 2)
    per_j[n_y] = 0
    per_j[n_y + 1] = n_y - 1

    if axis == 0:
        x = x_vector
        y = y_vector
        len_x = n_x
        min_d_ind0 = [0, 1, 2]
        min_d_ind1 = d_zeros
        max_d_ind0 = [-2, -1, 0]
        max_d_ind1 = d_zeros
        d_ind0 = [-1, 0, 1]
        d_ind1 = d_zeros
    elif axis == 1:  # switch x and y from this point.
        x = y_vector
        y = x_vector
        len_x = n_y
        min_d_ind0 = d_zeros
        min_d_ind1 = [0, 1, 2]
        max_d_ind0 = d_zeros
        max_d_ind1 = [-2, -1, 0]
        d_ind0 = d_zeros
        d_ind1 = [-1, 0, 1]

    m_size = np.array([n_x, n_y])
    n_m = n_x * n_y
    ind_i = []
    ind_j = []
    value = []
    for i in range(0, n_x):
        for j in range(0, n_y):
            row_ind = np.ravel_multi_index([i, j], m_size, order="F")
            if axis == 0:
                ind = i
            elif axis == 1:
                ind = j
            if ind == len_x - 1 and not periodic:
                xmin = x[ind - 2] - x[ind - 1]
                xplus = x[ind] - x[ind - 1]
                denum = xmin * xplus * (xmin - xplus)
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index([i + max_d_ind0[0], j + max_d_ind1[0]],
                                         m_size,
                                         order="F")
                ])
                value.extend([xplus**2 / denum])
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index([i + max_d_ind0[1], j + max_d_ind1[1]],
                                         m_size,
                                         order="F")
                ])
                value.extend([-(xmin**2 - 2 * xmin * xplus + xplus**2) / denum])
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index([i + max_d_ind0[2], j + max_d_ind1[2]],
                                         m_size,
                                         order="F")
                ])
                value.extend([-(2 * xmin * xplus - xmin**2) / denum])
            elif ind == 0 and not periodic:
                xmin = x[ind] - x[ind + 1]
                xplus = x[ind + 2] - x[ind + 1]
                denum = xmin * xplus * (xmin - xplus)
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index([i + min_d_ind0[0], j + min_d_ind1[0]],
                                         m_size,
                                         order="F")
                ])
                value.extend([(2 * xmin * xplus - xplus**2) / denum])
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index([i + min_d_ind0[1], j + min_d_ind1[1]],
                                         m_size,
                                         order="F")
                ])
                value.extend([(xmin**2 - 2 * xmin * xplus + xplus**2) / denum])
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index([i + min_d_ind0[2], j + min_d_ind1[2]],
                                         m_size,
                                         order="F")
                ])
                value.extend([-xmin**2 / denum])
            else:
                xmin = x[ind - 1] - x[ind]
                xplus = x[ind + 1] - x[ind]
                denum = xmin * xplus * (xmin - xplus)
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index(
                        [per_i[i + d_ind0[0]], per_j[j + d_ind1[0]]],
                        m_size,
                        order="F")
                ])
                value.extend([-xplus**2 / denum])
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index(
                        [per_i[i + d_ind0[1]], per_j[j + d_ind1[1]]],
                        m_size,
                        order="F")
                ])
                value.extend([(xplus**2 - xmin**2) / denum])
                ind_i.extend([row_ind])
                ind_j.extend([
                    np.ravel_multi_index(
                        [per_i[i + d_ind0[2]], per_j[j + d_ind1[2]]],
                        m_size,
                        order="F")
                ])
                value.extend([xmin**2 / denum])
    return sparse.csc_matrix((value, (ind_i, ind_j)), shape=(n_m, n_m))


def duplicate_boundary_data(shape, axis: int) -> sparse.csc.csc_matrix:
    '''
    duplicate_boundary_data creates a sparse matrix that duplicated the vectorized data at
    a boundary along the axis.
    For example: if you have a space A that is 4x5 represented by vector with 20 elements. Then
        duplicate_boundary_data(x_in, y_in, axis: 0 ) will generate a 25x20 matrix that will make
        a vector representation of a 5x5 matrix where the first row is duplicated as last row.
    (used to implement periodicity)
    '''
    if axis == 0:
        block_a = sparse.eye(shape[0] - 1).tocsr()
        block_b = sparse.csr_matrix((1, shape[0] - 1))
        block_b[0, 0] = 1
        block_matrix = sparse.vstack([block_a, block_b])
        per = sparse.block_diag(tuple(shape[1] * [block_matrix]))
        block_c = sparse.csr_matrix((shape[0] - 1, 1))
        block_matrix = sparse.hstack([block_a, block_c])
        per_reverse = sparse.block_diag(tuple(shape[1] * [block_matrix]))
    elif axis == 1:
        block_1 = sparse.eye(int(shape[0] * (shape[1] - 1)))
        block_b = sparse.eye(shape[0]).tocsr()
        block_2 = sparse.hstack([
            block_b,
            sparse.csr_matrix((shape[0], shape[0] * int(shape[1] - 2)))
        ])
        per = sparse.vstack([block_1, block_2])
        block_1 = sparse.eye(int(shape[0] * (shape[1] - 1)))
        block_2 = sparse.csr_matrix((shape[0] * int(shape[1] - 1), shape[0]))
        per_reverse = sparse.hstack([block_1, block_2])
    return per, per_reverse


# Geometry matrix functions
############################


def make_geometry_matrix_cubic(
        shape,
        reflection_symmetry,
        periodicity,
        periods,
        full_geometry_matrix: bool = False
) -> (sparse.csc.csc_matrix, sparse.csc.csc_matrix):
    """Make a matrix which expands a parametrizatoin vector according to periodicity and
    symmetry.

    Args:
        shape: Shape of the final parametrization.
        reflection_symmetry: Two element array indicating with or not there is symmetry
            in the x and y direction.
        periodicity: Two element array indicating whether or not the final parametrization
            is periodic in the x and y direction by adding a duplicate row or column.
        periods: Two element array indicating the amount of periods in the final
            parametrization in the x and y direction.
        full_geometry_matrix: Boolean indicating whether of not the periodicity is applied
            to the geometry matrix. (in parametrization this is taken into account
            in the cubic interpolation matrix and should not be part of the geometry
            matrix.) This does not effect the reverse geometry matrix. There the periodicity
            is always taken into account.

    Return:
        geometry matrix, reverse geometry matrix
    """

    periodicity_unitcell = [
        (p not in [0]) * s for p, s in zip(periods, reflection_symmetry)
    ]

    geometry_shape = list(shape)

    if periodicity[1]:
        add_boundary_y, rev_add_boundary_y = duplicate_boundary_data(
            geometry_shape, 1)
        geometry_shape[1] -= 1
    else:
        add_boundary_y = sparse.eye(geometry_shape[0] * geometry_shape[1])
        rev_add_boundary_y = sparse.eye(geometry_shape[0] * geometry_shape[1])
    geometry_shape = [int(sh) for sh in geometry_shape]
    if periodicity[0]:
        add_boundary_x, rev_add_boundary_x = duplicate_boundary_data(
            geometry_shape, 0)
        geometry_shape[0] -= 1
    else:
        add_boundary_x = sparse.eye(add_boundary_y.shape[1])
        rev_add_boundary_x = sparse.eye(rev_add_boundary_y.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]

    if periods[1] == 0:
        periodic_y = sparse.eye(add_boundary_x.shape[1])
        rev_periodic_y = sparse.eye(rev_add_boundary_x.shape[0])
    elif geometry_shape[1] % periods[1] == 0:
        periodic_y, rev_periodic_y = periodic_matrix(
            geometry_shape, axis=1, periods=periods[1])
        geometry_shape[1] /= periods[1]
    else:
        raise ValueError(
            "The parametrization shape does not match the periodicity " +
            "in the y direction.")
    geometry_shape = [int(sh) for sh in geometry_shape]
    if periods[0] == 0:
        periodic_x = sparse.eye(periodic_y.shape[1])
        rev_periodic_x = sparse.eye(rev_periodic_y.shape[0])
    elif geometry_shape[0] % periods[0] == 0:
        periodic_x, rev_periodic_x = periodic_matrix(
            geometry_shape, axis=0, periods=periods[0])
        geometry_shape[0] /= periods[0]
    else:
        raise ValueError(
            "The parametrization shape does not match the periodicity " +
            "in the x direction.")
    geometry_shape = [int(sh) for sh in geometry_shape]

    if periodicity_unitcell[1]:
        geometry_shape[1] += 1
        add_unitcellboundary_y, rev_add_unitcellboundary_y = duplicate_boundary_data(
            geometry_shape, 1)
    else:
        add_unitcellboundary_y = sparse.eye(periodic_x.shape[1])
        rev_add_unitcellboundary_y = sparse.eye(rev_periodic_x.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    if periodicity_unitcell[0]:
        geometry_shape[0] += 1
        add_unitcellboundary_x, rev_add_unitcellboundary_x = duplicate_boundary_data(
            geometry_shape, 0)
    else:
        rev_add_unitcellboundary_x = sparse.eye(
            rev_add_unitcellboundary_y.shape[1])
        add_unitcellboundary_x = sparse.eye(add_unitcellboundary_y.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]

    if reflection_symmetry[1]:
        symmetry_y, rev_symmetry_y = symmetry_matrix(geometry_shape, axis=0)
        geometry_shape[1] = np.ceil(geometry_shape[1] / 2)
    else:
        symmetry_y = sparse.eye(rev_add_unitcellboundary_x.shape[1])
        rev_symmetry_y = sparse.eye(add_unitcellboundary_x.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    if reflection_symmetry[0]:
        symmetry_x, rev_symmetry_x = symmetry_matrix(geometry_shape, axis=1)
        geometry_shape[0] = np.ceil(geometry_shape[0] / 2)
    else:
        symmetry_x = sparse.eye(symmetry_y.shape[1])
        rev_symmetry_x = sparse.eye(rev_symmetry_y.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]

    # add_boundary_y@add_boundary_x
    if full_geometry_matrix:
        geometry_matrix = add_boundary_y@add_boundary_x@periodic_y@periodic_x@ \
            rev_add_unitcellboundary_y@rev_add_unitcellboundary_x@symmetry_y@symmetry_x
    else:
        geometry_matrix = periodic_y@periodic_x@ \
            rev_add_unitcellboundary_y@rev_add_unitcellboundary_x@symmetry_y@symmetry_x
    rev_geometry_matrix = rev_symmetry_x@rev_symmetry_y@add_unitcellboundary_x@ \
        add_unitcellboundary_y@rev_periodic_x@rev_periodic_y@rev_add_boundary_x@ \
            rev_add_boundary_y

    return geometry_matrix, rev_geometry_matrix


def make_geometry_matrix_hermite(shape, reflection_symmetry, periodicity,
                                 periods):
    """Make a matrix which expands a parametrization vector of the form [f, fx, fy, fxy]
    according to periodicity and symmetry.

    Args:
        shape: Shape of the final parametrization.
        reflection_symmetry: Two element array indicating with or not there is symmetry
            in the x and y direction.
        periodicity: Two element array indicating whether or not the final parametrization
            is periodic in the x and y direction by adding a duplicate row or column.
        periods: Two element array indicating the amount of periods in the final
            parametrization in the x and y direction.
        full_geometry_matrix: Boolean indicating whether of not the periodicity is applied
            to the geometry matrix. (in parametrization this is taken into account
            in the cubic interpolation matrix and should not be part of the geometry
            matrix.) This does not effect the reverse geometry matrix. There the periodicity
            is always taken into account.

    Return:
        geometry matrix, reverse geometry matrix
    """

    periodicity_unitcell = [
        (p not in [0, 1]) * s for p, s in zip(periods, reflection_symmetry)
    ]

    geometry_shape = list(shape)

    if periodicity[1]:
        add_boundary_y, rev_add_boundary_y = duplicate_boundary_data(
            geometry_shape, 1)
        geometry_shape[1] -= 1
    else:
        add_boundary_y = sparse.eye(geometry_shape[0] * geometry_shape[1])
        rev_add_boundary_y = sparse.eye(geometry_shape[0] * geometry_shape[1])
    rev_add_boundary_y_derivatives = sparse.block_diag(
        4 * (rev_add_boundary_y,))
    geometry_shape = [int(sh) for sh in geometry_shape]
    if periodicity[0]:
        add_boundary_x, rev_add_boundary_x = duplicate_boundary_data(
            geometry_shape, 0)
        geometry_shape[0] -= 1
    else:
        add_boundary_x = sparse.eye(add_boundary_y.shape[1])
        rev_add_boundary_x = sparse.eye(rev_add_boundary_y.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    rev_add_boundary_x_derivatives = sparse.block_diag(
        4 * (rev_add_boundary_x,))

    if periods[1] == 0:
        periodic_y = sparse.eye(add_boundary_x.shape[1])
        rev_periodic_y = sparse.eye(rev_add_boundary_x.shape[0])
    elif geometry_shape[1] % periods[1] == 0:
        periodic_y, rev_periodic_y = periodic_matrix(
            geometry_shape, axis=1, periods=periods[1])
        geometry_shape[1] /= periods[1]
    else:
        raise ValueError(
            "The parametrization shape does not match the periodicity " +
            "in the y direction.")
    geometry_shape = [int(sh) for sh in geometry_shape]
    periodic_y_derivatives = sparse.block_diag(4 * (periodic_y,))
    rev_periodic_y_derivatives = sparse.block_diag(4 * (rev_periodic_y,))
    if periods[0] == 0:
        periodic_x = sparse.eye(periodic_y.shape[1])
        rev_periodic_x = sparse.eye(rev_periodic_y.shape[0])
    elif geometry_shape[0] % periods[0] == 0:
        periodic_x, rev_periodic_x = periodic_matrix(
            geometry_shape, axis=0, periods=periods[0])
        geometry_shape[0] /= periods[0]
    else:
        raise ValueError(
            "The parametrization shape does not match the periodicity " +
            "in the x direction.")
    geometry_shape = [int(sh) for sh in geometry_shape]
    periodic_x_derivatives = sparse.block_diag(4 * (periodic_x,))
    rev_periodic_x_derivatives = sparse.block_diag(4 * (rev_periodic_x,))

    if periodicity_unitcell[1]:
        geometry_shape[1] += 1
        add_unitcellboundary_y, rev_add_unitcellboundary_y = duplicate_boundary_data(
            geometry_shape, 1)
    else:
        add_unitcellboundary_y = sparse.eye(periodic_x.shape[1])
        rev_add_unitcellboundary_y = sparse.eye(rev_periodic_x.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    add_unitcellboundary_y_derivatives = sparse.block_diag(
        4 * (add_unitcellboundary_y,))
    rev_add_unitcellboundary_y_derivatives = sparse.block_diag(
        4 * (rev_add_unitcellboundary_y,))
    if periodicity_unitcell[0]:
        geometry_shape[0] += 1
        add_unitcellboundary_x, rev_add_unitcellboundary_x = duplicate_boundary_data(
            geometry_shape, 0)
    else:
        rev_add_unitcellboundary_x = sparse.eye(
            rev_add_unitcellboundary_y.shape[1])
        add_unitcellboundary_x = sparse.eye(add_unitcellboundary_y.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    add_unitcellboundary_x_derivatives = sparse.block_diag(
        4 * (add_unitcellboundary_x,))
    rev_add_unitcellboundary_x_derivatives = sparse.block_diag(
        4 * (rev_add_unitcellboundary_x,))

    #Correct the sign of the derivatives based on symmetry.
    sign_matrix_list = [sparse.eye(rev_add_unitcellboundary_x.shape[1])]
    if reflection_symmetry[0]:
        sign_matrix_x = symmetry_matrix_sign_correction(geometry_shape, 0, -1,
                                                        bool(periodicity[0]))
    else:
        sign_matrix_x = sparse.eye(rev_add_unitcellboundary_x.shape[1])
    sign_matrix_list.append(sign_matrix_x)
    if reflection_symmetry[1]:
        sign_matrix_y = symmetry_matrix_sign_correction(geometry_shape, 1, -1,
                                                        bool(periodicity[1]))
    else:
        sign_matrix_y = sparse.eye(rev_add_unitcellboundary_x.shape[1])
    sign_matrix_list.append(sign_matrix_y)
    sign_matrix_list.append(sign_matrix_x @ sign_matrix_y)
    symmetry_sign = sparse.block_diag(tuple(sign_matrix_list))

    #Make symmetry matrices.
    if reflection_symmetry[1]:
        symmetry_y, rev_symmetry_y = symmetry_matrix(geometry_shape, axis=0)
        geometry_shape[1] = np.ceil(geometry_shape[1] / 2)
    else:
        symmetry_y = sparse.eye(rev_add_unitcellboundary_x.shape[1])
        rev_symmetry_y = sparse.eye(add_unitcellboundary_x.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    symmetry_y_derivatives = sparse.block_diag(4 * (symmetry_y,))
    rev_symmetry_y_derivatives = sparse.block_diag(4 * (rev_symmetry_y,))
    if reflection_symmetry[0]:
        symmetry_x, rev_symmetry_x = symmetry_matrix(geometry_shape, axis=1)
        geometry_shape[0] = np.ceil(geometry_shape[0] / 2)
    else:
        symmetry_x = sparse.eye(symmetry_y.shape[1])
        rev_symmetry_x = sparse.eye(rev_symmetry_y.shape[0])
    geometry_shape = [int(sh) for sh in geometry_shape]
    symmetry_x_derivatives = sparse.block_diag(4 * (symmetry_x,))
    rev_symmetry_x_derivatives = sparse.block_diag(4 * (rev_symmetry_x,))

    geometry_matrix = periodic_y_derivatives@periodic_x_derivatives \
        @rev_add_unitcellboundary_y_derivatives@rev_add_unitcellboundary_x_derivatives\
        @symmetry_sign \
        @symmetry_y_derivatives@symmetry_x_derivatives
    rev_geometry_matrix = rev_symmetry_x_derivatives@rev_symmetry_y_derivatives \
        @symmetry_sign \
        @add_unitcellboundary_x_derivatives@add_unitcellboundary_y_derivatives \
        @rev_periodic_x_derivatives@rev_periodic_y_derivatives \
        @rev_add_boundary_x_derivatives@rev_add_boundary_y_derivatives

    return geometry_matrix, rev_geometry_matrix


def symmetry_matrix(shape: tuple, axis: int) -> sparse.coo.coo_matrix:
    '''
    SymmetryMatrix generated a symmetry matrix.
        For example, given a parametrization defined by an xy-grid that
        is symmetric in the x axis, this function will return a matrix that
        turn a parametrization that describes the right side of the grid in a
        parametrization over the entire grid. The matrix size will thus be
        (shape[0]*shape[1])x(shape[0]/2*shape[1]).
        Note: you give the symmetry axis if axis is 0 the second coordinate is
            symmetric. If the axis is 1 the first coordinate is symmetric

    Args:
        shape: Shape of the parametrization.
        axis: Symmetry axis.
    '''
    n_x = shape[0]
    n_y = shape[1]

    if axis == 0:
        block_1 = sparse.eye(int(n_x * np.ceil(n_y / 2)))
        block_b = sparse.eye(n_x).tocsr()
        block_b.indices = -1 * block_b.indices + block_b.shape[1] - 1
        block_2 = sparse.block_diag(tuple(int(n_y / 2) * [block_b])).tocsr()
        block_2.indices = -1 * block_2.indices + block_2.shape[1] - 1
        if bool(n_y % 2):
            block_2 = sparse.hstack(
                [block_2, sparse.csr_matrix((n_x * int(n_y / 2), n_x))])
        sym = sparse.vstack([block_1, block_2])
        sym_reverse = sparse.hstack(
            [block_1, sparse.csr_matrix(block_2.shape).T])
    elif axis == 1:
        block_a = sparse.eye(int(np.ceil(n_x / 2))).tocsr()
        block_b = sparse.eye(int(n_x / 2)).tocsr()
        block_b.indices = -1 * block_b.indices + block_b.shape[1] - 1
        if n_x % 2 == 1:
            block_b = sparse.hstack([block_b, sparse.csr_matrix((n_x // 2, 1))])
        block_matrix = sparse.vstack([block_a, block_b])
        sym = sparse.block_diag(tuple(n_y * [block_matrix]))
        block_matrix_reverse = sparse.vstack(
            [block_a, sparse.csr_matrix(block_b.shape)])
        sym_reverse = sparse.block_diag(tuple(
            n_y * [block_matrix_reverse])).transpose()

    return sym, sym_reverse


#deprecated function: use symmetry_matrix
def symmetry_matrix2D(x: np.array, y: np.array,
                      axis: int) -> sparse.coo.coo_matrix:
    '''
    SymmetryMatrix2D generated a symmetry matrix.
        For example, given a parametrization defined by an xy-grid that
        is symmetric in the x axis, this function will return a matrix that
        turn a parametrization that describes the right side of the grid in a
        parametrization over the entire grid. The matrix size will thus be
        (len(x)*len(y))x(len(x)/2*len(y)).
        Note: you give the symmetry axis if axis is 0 the second coordinate is
            symmetric. If the axis is 1 the first coordinate is symmetric

    Args:
        shape: Shape of the parametrization.
        axis: Symmetry axis.
    '''
    n_x = len(x)
    n_y = len(y)

    sym, sym_reverse = symmetry_matrix((n_x, n_y), axis)

    return sym, sym_reverse


def symmetry_matrix_sign_correction(shape: tuple,
                                    axis: int,
                                    sign: int = 1,
                                    periodicity: bool = False):
    '''
    SymmetryMatrix2D_sign_correction generates a matrix that corrects the sign of every
        parameter component (f, fx, fy, fxy) according to the parametrization being symmetric
        or anti-symmetric.

        The correct symmetry_matrix for [f, fx, fy, fxy] will be of the form:
            sign_matrix@block_diag(4*symmetry matrix)

    Args:
        shape: Shape of the parametrization.
        axis: Symmetry axis.
        sign: 1 or -1 depanding whether or not you have symmetry or anti-symmetry.
        periodicity: Indicating whether or not there is periodicity.

    Returns:
        periodicity matrix
    '''
    n_x = shape[0]
    n_y = shape[1]
    periodic = int(not periodicity)

    if axis == 1:
        periodic = int(not periodicity)
        block_1 = sparse.eye(int(n_x * (n_y // 2)))
        block_c = sign * (sign > 0) * sparse.eye(n_x, n_x)
        block_2 = sign * sparse.eye(int(n_x * (n_y // 2)))
        zero_0 = sparse.diags([periodic] * n_x + [1] * n_x * (n_y - 2) +
                              [periodic] * n_x)
        if bool(n_y % 2):
            sym_corr = sparse.block_diag((block_1, block_c, block_2)) @ zero_0
        else:
            sym_corr = sparse.block_diag((block_1, block_2)) @ zero_0
    elif axis == 0:
        block_1 = sparse.eye(n_x // 2)
        block_c = sign * (sign > 0) * sparse.eye(1, 1)
        block_2 = sign * sparse.eye(n_x // 2)
        zero_0 = sparse.diags([periodic] + [1] * (n_x - 2) + [periodic])
        if bool(n_x % 2):
            sym_corr_row = sparse.block_diag(
                (block_1, block_c, block_2)) @ zero_0
        else:
            sym_corr_row = sparse.block_diag((block_1, block_2)) @ zero_0
        sym_corr = sparse.block_diag(tuple(n_y * [sym_corr_row]))

    return sym_corr


#deprecated function: use symmetry_matrix_sign_correction
def symmetry_matrix2D_sign_correction(x_vector: np.array,
                                      y_vector: np.array,
                                      axis: int,
                                      sign: int = 1,
                                      periodicity: bool = False):
    '''
    SymmetryMatrix2D_sign_correction generates a matrixon3 that corrects the sign of every
        parameter component (f, fx, fy, fxy) according to the parametrization being symmetric
        or anti-symmetric.

        The correct symmetry_matrix for [f, fx, fy, fxy] will be of the form:
            sign_matrix@block_diag(4*symmetry matrix)

    Args:
     x_vector: x vector of the fine grid
     y_vector: y vector of the fine grid
     axis: Symmetry axis.
     sign: 1 or -1 depanding whether or not you have symmetry or anti-symmetry.
     periodicity: Indicating whether or not there is periodicity.

    Return:
     sign_correction matrix


    '''
    n_x = len(x_vector)
    n_y = len(y_vector)
    shape = (n_x, n_y)
    return symmetry_matrix_sign_correction(shape, axis, sign, periodicity)


def make_periodicity_matrix(shape, periodicity):
    '''
    Make_periodicity matrix makes a sparse matrix that will duplicate the first
    row and/or column as the last row and/or column according to the periodicity.

    Args:
        shape: Shape of the space.
        periodicity: Two element array where each element indicates if the axis
                     is periodic by either 1 or 0.

    Returns:
        geometry and reverse geometry matrix for the periodicity
    '''
    periodicity_matrix = sparse.eye(
        (shape[0] - periodicity[0]) * (shape[1] - periodicity[1]))
    periodicity_matrix_reverse = sparse.eye(
        (shape[0] - periodicity[0]) * (shape[1] - periodicity[1]))
    if periodicity[0] == 1:
        per, per_rev = duplicate_boundary_data(
            (shape[0], shape[1] - periodicity[1]), 0)
        periodicity_matrix = per @ periodicity_matrix
        periodicity_matrix_reverse = periodicity_matrix_reverse @ per_rev
    if periodicity[1] == 1:
        per, per_rev = duplicate_boundary_data(shape, 1)
        periodicity_matrix = per @ periodicity_matrix
        periodicity_matrix_reverse = periodicity_matrix_reverse @ per_rev

    return periodicity_matrix, periodicity_matrix_reverse


def periodic_matrix(shape: list, axis: int,
                    periods=int) -> sparse.coo.coo_matrix:
    '''
    periodic_matrix generated a sparse matrix that duplicates a
    parametrization vector according to a give periodicity

    If the periods is 0, it will only duplicate the outer parametrization in
    the axis direction, rather then duplicating the parametrization. (this
    is usefull to describe a structure with periodic boundary conditions)

    Args:
      shape: Shape of the matrix on which you want to make periodic.
      axis: Axis in which you want to have periodicity.
      periods: Amount of periods.

    Returns:
      periodicity matrix
    '''
    n_x = shape[0]
    n_y = shape[1]
    if axis == 0:
        if n_x % periods > 0:
            raise ValueError('x cannot be divided by periods')
        else:
            n_x_periodic = n_x // periods
            block_a = sparse.eye(n_x_periodic).tocsr()
            block_b = sparse.vstack(periods * [block_a])
            geometry_matrix = sparse.block_diag(tuple(n_y * [block_b]))
            block_a = sparse.eye(n_x_periodic).tocsr()
            matrix_list = periods * [sparse.csr_matrix(block_a.shape)]
            matrix_list[0] = block_a
            block_b = sparse.vstack(matrix_list)
            geometry_matrix_reverse = sparse.block_diag(tuple(
                n_y * [block_b])).transpose()
    elif axis == 1:
        if n_y % periods > 0:
            raise ValueError('y cannot be divided by periods')
        else:
            n_y_periodic = n_y // periods
            n = n_y_periodic * n_x
            block_a = sparse.eye(n).tocsr()
            geometry_matrix = sparse.vstack(periods * [block_a])
            matrix_list = periods * [sparse.csr_matrix(block_a.shape)]
            matrix_list[0] = block_a
            geometry_matrix_reverse = sparse.vstack(matrix_list).transpose()

    return geometry_matrix, geometry_matrix_reverse


#deprecated function: use periodic_matrix
def periodic_matrix2D(x: np.array, y: np.array, axis: int,
                      periods=int) -> sparse.coo.coo_matrix:
    '''
    periodic_matrix2D generated a sparse matrix that duplicates a
    parametrization vector according to a give periodicity

    If the periods is 0, it will only duplicate the outer parametrization in
    the axis direction, rather then duplicating the parametrization. (this
    is usefull to describe a structure with periodic boundary conditions)
    '''
    n_x = len(x)
    n_y = len(y)
    periodic_matrix((n_x, n_y), axis, periods)
