"""
Functions for far field analysis

In part of the code for the far field calculation and in the analysis functions
spherical coordinates are used. Theta is the angle with the z-axis (axis=2)
and varies from 0 to pi. Phi is the angle of the projection on the xy-plane
with the x-axis. It varies between -pi and pi. (note however that for the
analysis function triangle_selection it can be any angle)

"""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import spins.fdfd_tools as fdfd_tools
import spins.fdfd_tools.operators as operators
from typing import List


# Far field functions
######################
def make_near2farfield_matrix(points: np.array,
                              omegas: float,
                              grid,
                              dxes,
                              pos: np.array,
                              width: np.array,
                              polarity: int,
                              eps_0: float,
                              mu_0=1.0,
                              spherical_axis=2) -> sparse.spmatrix:
    '''
    This function returns a matrix that transforms the fields on a plane
    defined by pos and width to the farfield.
    The fields are calculated on a sphere centered in the origin.

    input:
    - points: points in the far field
    - omega: omega
    - grid: the grid object of the simulation
    - dxes: the dxes of the simulation
    - pos: center position of the plane you want to project out
    - width: size of the plane, (the normal vector is calculated based on
                the 0 value of this vector, e.g. if the width is [100,0,100] the
                normal is [0,1,0])
    - polarity: direction in which you want to project
    - eps_0
    - mu_0
    - spherical_axis: orientation of the spherical coordinate system
    output:
    - sparse tranformation matrix

    '''
    # prepare normal
    axis = abs(width).argmin()
    normal = np.array([0, 0, 0])
    normal[axis] = polarity

    # prepare the grid
    x = grid.xyz[0]
    y = grid.xyz[1]
    z = grid.xyz[2]
    shape = [x.size, y.size, z.size]

    # make E-fields to H-fields matrix
    arg = {'omega': omegas, 'dxes': dxes}
    e2h = operators.e2h(**arg)

    # matrix to move all the vector components to the H component
    move2H_E, move2H_H = move2H_matrix(axis, shape)

    # make the n_cross
    normal_grid = [
        normal[0] * np.ones(shape), normal[1] * np.ones(shape),
        normal[2] * np.ones(shape)
    ]
    normal_vec = fdfd_tools.vec(normal_grid)

    cross_normal = operators.vec_cross(normal_vec)

    # matrix to reduce the fields to the region defined by the slices
    find_ind = lambda x, vec: np.abs(x - vec).argmin()
    x_slice = slice(
        find_ind(pos[0] - width[0] / 2, x),
        find_ind(pos[0] + width[0] / 2, x) + abs(normal[0]), 1)
    y_slice = slice(
        find_ind(pos[1] - width[1] / 2, y),
        find_ind(pos[1] + width[1] / 2, y) + abs(normal[1]), 1)
    z_slice = slice(
        find_ind(pos[2] - width[2] / 2, z),
        find_ind(pos[2] + width[2] / 2, z) + abs(normal[2]), 1)
    x_crop, y_crop, z_crop, dx_crop, dy_crop, dz_crop, fos = fields_on_slice(
        x, y, z, dxes, x_slice, y_slice, z_slice)

    # Calculate the area elements
    d_crop = [dx_crop, dy_crop, dz_crop]
    d_crop[axis] = np.ones_like(d_crop[axis])
    d_area = d_crop[0] * d_crop[1] * d_crop[2]

    # make the Fourier matrix
    farfield_radius = 1
    fourier_matrix = make_fourier_matrix(
        x_crop, y_crop, z_crop, d_area,
        farfield_radius * np.squeeze(points[0:, 0]),
        farfield_radius * np.squeeze(points[0:, 1]),
        farfield_radius * np.squeeze(points[0:, 2]), omegas)
    # make the transformation matrix in cartesian coordinates
    k = omegas * points
    k_vec = k.flatten('F')
    cross_k = operators.vec_cross(k_vec)

    A = fourier_matrix @ fos @ cross_normal @ move2H_H @ e2h
    F = fourier_matrix @ fos @ (-cross_normal) @ move2H_E

    t_cart = -1j*omegas*(mu_0/(4*np.pi*farfield_radius)*np.exp(-1j*omegas*farfield_radius)*A)\
             -(-1j)/eps_0*eps_0/(4*np.pi*farfield_radius)*np.exp(-1j*omegas*farfield_radius)*(cross_k @ F)

    # transform to spherical coordinates
    cart2sp = cart2spheric_matrix(points[0:, 0], points[0:, 1], points[0:, 2],
                                  spherical_axis)

    t_sp = cart2sp @ t_cart

    # remove the radial components
    n_k = int(k_vec.shape[0] / 3)
    Id = sparse.eye(n_k)
    zeros = sparse.csr_matrix((n_k, n_k))
    rm_radial = sparse.bmat([[zeros, zeros, zeros], [zeros, Id, zeros],
                             [zeros, zeros, Id]])

    t = rm_radial @ t_sp

    return t


def make_near2farfield_box_matrix(points: np.array,
                                  omegas: float,
                                  grid,
                                  dxes,
                                  box_center: np.array,
                                  box_size: np.array,
                                  eps_0: float,
                                  mu_0=1.0,
                                  spherical_axis=2) -> sparse.spmatrix:
    '''
    This function returns a matrix that projects fields on a box
    to the farfield.
    (the far field matrices of all the sides of the box are calculated and
    summed)

    input:
    - points: points in the far field
    - omegas: omega
    - grid: the grid object of the simulation
    - dxes: the dxes of the simulation
    - box_center: center position of the box you want to project out
    - box_size: size of the box
    - eps_0
    - mu_0
    - spherical_axis: how the spherical coordinate system is oriented
            (defaut=z-axis)
    output:
    - tranformation matrix: sparse matrix

    '''
    #x0
    pos = box_center - np.array([box_size[0] / 2, 0, 0])
    width = np.array([0, box_size[1], box_size[2]])
    arg = {
        'points': points,
        'omegas': omegas,
        'grid': grid,
        'dxes': dxes,
        'pos': pos,
        'width': width,
        'polarity': -1,
        'eps_0': eps_0,
        'spherical_axis': spherical_axis
    }
    farfield_transform_x0 = make_near2farfield_matrix(**arg)

    #x1
    pos = box_center + np.array([box_size[0] / 2, 0, 0])
    width = np.array([0, box_size[1], box_size[2]])
    arg = {
        'points': points,
        'omegas': omegas,
        'grid': grid,
        'dxes': dxes,
        'pos': pos,
        'width': width,
        'polarity': 1,
        'eps_0': eps_0,
        'spherical_axis': spherical_axis
    }
    farfield_transform_x1 = make_near2farfield_matrix(**arg)

    #y0
    pos = box_center - np.array([0, box_size[1] / 2, 0])
    width = np.array([box_size[0], 0, box_size[2]])
    arg = {
        'points': points,
        'omegas': omegas,
        'grid': grid,
        'dxes': dxes,
        'pos': pos,
        'width': width,
        'polarity': -1,
        'eps_0': eps_0,
        'spherical_axis': spherical_axis
    }
    farfield_transform_y0 = make_near2farfield_matrix(**arg)

    #y1
    pos = box_center + np.array([0, box_size[1] / 2, 0])
    width = np.array([box_size[0], 0, box_size[2]])
    arg = {
        'points': points,
        'omegas': omegas,
        'grid': grid,
        'dxes': dxes,
        'pos': pos,
        'width': width,
        'polarity': 1,
        'eps_0': eps_0,
        'spherical_axis': spherical_axis
    }
    farfield_transform_y1 = make_near2farfield_matrix(**arg)

    #z0
    pos = box_center - np.array([0, 0, box_size[2] / 2])
    width = np.array([box_size[0], box_size[1], 0])
    arg = {
        'points': points,
        'omegas': omegas,
        'grid': grid,
        'dxes': dxes,
        'pos': pos,
        'width': width,
        'polarity': -1,
        'eps_0': eps_0,
        'spherical_axis': spherical_axis
    }
    farfield_transform_z0 = make_near2farfield_matrix(**arg)

    #z1
    pos = box_center + np.array([0, 0, box_size[2] / 2])
    width = np.array([box_size[0], box_size[1], 0])
    arg = {
        'points': points,
        'omegas': omegas,
        'grid': grid,
        'dxes': dxes,
        'pos': pos,
        'width': width,
        'polarity': 1,
        'eps_0': eps_0,
        'spherical_axis': spherical_axis
    }
    farfield_transform_z1 = make_near2farfield_matrix(**arg)

    return (
        farfield_transform_x0 + farfield_transform_x1 + farfield_transform_y0 +
        farfield_transform_y1 + farfield_transform_z0 + farfield_transform_z1)


# Functions needed for the far field transform
################################################
def fields_on_slice(xs: np.array, ys: np.array, zs: np.array, dxes,
                    x_slice: slice, y_slice: slice, z_slice: slice
                   ) -> (np.array, np.array, np.array, sparse.spmatrix):
    '''
    make a matrix that makes only keeps the fields defined by some slices.

    input:
    - xs, ys, zs: vectors of the large grid
    - x_slice, y_slice, z_slice: slices defining the region want to keep

    output:
    - crop_x, crop_y, crop_z: vectors of the new grid
    - Crop_matrix_fields: sparse matrix that keeps the field in the new region

    '''

    x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
    dx, dy, dz = np.meshgrid(dxes[1][0], dxes[1][1], dxes[1][2], indexing='ij')

    pos = np.zeros_like(x)
    pos[x_slice, y_slice, z_slice] = 1
    pos_vector = pos.flatten(order='F')

    pos_matrix = sparse.csr_matrix(sparse.diags(pos_vector))
    nonzero_rows, _ = pos_matrix.nonzero()
    crop_matrix = pos_matrix[nonzero_rows]
    crop_matrix_fields = sparse.vstack([
        sparse.hstack([
            crop_matrix,
            sparse.csr_matrix(crop_matrix.shape),
            sparse.csr_matrix(crop_matrix.shape)
        ]),
        sparse.hstack([
            sparse.csr_matrix(crop_matrix.shape), crop_matrix,
            sparse.csr_matrix(crop_matrix.shape)
        ]),
        sparse.hstack([
            sparse.csr_matrix(crop_matrix.shape),
            sparse.csr_matrix(crop_matrix.shape), crop_matrix
        ])
    ])

    crop_x = crop_matrix * x.flatten(order='F')
    crop_y = crop_matrix * y.flatten(order='F')
    crop_z = crop_matrix * z.flatten(order='F')
    crop_dx = crop_matrix * dx.flatten(order='F')
    crop_dy = crop_matrix * dy.flatten(order='F')
    crop_dz = crop_matrix * dz.flatten(order='F')

    return crop_x, crop_y, crop_z, crop_dx, crop_dy, crop_dz, crop_matrix_fields


def move2H_matrix(axis: int,
                  shape: List[int]) -> (sparse.spmatrix, sparse.spmatrix):
    '''
    This function make 2 matrices that interpolate the E and H field to the
    position of
    Hi in the Yee cell, where i is the axis given

    input:
    - axis: the axis of the H vector you want to move to
    - shape: shape of the simulation grid
    output:
    - mv_E: transformation matrix that moves the E fields
    - mv_H: transformatino matrix that moves the H fields

    '''
    #get the averaging matrices for every direction
    fwX = fdfd_tools.operators.avgf(0, shape)
    bwX = fdfd_tools.operators.avgb(0, shape)
    fwY = fdfd_tools.operators.avgf(1, shape)
    bwY = fdfd_tools.operators.avgb(1, shape)
    fwZ = fdfd_tools.operators.avgf(2, shape)
    bwZ = fdfd_tools.operators.avgb(2, shape)
    n = np.prod(shape)
    # make averaging matrix for every field component
    if axis == 0:
        av_Ex = fwZ @ fwY @ bwX
        av_Ey = fwZ
        av_Ez = fwY
        av_Hx = sparse.eye(n)
        av_Hy = fwY @ bwX
        av_Hz = fwZ @ bwX
    if axis == 1:
        av_Ex = fwZ
        av_Ey = fwZ @ fwX @ bwY
        av_Ez = fwX
        av_Hx = fwX @ bwY
        av_Hy = sparse.eye(n)
        av_Hz = fwZ @ bwY
    if axis == 2:
        av_Ex = fwY
        av_Ey = fwX
        av_Ez = fwY @ fwX @ bwZ
        av_Hx = fwX @ bwZ
        av_Hy = fwY @ bwZ
        av_Hz = sparse.eye(n)

    # compose total matrix
    zeros = sparse.csr_matrix((n, n), dtype=float)
    move_E = sparse.vstack([
        sparse.hstack([av_Ex, zeros, zeros]),
        sparse.hstack([zeros, av_Ey, zeros]),
        sparse.hstack([zeros, zeros, av_Ez])
    ])
    move_H = sparse.vstack([
        sparse.hstack([av_Hx, zeros, zeros]),
        sparse.hstack([zeros, av_Hy, zeros]),
        sparse.hstack([zeros, zeros, av_Hz])
    ])

    return move_E, move_H


def make_fourier_matrix(x: np.array, y: np.array, z: np.array, d_area: np.array,
                        x_ff: np.array, y_ff: np.array, z_ff: np.array,
                        omega: float) -> sparse.spmatrix:
    '''
    fourier matrix:
        fourier_matrix(i,0:)=exp(kx(i)*x+ky(i)*y+kz(i)*z)

    input:
        - x, y, z: vector with all the x values vor every point (not a mesh vector)
        - d_area: area at the x, y, z positions
        - x_ff, y_ff, z_ff: point in the farfields
        - omega
    output:
        - fourier_matrix

    '''

    r_ff = np.squeeze(x_ff**2 + y_ff**2 + z_ff**2)**0.5

    kx = omega * np.squeeze(x_ff) / r_ff
    ky = omega * np.squeeze(y_ff) / r_ff
    kz = omega * np.squeeze(z_ff) / r_ff

    # Matrix for a single component
    single_fourier_matrix = (kx[:, np.newaxis] @ np.squeeze(x)[np.newaxis, :] +
                             ky[:, np.newaxis] @ np.squeeze(y)[np.newaxis, :] +
                             kz[:, np.newaxis] @ np.squeeze(z)[np.newaxis, :])
    single_fourier_matrix = np.exp(1j * single_fourier_matrix) @ sparse.diags(
        d_area, 0)

    # stack the matrices for all the vector components
    zeros = sparse.csr_matrix(
        (single_fourier_matrix.shape[0], single_fourier_matrix.shape[1]),
        dtype=float)
    fourier_matrix = sparse.vstack([
        sparse.hstack([single_fourier_matrix, zeros, zeros]),
        sparse.hstack([zeros, single_fourier_matrix, zeros]),
        sparse.hstack([zeros, zeros, single_fourier_matrix])
    ])

    return fourier_matrix


def make_sphere_point(interpolation_count: int) -> (np.array, np.array):
    '''
    This function creates a sphere of relatively even distributed point.
    It start with all the unit vectors and then interpolates point on
    a sphere in between these points. The more interpolation steps you take the
    points you will have. Typically 4 interpolation steps is enough.

    input:
    - interpolation_count: the amount of interpolation steps
    output:
    - points: array with all the points
    - triangles: array with all the triangles that connect these point
            (these integer values refere to the points)

    '''

    # Starting point for sphere
    points = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1]])
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 5],
                          [1, 2, 5], [2, 3, 5], [3, 0, 5]])
    # refine mesh on sphere
    for n in range(interpolation_count):
        for i in range(triangles.shape[0]):
            # Make points on the edges of every triangle
            p1 = points[triangles[i, 0]] + points[triangles[i, 1]]
            p1 = p1 / (np.sum(p1**2)**0.5)
            p2 = points[triangles[i, 1]] + points[triangles[i, 2]]
            p2 = p2 / (np.sum(p2**2)**0.5)
            p3 = points[triangles[i, 2]] + points[triangles[i, 0]]
            p3 = p3 / (np.sum(p3**2)**0.5)
            # check if the new points are already in the list
            p1_excist = [
                np.all(points[j] == p1) for j in range(points.shape[0])
            ]
            if any(p1_excist):
                index1 = np.where(p1_excist)[0][0]
            else:
                points = np.vstack([points, p1])
                index1 = points.shape[0] - 1
            p2_excist = [
                np.all(points[j] == p2) for j in range(points.shape[0])
            ]
            if any(p2_excist):
                index2 = np.where(p2_excist)[0][0]
            else:
                points = np.vstack([points, p2])
                index2 = points.shape[0] - 1
            p3_excist = [
                np.all(points[j] == p3) for j in range(points.shape[0])
            ]
            if any(p3_excist):
                index3 = np.where(p3_excist)[0][0]
            else:
                points = np.vstack([points, p3])
                index3 = points.shape[0] - 1
            # Create new triangles
            new_triangles = np.array(
                [[index1, index2, index3], [triangles[i, 0], index1, index3],
                 [triangles[i, 1], index1,
                  index2], [triangles[i, 2], index3, index2]])
            # add the new triangles to the triangle list
            if i == 0:
                triangles_temp = new_triangles
            else:
                triangles_temp = np.vstack([triangles_temp, new_triangles])
        # update the triangle list
        triangles = triangles_temp

    return points, triangles


def make_half_sphere_point(interpolation_count: int,
                           polarity: int) -> (np.array, np.array):
    '''
    This function creates a half sphere of relatively even distributed point.
    It start with all the unit vectors and then interpolates point on
    a sphere in between these points. The more interpolation steps you take the
    points you will have. Typically 4 interpolation steps is enough.

    input:
    - interpolation_count: the amount of interpolation steps
    - polarity: 1 is z>0, -1 is z<0
    output:
    - points: array with all the points
    - triangles: array with all the triangles that connect these point
            (these integer values refere to the points)

    '''

    # Starting point for sphere
    points = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
                       [0, 0, polarity]])
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    # refine mesh on sphere
    for n in range(interpolation_count):
        for i in range(triangles.shape[0]):
            # Make points on the edges of every triangle
            p1 = points[triangles[i, 0]] + points[triangles[i, 1]]
            p1 = p1 / np.linalg.norm(p1)
            p2 = points[triangles[i, 1]] + points[triangles[i, 2]]
            p2 = p2 / np.linalg.norm(p2)
            p3 = points[triangles[i, 2]] + points[triangles[i, 0]]
            p3 = p3 / np.linalg.norm(p3)
            # check if the new points are already in the list
            p1_exist = [np.all(points[j] == p1) for j in range(points.shape[0])]
            if any(p1_exist):
                index1 = np.where(p1_exist)[0][0]
            else:
                points = np.vstack([points, p1])
                index1 = points.shape[0] - 1
            p2_exist = [np.all(points[j] == p2) for j in range(points.shape[0])]
            if any(p2_exist):
                index2 = np.where(p2_exist)[0][0]
            else:
                points = np.vstack([points, p2])
                index2 = points.shape[0] - 1
            p3_exist = [np.all(points[j] == p3) for j in range(points.shape[0])]
            if any(p3_exist):
                index3 = np.where(p3_exist)[0][0]
            else:
                points = np.vstack([points, p3])
                index3 = points.shape[0] - 1
            # Create new triangles
            new_triangles = np.array(
                [[index1, index2, index3], [triangles[i, 0], index1, index3],
                 [triangles[i, 1], index1,
                  index2], [triangles[i, 2], index3, index2]])
            # add the new triangles to the triangle list
            if i == 0:
                triangles_temp = new_triangles
            else:
                triangles_temp = np.vstack([triangles_temp, new_triangles])
        # update the triangle list
        triangles = triangles_temp

    return points, triangles


def cart2spheric_matrix(x: np.array, y: np.array, z: np.array,
                        axis=2) -> sp.sparse.csr.csr_matrix:
    '''
    transformation matrix for vectors, for a cartesian to spherical coordinate
    system.

    input:
    - x, y, z positions
    output:
    - transformation matrix

    '''
    # get spherical coordinates
    r = (x**2 + y**2 + z**2)**0.5
    if axis == 2:
        th = np.arccos(z / r)
        ph = np.angle(x + 1j * y)
    elif axis == 1:
        th = np.arccos(y / r)
        ph = np.angle(z + 1j * x)
    elif axis == 0:
        th = np.arccos(x / r)
        ph = np.angle(y + 1j * z)

    # make transformation matrix assuming the z for theta
    t_3d = np.array(
        [[np.sin(th) * np.cos(ph),
          np.sin(th) * np.sin(ph),
          np.cos(th)],
         [np.cos(th) * np.cos(ph),
          np.cos(th) * np.sin(ph), -np.sin(th)], [-np.sin(ph),
                                                  np.cos(ph), 0]])
    # roll to get the axis correct
    t_3d = np.roll(t_3d, 2 - axis, axis=1)
    # make sparse transformation matrix
    t00 = sparse.csr_matrix(sp.sparse.diags(t_3d[0, 0] * np.ones_like(x)))
    t01 = sparse.csr_matrix(sp.sparse.diags(t_3d[0, 1] * np.ones_like(x)))
    t02 = sparse.csr_matrix(sp.sparse.diags(t_3d[0, 2] * np.ones_like(x)))
    t10 = sparse.csr_matrix(sp.sparse.diags(t_3d[1, 0] * np.ones_like(x)))
    t11 = sparse.csr_matrix(sp.sparse.diags(t_3d[1, 1] * np.ones_like(x)))
    t12 = sparse.csr_matrix(sp.sparse.diags(t_3d[1, 2] * np.ones_like(x)))
    t20 = sparse.csr_matrix(sp.sparse.diags(t_3d[2, 0] * np.ones_like(x)))
    t21 = sparse.csr_matrix(sp.sparse.diags(t_3d[2, 1] * np.ones_like(x)))
    t22 = sparse.csr_matrix(sp.sparse.diags(t_3d[2, 2] * np.ones_like(x)))

    t = sparse.bmat([[t00, t01, t02], [t10, t11, t12], [t20, t21, t22]])

    return t


def spheric2cart_matrix(r: np.array, th: np.array, ph: np.array,
                        axis=2) -> sp.sparse.csr.csr_matrix:
    '''
    transformation matrix for vectors, for a cartesian to spherical coordinate
    system.

    input:
    - x, y, z positions
    output:
    - transformation matrix

    '''
    # make transformation matrix assuming the z for theta
    t_3d = np.array(
        [[np.sin(th) * np.cos(ph),
          np.cos(th) * np.cos(ph), -np.sin(ph)],
         [np.sin(th) * np.sin(ph),
          np.cos(th) * np.sin(ph),
          np.cos(ph)], [np.cos(th), -np.sin(th), 0]])
    # roll to get the axis correct
    t_3d = np.roll(t_3d, 2 - axis, axis=0)

    # make sparse transformation matrix
    t00 = sparse.csr_matrix(sp.sparse.diags(t_3d[0, 0] * np.ones_like(x)))
    t01 = sparse.csr_matrix(sp.sparse.diags(t_3d[0, 1] * np.ones_like(x)))
    t02 = sparse.csr_matrix(sp.sparse.diags(t_3d[0, 2] * np.ones_like(x)))
    t10 = sparse.csr_matrix(sp.sparse.diags(t_3d[1, 0] * np.ones_like(x)))
    t11 = sparse.csr_matrix(sp.sparse.diags(t_3d[1, 1] * np.ones_like(x)))
    t12 = sparse.csr_matrix(sp.sparse.diags(t_3d[1, 2] * np.ones_like(x)))
    t20 = sparse.csr_matrix(sp.sparse.diags(t_3d[2, 0] * np.ones_like(x)))
    t21 = sparse.csr_matrix(sp.sparse.diags(t_3d[2, 1] * np.ones_like(x)))
    t22 = sparse.csr_matrix(sp.sparse.diags(t_3d[2, 2] * np.ones_like(x)))

    t = sparse.bmat([[t00, t01, t02], [t10, t11, t12], [t20, t21, t22]])

    return t


def spheric2spheric_matrix(r: np.array,
                           th: np.array,
                           ph: np.array,
                           initial_axis=2,
                           new_axis=2) -> sp.sparse.csr.csr_matrix:

    x = r * np.cos(ph) * np.sin(th)
    y = r * np.sin(ph) * np.sin(th)
    z = r * np.cos(th)

    sp2cart = spheric2cart_matrix(r, th, ph, initial_axis)
    cart2sp = cart2spheric_matrix(x, y, z, new_axis)

    t = cart2sp @ sp2cart

    return t


# Functions needed to plot the far field
########################################


def get_jet_colors(x: float) -> np.array:
    '''
    gives the rgb values of the jet colormap for a value x between 0 and 1
    '''
    r = np.array([0, 0, 0, 1, 1])
    g = np.array([0, 1, 1, 1, 0])
    b = np.array([1, 1, 0, 0, 0])
    x_rgb = np.linspace(0, 1, len(r))

    r = np.interp(x, x_rgb, r)
    g = np.interp(x, x_rgb, g)
    b = np.interp(x, x_rgb, b)

    return np.hstack([r, g, b])


def scatter_plot(ax, points: np.array, triangles: np.array, E2: np.array):
    '''
    plots scatter data
    Note: The axis has to be made with mpl_toolkits.mplot3d.Axes3D
    '''
    # get the maximum value of E2
    r_max = np.max(E2)
    for i in range(triangles.shape[0]):
        vtx = np.vstack([
            E2[int(triangles[i, 0])] * points[int(triangles[i, 0])],
            E2[int(triangles[i, 1])] * points[int(triangles[i, 1])],
            E2[int(triangles[i, 2])] * points[int(triangles[i, 2])]
        ])
        r = np.sum((np.sum(vtx, axis=0) / 3)**2)**0.5
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(colors.rgb2hex(get_jet_colors(r / r_max)))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)


# Analysis functions
#####################


def points2triangles_averaging_matrix(points: np.ndarray,
                                      triangles: np.ndarray):
    '''
    gives a matrix that when multiplied with vector that has a value for every
    point in points, it will produces a vector with the average value of every
    triangle
    '''
    flatten = lambda l: [item for sublist in l for item in sublist]
    rowind = flatten([[i] * 3 for i in range(triangles.shape[0])])
    colind = flatten(triangles)
    data = 1 / 3 * np.ones_like(rowind)

    av_matrix = sparse.csr_matrix((data, (rowind, colind)))

    return av_matrix


def area_selection_vector(points: np.ndarray, triangles: np.ndarray,
                          bound_th: List[float],
                          bound_ph: List[float]) -> np.ndarray:
    '''
    gives a vector that when multiplied with a vector that gives the value for
    every triangle gives you the integration over a spacial region defined by
    bound_th and bound_ph
    '''
    s = triangle_selection_vector(points, triangles, bound_th, bound_ph)
    area = triangle_area_vector(points, triangles)

    return s * area


def triangle_selection_vector(points: np.ndarray, triangles: np.ndarray,
                              bound_th: List[float],
                              bound_ph: List[float]) -> np.ndarray:
    '''
    Returns a vector with one entry per triangle. The ith entry is 1 if the
    center point of the ith triangle is located in the spatial region defined
    by bound_th and bound_ph; otherwise the ith entry is 0.
        (th values must be between 0 and pi, ph can be any value)
    '''
    triangle_center = (points[triangles[:, 0]] + points[triangles[:, 1]] +
                       points[triangles[:, 2]]) / 3
    s = np.zeros(triangles.shape)

    r = np.sum(triangle_center**2, axis=1)**0.5
    th = np.arccos(triangle_center[0:, 2] / r)
    ph = np.angle(triangle_center[0:, 0] + 1j * triangle_center[0:, 1])

    s_th = (th > bound_th[0]) & (th < bound_th[1])

    php = np.cos(ph) + 1j * np.sin(ph)
    ph0 = np.cos(bound_ph[0]) + 1j * np.sin(bound_ph[0])
    ph1 = np.cos(bound_ph[1]) + 1j * np.sin(bound_ph[1])
    if (np.diff(bound_ph)[0] <= np.pi):
        s_ph = np.logical_and(
            np.angle(php / ph0) < np.angle(ph1 / ph0),
            np.angle(php / ph0) > 0)
    else:
        s_ph = np.logical_not(
            np.logical_and(
                np.angle(php / ph1) < np.angle(ph0 / ph1),
                np.angle(php / ph1) > 0))

    return 1 * np.logical_and(s_th, s_ph)


def triangle_area_vector(points: np.ndarray,
                         triangles: np.ndarray) -> np.ndarray:
    '''
    gives a vector with the area of every triangle in triangles
    '''
    p0 = points[triangles[:, 1]] - points[triangles[:, 0]]
    p1 = points[triangles[:, 2]] - points[triangles[:, 0]]

    area = 0.5 * ((p0[:, 1] * p1[:, 2] - p0[:, 2] * p1[:, 1])**2 +
                  (p0[:, 2] * p1[:, 0] - p0[:, 0] * p1[:, 2])**2 +
                  (p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0])**2)**0.5

    return area


# main for quick test
######################
if __name__ == '__main__':

    P, C = make_sphere_point(4)
    x = np.array([1.])  #P[0:,0]
    y = np.array([1.])  #P[0:,1]
    z = np.array([1.])  #P[0:,2]
    r = (x**2 + y**2 + z**2)**0.5
    th = np.arctan(y / x)
    for i in range(len(x)):
        if np.isnan(th[i]):
            th[i] = np.sign(z[i])
    ph = np.arccos(z / r)
    T = spheric2spheric_matrix(r, th, ph, 2, 2)
    print(spheric2cart_matrix(r, th, ph, 2))
    print(cart2spheric_matrix(x, y, z, 2))
    print('T:')
    print(T)

    print('no test')
    P, C = make_sphere_point(4)
    a = triangle_area_vector(P, C)
    print(np.sum(a))

    av_matrix = points2triangle_averaging_matrix(P, C)
    s = area_selection_vector(P, C, [0, np.pi], [-np.pi / 2, np.pi / 2])
    print(av_matrix.shape)
    print(np.sum(s @ av_matrix))
    print(np.sum(s))

    s = triangle_selection_vector(P, C, [1 / 4 * np.pi, 3 / 4 * np.pi],
                                  [3 / 4 * np.pi, 5 / 4 * np.pi])

    fig = pl.figure()
    ax = a3.Axes3D(fig)
    S = sparse.csr_matrix(sparse.diags(s))
    s_n0, _ = S.nonzero()
    scatter_plot(ax, P, S[s_n0] @ C, np.ones_like(P[0:, 0]))
    pl.show()

    P, C = make_half_sphere_point(4, -1)
    fig = pl.figure()
    ax = a3.Axes3D(fig)
    S = sparse.csr_matrix(sparse.diags(s))
    s_n0, _ = S.nonzero()
    scatter_plot(ax, P, C, np.ones_like(P[0:, 0]))
    pl.show()
