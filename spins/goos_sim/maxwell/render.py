from typing import List

import numpy as np

from spins import goos
from spins.goos import flows
from spins.goos_sim.maxwell import simspace
from spins import gridlock


class RenderShape(goos.Function):
    node_type = "sim.maxwell.render"

    def __init__(
            self,
            shape: goos.Shape,
            region: goos.Box3d,
            mesh: simspace.MeshModel,
            wavelength: float,
            background: goos.material.Material = None,
            simulation_symmetry: List[int] = None,
            num_points_per_arclen: float = 0.084,
    ) -> None:
        super().__init__(shape)
        self._edge_coords = simspace.create_edge_coords(region, mesh.dx,
                                                        simulation_symmetry)
        if background:
            bg_eps = goos.material.get_material(background).permittivity(
                wavelength)
        else:
            bg_eps = 0
        self._grid = gridlock.Grid(self._edge_coords,
                                   ext_dir=gridlock.Direction.z,
                                   initial=bg_eps,
                                   num_grids=3)
        self._wlen = wavelength
        self._pts_per_arclen = num_points_per_arclen

    def eval(self, inputs: List[goos.ShapeFlow]) -> flows.NumericFlow:
        self._grid.clear()
        extra_grids = self._render(self._grid, inputs[0])
        self._grid.render()

        grids = self._grid.grids
        for grid in extra_grids:
            for i in range(3):
                grids[i] += grid.grids[i]

        return flows.NumericFlow(grids)

    def grad(self, input_vals: List[goos.Shape],
             grad_val: goos.NumericFlow.Grad) -> List[goos.ShapeFlow.Grad]:
        self._grid.clear()
        self._grid.grids = grad_val.array_grad

        return [_grad(self._grid, input_vals[0], self._wlen)]

    def _render(self, grid, shape) -> List[gridlock.Grid]:
        extra_grids = []
        # TODO(logansu): Warn about rotation.
        #if not np.array_equal(shape.rot, [0, 0, 0]):
        #    raise NotImplemented("Render cannot handle objects with rotation.")

        if isinstance(shape, goos.CuboidFlow):
            if np.all(shape.extents != 0):
                grid.draw_cuboid(shape.pos, shape.extents,
                                 shape.material.permittivity(self._wlen))
        elif isinstance(shape, goos.CylinderFlow):
            radius = shape.radius.item()
            num_points = int(np.ceil(self._pts_per_arclen * 2 * np.pi * radius))
            grid.draw_cylinder(shape.pos, radius, shape.height.item(),
                               num_points,
                               shape.material.permittivity(self._wlen))
        elif isinstance(shape, goos.ArrayFlow):
            for s in shape:
                extra_grids += self._render(grid, s)
        elif isinstance(shape, goos.PixelatedContShapeFlow):
            # Draw a cuboid in the original grid to overwrite any shapes in the
            # shape region, but draw the continuous permittivity on an extra
            # grid.
            grid.draw_cuboid(shape.pos, shape.extents,
                             shape.material.permittivity(self._wlen))
            new_grid = gridlock.Grid(grid.exyz,
                                     ext_dir=grid.ext_dir,
                                     initial=0,
                                     num_grids=3)
            contrast = shape.material2.permittivity(
                self._wlen) - shape.material.permittivity(self._wlen)
            shape_coords = shape.get_edge_coords()
            for axis in range(3):
                grid_coords = new_grid.shifted_exyz(
                    axis, which_grid=gridlock.GridType.COMP)
                # Remove ghost cells at the end.
                grid_coords = [
                    c if c.shape == co.shape else c[:-1]
                    for c, co in zip(grid_coords, grid.exyz)
                ]
                mat = get_rendering_matrix(shape_coords, grid_coords)
                grid_vals = contrast * mat @ shape.array.flatten()
                new_grid.grids[axis] = np.reshape(grid_vals,
                                                  new_grid.grids[axis].shape)
            extra_grids.append(new_grid)
        else:
            raise ValueError("Encountered unrenderable type, got {}".format(
                type(shape)))
        return extra_grids


def _grad(grid, shape, wlen) -> goos.ShapeFlow.Grad:
    if isinstance(shape, goos.CuboidFlow):
        return None
    elif isinstance(shape, goos.CylinderFlow):
        return None
    elif isinstance(shape, goos.ArrayFlow):
        return goos.ArrayFlow.Grad([_grad(grid, s, wlen) for s in shape])
    elif isinstance(shape, goos.PixelatedContShapeFlow):
        contrast = shape.material2.permittivity(
            wlen) - shape.material.permittivity(wlen)
        shape_coords = shape.get_edge_coords()
        grad = np.zeros_like(shape.array)
        for axis in range(3):
            grid_coords = grid.shifted_exyz(axis,
                                            which_grid=gridlock.GridType.COMP)
            # Remove ghost cells at the end.
            grid_coords = [
                c if c.shape == co.shape else c[:-1]
                for c, co in zip(grid_coords, grid.exyz)
            ]
            mat = get_rendering_matrix(shape_coords, grid_coords)
            grid_vals = contrast * mat.T @ grid.grids[axis].flatten()
            grad += np.real(np.reshape(grid_vals, shape.array.shape))
            # TODO(logansu): Fix complex gradient.
            if np.iscomplexobj(grid_vals):
                grad *= 2
        return goos.PixelatedContShapeFlow.Grad(array_grad=grad)
    else:
        raise ValueError("Encountered unrenderable type, got {}".format(
            type(shape)))


import scipy.sparse


def get_rendering_matrix(shape_edge_coords, grid_edge_coords):
    mats = [
        get_rendering_matrix_1d(se, re)
        for se, re in zip(shape_edge_coords, grid_edge_coords)
    ]
    return scipy.sparse.kron(scipy.sparse.kron(mats[0], mats[1]), mats[2])


def get_rendering_matrix_1d(shape_coord, grid_coord):
    weights = []
    grid_inds = []
    shape_inds = []

    edge_inds = np.digitize(shape_coord, grid_coord)
    for i, (start_ind, end_ind) in enumerate(zip(edge_inds[:-1],
                                                 edge_inds[1:])):
        # Shape is outside of the first grid cell.
        if end_ind < 1:
            continue
        last_coord = shape_coord[i]
        for j in range(start_ind, end_ind):
            if j >= 1:
                weights.append((grid_coord[j] - last_coord) /
                               (grid_coord[j] - grid_coord[j - 1]))
                grid_inds.append(j - 1)
                shape_inds.append(i)
            last_coord = grid_coord[j]

        if last_coord != shape_coord[i + 1] and end_ind < len(grid_coord):
            weights.append((shape_coord[i + 1] - last_coord) /
                           (grid_coord[end_ind] - grid_coord[end_ind - 1]))
            grid_inds.append(end_ind - 1)
            shape_inds.append(i)
    return scipy.sparse.csr_matrix(
        (weights, (grid_inds, shape_inds)),
        shape=(len(grid_coord) - 1, len(shape_coord) - 1))
