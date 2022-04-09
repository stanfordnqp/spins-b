from typing import List, Union

import dataclasses

import numpy as np
import scipy.sparse

from spins import goos
from spins.goos import flows
from spins.goos_sim import maxwell
from spins.goos_sim.maxwell import simspace
from spins import gridlock


@dataclasses.dataclass
class RenderParams:
    wlen: float
    pts_per_arclen: float


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

        self._render_params = RenderParams(wlen=wavelength,
                                           pts_per_arclen=num_points_per_arclen)

    def eval(self, inputs: List[goos.ShapeFlow]) -> flows.NumericFlow:
        self._grid.clear()
        # Save geometry for backprop.
        self._geom = _create_geometry(inputs[0])
        extra_grids = self._geom.eval(self._grid, self._render_params)
        self._grid.render()

        if extra_grids is None:
            extra_grids = []
        elif not isinstance(extra_grids, list):
            extra_grids = [extra_grids]

        grids = self._grid.grids
        for grid in extra_grids:
            for i in range(3):
                grids[i] += grid.grids[i]

        return flows.NumericFlow(grids)

    def grad(self, input_vals: List[goos.Shape],
             grad_val: goos.NumericFlow.Grad) -> List[goos.ShapeFlow.Grad]:
        self._grid.clear()
        self._grid.grids = grad_val.array_grad

        return [self._geom.grad(self._grid, self._render_params)]


class GeometryImpl:

    def __init__(self, shape):
        self.shape = shape

    def eval(self, grid: gridlock.Grid, params: RenderParams):
        pass

    def grad(self, grid: gridlock.Grid, params: RenderParams):
        pass


@maxwell.register(goos.CuboidFlow)
class CuboidFlowImpl(GeometryImpl):

    def eval(self, grid: gridlock.Grid, params: RenderParams):
        if np.all(self.shape.extents != 0):
            grid.draw_cuboid(self.shape.pos, self.shape.extents,
                             self.shape.material.permittivity(params.wlen))


@maxwell.register(goos.CylinderFlow)
class CylinderFlowImpl(GeometryImpl):

    def eval(self, grid: gridlock.Grid, params: RenderParams):
        radius = self.shape.radius.item()
        num_points = int(np.ceil(params.pts_per_arclen * 2 * np.pi * radius))
        grid.draw_cylinder(self.shape.pos, radius, self.shape.height.item(),
                           num_points,
                           self.shape.material.permittivity(params.wlen))


@maxwell.register(goos.PixelatedContShapeFlow)
class PixelatedContShapeFlowImpl(GeometryImpl):

    def eval(self, grid: gridlock.Grid, params: RenderParams):
        # Draw a cuboid in the original grid to overwrite any shapes in the
        # shape region, but draw the continuous permittivity on an extra
        # grid.
        grid.draw_cuboid(self.shape.pos, self.shape.extents,
                         self.shape.material.permittivity(params.wlen))
        new_grid = gridlock.Grid(grid.exyz,
                                 ext_dir=grid.ext_dir,
                                 initial=0,
                                 num_grids=3)
        contrast = self.shape.material2.permittivity(
            params.wlen) - self.shape.material.permittivity(params.wlen)
        shape_coords = self.shape.get_edge_coords()
        for axis in range(3):
            grid_coords = new_grid.shifted_exyz(
                axis, which_grid=gridlock.GridType.COMP)
            # Remove ghost cells at the end.
            grid_coords = [
                c if c.shape == co.shape else c[:-1]
                for c, co in zip(grid_coords, grid.exyz)
            ]
            mat = get_rendering_matrix(shape_coords, grid_coords)
            grid_vals = contrast * mat @ self.shape.array.flatten()
            new_grid.grids[axis] = np.reshape(grid_vals,
                                              new_grid.grids[axis].shape)
        return new_grid

    def grad(self, grid: gridlock.Grid, params: RenderParams):
        contrast = self.shape.material2.permittivity(
            params.wlen) - self.shape.material.permittivity(params.wlen)
        shape_coords = self.shape.get_edge_coords()
        grad = np.zeros_like(self.shape.array)
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
            grad += np.real(np.reshape(grid_vals, self.shape.array.shape))
            # TODO(logansu): Fix complex gradient.
            if np.iscomplexobj(grid_vals):
                grad *= 2

        return goos.PixelatedContShapeFlow.Grad(array_grad=grad)


@maxwell.register(goos.ArrayFlow)
class ArrayFlowImpl(GeometryImpl):

    def __init__(self, flow: goos.ArrayFlow):
        self._shapes = [_create_geometry(f) for f in flow]

    def eval(self, grid: gridlock.Grid, params: RenderParams):
        extra_grids = []
        for s in self._shapes:
            extra_grid = s.eval(grid, params)
            if extra_grid:
                extra_grids.append(extra_grid)
        return extra_grids

    def grad(self, grid: gridlock.Grid, params: RenderParams):
        return goos.ArrayFlow.Grad([s.grad(grid, params) for s in self._shapes])


def _create_geometry(shape: Union[goos.ArrayFlow, goos.Shape]) -> GeometryImpl:
    for flow_name, flow_entry in reversed(
            maxwell.GEOM_REGISTRY.get_map().items()):
        if isinstance(shape, flow_entry.schema):
            return flow_entry.creator(shape)

    raise ValueError("Encountered unrenderable type, got {}.".format(
        type(shape)))


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


if __name__ == "__main__":
    pass
