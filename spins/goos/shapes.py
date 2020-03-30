from typing import Callable, List, Optional, Tuple, Union

import copy
import numpy as np

from spins import goos
from spins.goos import flows


class ShapeFlow(flows.Flow):
    pos: np.ndarray = flows.np_zero_field(3)
    rot: np.ndarray = flows.np_zero_field(3)
    priority: int = flows.constant_field(default=0)
    material: str = flows.constant_field(default=None)


class Shape(goos.ProblemGraphNode):
    node_type = "goos.shape"

    def translate(self, offset: np.ndarray) -> "Shape":
        return TranslateShape(self, offset)


class CuboidFlow(ShapeFlow):
    """Represents a rectangular prism.

    Attributes:
        extents: A 3 element array containing the length, width, and height
            (x-length, y-length, z-length) of the prism.
    """
    extents: List[float] = flows.np_zero_field(3)


class Cuboid(Shape):
    node_type = "goos.shape.cuboid"

    def __init__(self,
                 extents: goos.Function,
                 pos: goos.Function,
                 rot: goos.Function = None,
                 material: goos.material.Material = None) -> None:
        if rot is None:
            rot = goos.Constant([0, 0, 0])
        super().__init__([extents, pos, rot])
        self._mat = goos.material.get_material(material)

    def eval_const_flags(
        self, inputs: List[flows.NumericFlow.ConstFlags]
    ) -> CuboidFlow.ConstFlags:
        return CuboidFlow.ConstFlags(extents=inputs[0].array,
                                     pos=inputs[1].array,
                                     rot=inputs[2].array)

    def eval(self, inputs: List[flows.NumericFlow]) -> CuboidFlow:
        return CuboidFlow(extents=inputs[0].array,
                          pos=inputs[1].array,
                          rot=inputs[2].array,
                          material=self._mat)

    def grad(self, inputs: List[flows.NumericFlow],
             grad_val: CuboidFlow.Grad) -> List[flows.NumericFlow.Grad]:
        return [
            flows.NumericFlow.Grad(grad_val.extents_grad),
            flows.NumericFlow.Grad(grad_val.pos_grad),
            flows.NumericFlow.Grad(grad_val.rot_grad),
        ]


class CylinderFlow(ShapeFlow):
    """Represents a rectangular prism.

    Attributes:
        radius: The radius of the cylinder.
        height: The height of the cylinder.
        num_points: Number of points to use to approximate the cylinder.
    """
    radius: float = 100
    height: float = 100
    num_points: int = 32


class Cylinder(Shape):
    node_type = "goos.shape.cylinder"

    def __init__(self,
                 pos: goos.Function,
                 radius: goos.Function,
                 height: goos.Function,
                 rot: goos.Function = None,
                 material: goos.material.Material = None) -> None:
        if rot is None:
            rot = goos.Constant([0, 0, 0])
        super().__init__([pos, radius, height, rot])
        self._mat = goos.material.get_material(material)

    def eval_const_flags(
        self, inputs: List[flows.NumericFlow.ConstFlags]
    ) -> CylinderFlow.ConstFlags:
        return CylinderFlow.ConstFlags(pos=inputs[0].array,
                                       radius=inputs[1].array,
                                       height=inputs[2].array,
                                       rot=inputs[3].array)

    def eval(self, inputs: List[flows.NumericFlow]) -> CuboidFlow:
        return CylinderFlow(pos=inputs[0].array,
                            radius=inputs[1].array,
                            height=inputs[2].array,
                            rot=inputs[3].array,
                            material=self._mat)

    def grad(self, inputs: List[flows.NumericFlow],
             grad_val: CylinderFlow.Grad) -> List[flows.NumericFlow.Grad]:
        return [
            flows.NumericFlow.Grad(grad_val.pos_grad),
            flows.NumericFlow.Grad(grad_val.radius_grad),
            flows.NumericFlow.Grad(grad_val.height_grad),
            flows.NumericFlow.Grad(grad_val.rot_grad),
        ]


class PixelatedContShapeFlow(ShapeFlow, goos.NumericFlow):
    """Represents a shape with continuous permittivity distribution.

    A pixelated continuous shape flow splits up a rectangular block into pixels
    with size given by `pixel_size`. Each pixel has a given value as given by
    `array`. The centers of the pixels relative to the center of the shape
    can be computed by calling `get_relative_cell_coords`.

    Attributes:
        material2: The "upper" material. This material permittivity is used
            when the value is equal to 1 (`material` defined in `ShapeFlow`
            is used when the value is equal to 0).
        array: List of values corresponding to the normalized permittivity,
            usually between 0 and 1.
        extents: Extents of space that the shape occupies. The extruded
            direction is along z-axis.
        pixel_size: List of 3 values indicating size of the pixels along x, y,
            and z axes.
    """
    material2: str = flows.constant_field(default=None)
    extents: np.ndarray = flows.np_zero_field(3)
    pixel_size: np.ndarray = flows.constant_field(
        default_factory=lambda: np.zeros(3))

    @classmethod
    def get_relative_cell_coords(cls, extents: np.ndarray,
                                 pixel_size: np.ndarray) -> List[np.ndarray]:
        """Returns coordinates of the cell relative to the shape center.

        Args:
            extents: Extents of the shape.
            pixel_size: Size of each pixel.

        Returns:
            List `(x_coords, y_coords, z_coords)` where `x_coords` is a list of
            the x-coordinate centers and so forth.
        """
        edge_coords = cls.get_relative_edge_coords(extents, pixel_size)
        return [(coord[:-1] + coord[1:]) / 2 for coord in edge_coords]
        coords = []
        for i in range(len(extents)):
            coord = pixel_size[i] * np.arange(
                0,
                int(extents[i] / pixel_size[i]) + 1)
            if coord[-1] >= extents[i]:
                coords.append(coord[:-1])
            else:
                coords.append(coord)
        return [coord - np.mean(coord) for coord in coords]

    def get_cell_coords(self) -> List[np.ndarray]:
        """Returns the absolute coordinates of the cells."""
        rel_cell_coords = PixelatedContShapeFlow.get_relative_cell_coords(
            self.extents, self.pixel_size)
        return [coords + p for coords, p in zip(rel_cell_coords, self.pos)]

    @classmethod
    def get_relative_edge_coords(cls, extents: np.ndarray,
                                 pixel_size: np.ndarray) -> List[np.ndarray]:
        """Returns coordinates of the cell edges relative to the shape center.

        Args:
            extents: Extents of the shape.
            pixel_size: Size of each pixel.

        Returns:
            List `(x_coords, y_coords, z_coords)` where `x_coords` is a list of
            the x-coordinate edges and so forth.
        """
        # Compute edge coordinates assuming that each pixel is `pixel_size`
        # large.
        edge_coords = []
        for i in range(len(extents)):
            num_pixels = np.ceil(extents[i] / pixel_size[i])
            if num_pixels * pixel_size[i] < extents[i]:
                num_pixels += 1
            coord = pixel_size[i] * np.arange(0, num_pixels + 1)
            edge_coords.append(coord - (coord[0] + coord[-1]) / 2)

        # The edges could have smaller pixel sizes because they are cut off.
        # Account for this difference.
        for i in range(len(extents)):
            edge_coords[i][0] = -extents[i] / 2
            edge_coords[i][-1] = extents[i] / 2
        return edge_coords

    def get_edge_coords(self) -> List[np.ndarray]:
        """Returns the absolute coordinates of the cell edges."""
        rel_edge_coords = PixelatedContShapeFlow.get_relative_edge_coords(
            self.extents, self.pixel_size)
        return [coords + p for coords, p in zip(rel_edge_coords, self.pos)]

    @classmethod
    def get_shape(cls, extents: np.ndarray,
                  pixel_size: np.ndarray) -> List[int]:
        """Returns shape of the values array.

        Args:
            extents: Extents of the shape.
            pixel_size: Size of each pixel.

        Returns:
            List of the shape of the values array.
        """
        shape = []
        for i in range(3):
            size = int(extents[i] / pixel_size[i]) + 1
            if pixel_size[i] * (size - 1) >= extents[i]:
                size -= 1
            shape.append(size)
        return shape


class PixelatedContShape(Shape, goos.Function):
    node_type = "goos.shape.pixelated_cont_shape"

    def __init__(self,
                 array: goos.Function,
                 pixel_size: np.ndarray,
                 pos: goos.Function,
                 extents: np.ndarray,
                 rot: goos.Function = None,
                 material: goos.material.Material = None,
                 material2: goos.material.Material = None) -> None:
        if rot is None:
            rot = goos.Constant([0, 0, 0])
        super().__init__([pos, rot, array])

        self._pixel_size = pixel_size
        self._extents = extents
        self._mat = goos.material.get_material(material)
        self._mat2 = goos.material.get_material(material2)

    def eval_const_flags(
        self, inputs: List[flows.NumericFlow.ConstFlags]
    ) -> PixelatedContShapeFlow.ConstFlags:
        return PixelatedContShapeFlow.ConstFlags(pos=inputs[0].array,
                                                 rot=inputs[1].array,
                                                 array=inputs[2].array)

    def eval(self, inputs: List[flows.NumericFlow]) -> PixelatedContShapeFlow:
        return PixelatedContShapeFlow(pos=inputs[0].array,
                                      rot=inputs[1].array,
                                      array=inputs[2].array,
                                      material=self._mat,
                                      material2=self._mat2,
                                      pixel_size=self._pixel_size,
                                      extents=self._extents)

    def grad(
        self, inputs: List[flows.NumericFlow],
        grad_val: PixelatedContShapeFlow.Grad
    ) -> List[flows.NumericFlow.Grad]:
        return [
            flows.NumericFlow.Grad(grad_val.pos_grad),
            flows.NumericFlow.Grad(grad_val.rot_grad),
            flows.NumericFlow.Grad(grad_val.array_grad),
        ]


def pixelated_cont_shape(
        initializer: Callable,
        extents: np.ndarray,
        pixel_size: np.ndarray,
        pos: Union[np.ndarray, goos.Function],
        var_name: Optional[str] = None,
        **kwargs,
) -> Tuple[goos.Variable, PixelatedContShape]:
    """Creates a new `PixelatedContShape`.

    Args:
        initializer: A callable to initialize values for the shape. This should
            be function that accepts a single argument `size` and returns
            an array of values with shape `size`.
        extents: Extents of the shape.
        pixel_size: Size of each pixel.
        var_name: Name to give the variable (setting `name` argument sets the
            name of the shape).
        pos: Position of the shape.
        kwargs: Additional arguments to pass to the shape constructor.

    Returns:
        A tuple `(var, shape)` where `var` is the variable containing the values
        and `shape` is the newly created shape.
    """
    cell_coords = PixelatedContShapeFlow.get_relative_cell_coords(
        extents, pixel_size)
    values = initializer(
        (len(cell_coords[0]), len(cell_coords[1]), len(cell_coords[2])))
    var = goos.Variable(np.array(values),
                        name=var_name,
                        lower_bounds=0,
                        upper_bounds=1)

    if not isinstance(pos, goos.Function):
        pos = goos.Constant(pos)
    return var, PixelatedContShape(array=var,
                                   extents=extents,
                                   pixel_size=pixel_size,
                                   pos=pos,
                                   **kwargs)


class GroupShape(goos.ArrayFlowOpMixin, Shape):
    node_type = "goos.shape.group"

    def __init__(self, shapes: List[Shape]) -> None:
        output_flow_types = [type(shape) for shape in shapes]
        super().__init__(shapes, flow_types=output_flow_types)

    def eval_const_flags(
            self,
            inputs: List[ShapeFlow.ConstFlags]) -> goos.ArrayFlow.ConstFlags:
        return goos.ArrayFlow.ConstFlags(inputs)

    def eval(self, inputs: List[ShapeFlow]) -> goos.ArrayFlow:
        return goos.ArrayFlow(inputs)

    def grad(self, inputs: List[ShapeFlow],
             grad_flow: goos.ArrayFlow.Grad) -> List[ShapeFlow.Grad]:
        return [grad_flow[i] for i in range(len(inputs))]


class TranslateShape(Shape):
    node_type = "goos.shape.translate"

    def __init__(self, shape: Shape, offset: np.ndarray) -> None:
        super().__init__(shape)
        self._offset = offset

    def eval_const_flags(
            self, inputs: List[ShapeFlow.ConstFlags]) -> ShapeFlow.ConstFlags:
        return inputs[0]

    def eval(self, inputs: List[ShapeFlow]) -> ShapeFlow:
        flow = copy.deepcopy(inputs[0])
        flow.pos += self._offset
        return flow

    def grad(self, inputs: List[ShapeFlow],
             grad_val: ShapeFlow.Grad) -> List[ShapeFlow.Grad]:
        return [grad_val]
