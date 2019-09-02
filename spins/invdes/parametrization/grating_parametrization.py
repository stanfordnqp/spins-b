"""Module for discrete parametrization for grating designs."""
from typing import Dict
import numpy as np
import scipy.sparse.linalg as linalg

from spins.gridlock import float_raster as raster
from spins.invdes.parametrization import parametrization


class GratingParam(parametrization.Parametrization):
    """Parametrizes the grating coupler with the positions of its edges.

    A grating structure can be parametrized with the location of its edges.
    During optimization, the edges can be changed continuously using a
    gradient based optimization. Fabrication constraints on the grating
    coupler can be imposed by requiring that the edges are separated by the
    minimum feature size that is desired.

    In the scheme of inverse-design, the design region can be described by a
    `z` vector, to which the permittivity distribution is related via a
    selection matrix `S` and background permittivity distribution `eps_bg`:
                              `eps = S * z + eps_bg`
    The vector `z`, for gratings, is a 1D array of each pixel in the grating
    region. Each pixel can either be a 0 or 1 corresponding to one material or
    the other (note that the pixels which have the grating edges in them can
    have intermediate values to alias the edge properly). A grating can then be
    parametrized by a list of edges, from which the vector `z` can be computed.
    This parametrization assumes that the list of edges has an even number of
    elements `[p_0, p_1 .... p_{N - 1}]` (where `N` is even), with the first
    edge is a rising edge and the last edge is a falling edge - for e.g. `z`
    corresponding to `N = 8` is given by the following diagram:
                ____     ________          ______             ____
               |    |   |        |        |      |           |    |
             __|    |___|        |________|      |___________|    |__
              p_0  p_1  p_2     p_3      p_4    p_5         p_6  p_7
    It is assumed that the edges are specified in a coordinate system where the
    starting edge of the first pixel is at 0, and the width of each pixel is 1.
    """

    def __init__(self,
                 initial_value: np.ndarray,
                 num_pixels: int,
                 inverted: bool = False) -> None:
        """Creates a new `GratingParam` object.

        Args:
            initial_value: The initial value of the edges for the
                parametrization.
            num_pixels: The number of pixels in the design region.
            inverted: If `True`, invert the pixel values so that each of pair
                of edges correspond to a "hole". That is, p_0 is a falling edge
                and p_1 is a rising edge, etc.

        Raises:
            ValueError: If the number of edges (size of `initial_value`) is not
                even.
        """
        # Validate that the number of edges are correct.
        if len(initial_value) % 2 == 1:
            raise ValueError("The number of edges in the grating expected to "
                             "be even, got {} instead.".format(
                                 len(initial_value)))

        # `self._edges` holds the state of the parametrization. The edges are
        # stored in ascending order.
        self._edges = np.sort(initial_value)
        self._num_pixels = num_pixels

        # Defining the grid on which rendering the grating edges allows us to
        # compute the values of the `pixels` in the grating region. This grid
        # has edges at `[0, 1, 2 ... num_pixels]`.
        self._x_coords = np.arange(num_pixels + 1)

        self._inverted = inverted

    def get_structure(self) -> np.ndarray:
        """Calculate the value of `z` from the edges.

        This function renders the grating structure specified as a list of
        edges onto the pixels in the grating.

        Returns:
            A 1D numpy array of size `num_pixels` corresponding to the rendered
            structure.
        """
        # Compute the widths and the centers of the grating teeth i.e. regions
        # which have a value of 1.
        widths = self._edges[1::2] - self._edges[0::2]
        centers = 0.5 * (self._edges[1::2] + self._edges[0::2])

        # Initialize the pixels in the structure to 0.
        pixel_vals = np.zeros(self._num_pixels)

        # Render the grating structure.
        for width, center in zip(widths, centers):
            pixel_vals += raster.raster_1D(
                np.array([center - 0.5 * width, center + 0.5 * width]),
                self._x_coords)

        if self._inverted:
            return 1 - pixel_vals
        return pixel_vals

    def calculate_gradient(self) -> linalg.LinearOperator:
        """Compute the gradient of the structure `z` with respect to edges.

        The gradient of an element of `z` with respect to an edge is 0 if the
        edge doesn't lie in the pixel corresponding to the element. If the
        pixel does have the edge, then the gradient is -1 if the edge is a
        rising edge, and 1 if the edge is a falling edge. Note that if the
        edge lies right on the border of the pixel under consideration, then
        the derivative of the corresponding element of `z` is not defined.
        To obviate this issue, we compute a one-sided derivative of the `z`
        with respect to the edges. In this implementation, we compute the
        forward derivative of `z` with respect to the edges.

        Returns:
            The gradient of `z` with respect to the pixels as a linear operator.
        """

        # If `z = [z_0, z_1 ... z_{M - 1}]` and the edges are given by
        # `[p_0, p_1 ... p_{N - 1}]`, the jacobian of `z` with respect to `p`
        # is a matrix `J` with elements `J_ij` given by:
        #                       `J_ij = dz_i / dp_j`
        # To construct `J` as a Linear operator, it is needed to implement a
        # function that can compute `G * x` for an arbitrary vector `x`.
        def _jacobian_vec_prod(vec: np.ndarray) -> np.ndarray:
            """Compute the product of the jacobian with a vector.

            The jacobian vector product is computed as follows - Note that the
            vector is a vector of length equal to the number of edges. For each
            pixel `i`, we compute the edges `j_i` which lie in the pixel, then
                            `(J * x)_i = -sum_j f(j_i) x_j`
            where `f(j)` is 1 if the jth edge is a rising edge and -1 if the
            jth edge is a falling edge.

            Note, if `p_i` falls on the boundary between two `pixels`, since we
            implement a forward derivative
            (i.e. `df / dx = [f(x + dx) - f(x)] / dx`), we use `j` as the pixel
            that lies to the right of the edge.

            Args:
                vec: The vector that is to be multiplied by the jacobian.

            Returns:
                The result of the jacobian-vector product.
            """
            # Initialize jacobian vector product.
            jac_vec_prod = np.zeros(self._num_pixels)

            # Construct the jacobian vector product.
            for edge_index, (edge, vec_val) in enumerate(zip(self._edges, vec)):
                # `edge_dir` is 1 if the edge is a rising edge, and -1 if it
                # is a falling edge.
                edge_dir = 1 if edge_index % 2 == 0 else -1
                if self._inverted:
                    edge_dir = -edge_dir
                # Compute the pixel index corresponding to `edge`.
                pixel_index = np.floor(edge).astype(int)
                # Update the `jac_vec_prod` only if the edge lies inside the
                # grating region - i.e. `edge` should be in between `0` and
                # `self._num_pixels`.
                if 0 <= edge < self._num_pixels:
                    jac_vec_prod[pixel_index] -= edge_dir * vec_val

            return jac_vec_prod

        # TODO(logansu): Just explicitly construct the Jacobian matrix rather
        # than relying on `_jacobian_vec_prod`. We keep this here for now
        # simply because we need to hack the Jacobian due to the selection
        # matrix.
        mat = np.zeros((self._num_pixels, self._edges.size))
        for i in range(self._edges.size):
            unit_vec = np.zeros(self._edges.size)
            unit_vec[i] = 1
            mat[:, i] = _jacobian_vec_prod(unit_vec)
        return mat

    def encode(self) -> np.ndarray:
        """Encode the parametrization into a vector form.

        Returns:
            The vector corresponding to the current state of the
            parametrization.
        """
        return self._edges

    def decode(self, vector: np.ndarray) -> None:
        """Decide the parametrization from vector form.

        Args:
            vector: The vector to be decoded.
        """
        self._edges = np.sort(vector)

    def serialize(self) -> Dict:
        serialized = super().serialize()
        serialized.update({"inverted": self._inverted})
        return serialized

    def deserialize(self, data: Dict) -> None:
        super().deserialize(data)

        if "inverted" in data:
            self._inverted = data["inverted"]
        else:
            self._inverted = False
