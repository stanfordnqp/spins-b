"""Module to implement a composite parametrization.

A composite parametrization combines multiple parametrization objects into the
same parametrization object. The need for a composite parametrization may arise
for e.g. if one is designing a nanophotonic device with multiple design regions
(e.g. multiple layers), with each of the design regions being parametrized by
an individual parametrization. In such a case, the permittivity `eps` of the
structure is given by:
            `eps = eps_bg + S_1 * z_1 + S_2 * z_2 .... S_N * z_N`
where `eps_bg` is the background permittivity distribution, `z_i` are the
structure vectors corresponding to the ith design region and `S_i` is the
selection matrix corresponding to the ith design region. Alternatively, this
computation can be viewed as:
                                                   _     _ `
                             _                  _ |  z_1  |
            `eps = eps_bg + |_ S_1  S_2 ... S_N _||  z_2  |
                                                  |  ...  |
                                                  |_ z_N _|
Thus, multiple parametrizations can be wrapped into a single parametrization
object, whose parameters are a concatenation of the individual parameters and
which returns a `z` vector which is a concatenation of individual `z` vectors.
Note that it is the responsibility of the user to use this parametrization with
the appropriately constructed selection matrix.
"""
from typing import List, Tuple, Dict

import numpy as np
import scipy.sparse.linalg as linalg

from spins.invdes.parametrization import parametrization


def _split_vector(vec: np.ndarray,
                  sizes: List[int]) -> List[np.ndarray]:
    """Splits a vector `vec` into vectors of sizes given by `sizes`.

    Args:
        vec: A 1D numpy array.
        sizes: A list of integers giving the sizes of the vector into which
            `vec` needs to be split.

    Returns:
        A list of numpy arrays obtain on splitting `vec` into sub vectors of
        sizes given by `sizes`.
    """
    # Calculate the indices of the split. This is obtained by computing the
    # cumulative sum of the `sizes`, and throwing away the last element.
    # This is done because `np.split`, for a given set of indices
    # `[i_1, i_2 ... i_N]` splits a vector as `vec[:i_1]`, `vec[i_1:i_2]` ...
    # `vec[i_N:]`.
    split_indices = np.cumsum(sizes)[:-1].tolist()

    # Split the numpy array.
    return np.split(vec, split_indices)


class CompositeParam(parametrization.Parametrization):
    """Class implementing composite parametrization."""
    def __init__(self,
                 params: List[parametrization.Parametrization]) -> None:
        """Create a new `CompositeParam` object.

        Args:
            params: A list of parametrization objects that are used to compose
                the composite parametrization object. Note that the order of
                parametrization objects in the list governs the order in which
                the parametrization are concated.
        """
        # List of parametrizations composing the composite parametrization.
        self._params = params

        # Construct a list of sizes vectors corresponding to the individual
        # parametrization by calling the `encode` function of the individual
        # parametrization, and then calculating the size of the output vector.
        # Note that this list of sizes is necessary for the `decode` function
        # of the composite parametrization, since that would need to take as an
        # input a huge vector that is the concatenation of all parameters of the
        # individual parametrization, and split it into smaller vectors to pass
        # to the individual parametrization objects.
        self._param_sizes = [param.encode().size for param in self._params]

        # In addition to the sizes of the `encode` function, parametrizations
        # may choose to implement `to_vector` that converts the parametrization
        # into a vector to be used by external clients. In order to implement
        # the `from_vector` function (which converts an external vector to the
        # parametrization), we need the sizes of the vectors produced and
        # processed by the individual `to_vector` and `from_vector` functions.
        self._ext_vec_sizes = [param.to_vector().size for param in self._params]

        # Compute the total size of the `z` vector corresponding to the
        # composite parametrization.
        self._z_size = np.sum([param.get_structure().size
                               for param in self._params])

    def get_structure(self) -> np.ndarray:
        """Calculate the `z` vector for the parametrization.

        Returns:
            A numpy array corresponding to the structure for the composite
            parametrization. As is described in the module docstring, this is
            done by simply concatenating the structure vectors of the individual
            parametrization together.
        """
        return np.hstack([param.get_structure() for param in self._params])

    def calculate_gradient(self) -> linalg.LinearOperator:
        """Calculate the jacobian of `z` with respect to the parameters.

        The `z` vector for the composite parametrization is a concatenation of
        the individual `z_i` vectors. The parameter vector `p` for the composite
        parametrization is also a concatention of individual parameter vectors
        `p_i`. Thus, the jacobian of `z` with respect to `p` can be obtained by
        computing the jacobian of `z_i` with respect to `p_i`, and laying it
        along the diagonal of the full jacobian:
                `dz / dp = diag([dz_1 / dp_1, dz_2 / dp_2 .... dz_N / dp_N])

        Returns:
            A linear operator corresponding to the jacobian corresponding to the
            composite parametrization.
        """
        def _jacobian_vec_prod(vec: np.ndarray) -> np.ndarray:
            """Calculate the product of the jacobian with a given vector.

            This function computes the product of the jacobian corresponding to
            the composite parametrization with a given vector. Since the
            jacobian of the composite parametrization is simply the jacobian of
            the individual parametrizations laid along the diagonal, this
            computations proceeds by splitting the vector `vec` into smaller
            vectors corresponding to the individual parametrization, and then
            computing the matrix-vector products of the individual jacobians
            with these smaller vectors.

            Args:
                 vec: The numpy array to multiply the jacobian to.

            Returns:
                 The product of the `jacobian` with the vector `vec`.
            """
            # Split the vector into smaller vectors.
            split_vec = _split_vector(vec, self._param_sizes)

            # Compute jacobian-vector product.
            return np.hstack([param.calculate_gradient() @ sub_vec for
                              param, sub_vec in zip(self._params, split_vec)])

        return linalg.LinearOperator((self._z_size, np.sum(self._param_sizes)),
                                     matvec=_jacobian_vec_prod)

    def encode(self) -> np.ndarray:
        """Encode the parametrization into a vector form.

        Returns:
            A 1D numpy array corresponding to the current state of the
            parametrization.
        """
        return np.hstack([param.encode() for param in self._params])

    def decode(self, vector: np.ndarray):
        """Decode the parametrization from a vector form.

        Args:
            vector: The vector to be decoded into the parametrization.
        """
        split_vector = _split_vector(vector, self._param_sizes)
        for param, sub_vec in zip(self._params, split_vector):
            param.decode(sub_vec)

    def project(self) -> None:
        """Projects the parametrization into its feasible sset."""
        for param in self._params:
            param.project()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get box constraints on the composite parametrization.

        Returns a tuple of two 1D numpy arrays corresponding to the upper and
        lower bounds on the parametrization.

        Returns:
            A tuple of two numpy arrays which correspond to the lower and upper
            bounds to the parametrization.
        """
        lower_bounds = []
        upper_bounds = []
        for param, size in zip(self._params, self._param_sizes):
            # We take care the handle the case if there is no bound in any one
            # of the parametrization. If `param` returns `None` for bounds, then
            # we append a numpy array of `None` for both lower and upper bounds.
            # This is necessary to handle the case where one of the
            # parametrization objects might have no bounds, whereas the rest
            # might have bounds.
            if param.get_bounds():
                lower_bounds_param, upper_bounds_param = param.get_bounds()
                lower_bounds.append(lower_bounds_param)
                upper_bounds.append(upper_bounds_param)
            else:
                lower_bounds.append(np.array([None] * size))
                upper_bounds.append(np.array([None] * size))

        return np.hstack(lower_bounds), np.hstack(upper_bounds)

    def to_vector(self) -> np.ndarray:
        """Convert a parametrization to vector representation.

        Refer to the documentation of the corresponding function in
        `Parametrization` in `spins/invdes/parametrization/parametrization.py`
        for the difference between this function and `encode`.

        Returns:
            A numpy array corresponding to the vector representation of the
            parametrization.
        """
        return np.hstack([param.to_vector() for param in self._params])

    def from_vector(self, vector: np.ndarray):
        """Convert a vector representation into parametrization.

        Refer to the documentation of the corresponding function in
        `Parametrization` in `spins/invdes/parametrization/parametrization.py`
        for the differences between this function and `decode`.

        Args:
            The vector to be converted into the parametrization.
        """
        split_vector = _split_vector(vector, self._ext_vec_sizes)
        for param, sub_vec in zip(self._params, split_vector):
            param.from_vector(sub_vec)

    def serialize(self) -> Dict:
        """Serialize the parametrization information.

        Serialize here returns a nested dictionary, with the keys of the outer
        dictionary being labelled as `param_i`, and the values being
        dictionaries obtained by serializing the individual parametrizations.

        Returns:
            A dictionary corresponding to the serialized version of the
            parametrization.
        """
        return {"param_{}".format(k): param.serialize() for
                k, param in enumerate(self._params)}

    def deserialize(self, data: Dict):
        """Desrializes the data that was `serialized` by `serialize`.

        Args:
            data: The serialized data that needs to be parsed.
        """
        for param_key, param_data in data.items():
            # Strip off the "param_" prefix to compute index.
            ind = int(param_key[len("param_"):])
            self._params[ind].deserialize(param_data)
