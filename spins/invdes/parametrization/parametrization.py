"""
Defines parametrizations of the structure. A parametrization is a mapping from a
set of values to a structure (z-values).
"""
import abc
import numbers
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

from spins.invdes.parametrization import cubic_utils


class Parametrization(metaclass=abc.ABCMeta):
    """Represents an abstract parametrization.

    All parametrizations should inherit from this class.
    """
    @abc.abstractmethod
    def get_structure(self) -> np.ndarray:
        """Produces the corresponding structure.

        `get_structure` assumes that the parametrization values represent a
        feasible structure. Call `project` before calling `get_structure` if
        there is a possibility that the parametrization values may represent an
        infeasible structure.

        Returns:
            A vector corresponding to the z.
        """
        raise NotImplementedError('get_structure method not defined')

    @abc.abstractmethod
    def calculate_gradient(self) -> LinearOperator:
        """Calculates the gradient of the parametrization.

        Note that implementations should consider caching the gradient if the
        operation is expensive.

        Returns:
            A linear operator that represents the Jacobian of the
            parametrization.
        """
        raise NotImplementedError('calculate_gradient not defined')

    def project(self) -> None:
        """Projects the parametrization to a feasible structure.

        The parametrization may be modified to have values that do not
        correspond to a feasible structure. Calling this method before
        get_structure causes the parametrization values to be modified so that
        the structure is in the feasible set.
        """
        pass

    def get_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:  # pylint: disable=R0201
        """Gets box constraints for the parametrization.

        Return a list of lower and upper bounds for each entry of the
        parametrization vector. Return `None` for a bound if it is
        unbounded. Return `None` for bounds if there are no bounds at all.
        """
        return None

    @abc.abstractmethod
    def encode(self) -> np.ndarray:
        """Encode the parametrization into vector form. """
        raise NotImplementedError('encode not implemented')

    @abc.abstractmethod
    def decode(self, vector: np.ndarray) -> None:
        """Decode the parametrization from vector form. """
        raise NotImplementedError('decode not implemented')

    def to_vector(self) -> np.ndarray:
        """Converts parametrization to vector representation.

        to_vector/from_vector are intended to be called by external clients.
        encode/decode are meant to be called internally.

        The difference between encode/decode and to_vector/from_vector
        lies in the fact that encode/decode are guaranteed to be symmetric
        whereas to_vector/from_vector need not be, i.e. decode(encode())
        should be an effective no-op whereas from_vector(to_vector()) might
        not be. Specifically, encode/decode converts between the parametrization
        and the raw vector, which may contain invalid values (e.g. negative
        radius). On the other hand, to_vector/from_vector guarantee that the
        parametrization is valid.
        """
        return self.encode()

    def from_vector(self, vector: np.ndarray) -> None:
        """Converts vector representation into parametrization. """
        self.decode(vector)
        self.project()

    def serialize(self) -> dict:
        """Serializes parametrization information.

        Serialize returns a dictionary of all the information necessary to
        recover the parametrization (via deserialize). This includes
        the current parametrization vector as well as any other parametrization
        metadata (e.g. etch fraction).
        """
        return {"vector": self.to_vector()}

    def deserialize(self, data):
        """Deserializes parametrization dictionary.

        Deserializes parametrization information that was serialized using
        serialize().
        """
        self.from_vector(data["vector"])


class DirectParam(Parametrization):
    """ Represents a direct parametrization.

    A direct parametrization holds the z-value of each pixel.
    Projection is defined to keep the z-value between 0 and 1.
    """
    def __init__(
            self, initial_value: np.ndarray,
            bounds: List[float] = (0, 1)) -> None:
        self.vector = np.array(initial_value).astype(float)
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        # Expand upper and lower bounds into arrays.
        if isinstance(self.lower_bound, numbers.Number):
            self.lower_bound = (self.lower_bound, ) * len(self.vector)
        if isinstance(self.upper_bound, numbers.Number):
            self.upper_bound = (self.upper_bound, ) * len(self.vector)
        if self.lower_bound is None:
            self.lower_bound = (None, ) * len(self.vector)
        if self.upper_bound is None:
            self.upper_bound = (None, ) * len(self.vector)

    def get_structure(self) -> np.ndarray:
        return self.vector

    def project(self) -> None:
        # np.clip does not except None as valid bound.
        # Therefore, we change Nones to +/- inf.
        lower_bound = [
            b if b is not None else -np.inf for b in self.lower_bound
        ]
        upper_bound = [
            b if b is not None else np.inf for b in self.upper_bound
        ]
        self.vector = np.clip(self.vector, lower_bound, upper_bound)

    def get_bounds(self):
        return (self.lower_bound, self.upper_bound)

    def calculate_gradient(self) -> None:
        return sparse.eye(len(self.vector))

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        self.vector = vector


class CubicParam(Parametrization):
    """ Parametrization that interpolates a coarse grid to a finer grid that is
        used as z.

    initial_value = initial parametrization (on coarse grid) taking into account the
                        the symmetry and periodicity
    coarse_x, coarse_y = coarse grid
    fine_x, fine_y = fine grid
    symmetry = 2 element array that indicates with 0 or 1 if there is symmetry in the x
                    and or y direction (this is imposed on the coarse grid)
    periodicity = 2 element array that indicates with 0 or 1 if the boundaries in the x
                    and/or y direction are the same
    periods = 2 element array that indicates how many periods there are in the x and/or y
                    direction
    lower_bound, upper_bound = lower and upper bound of the parametrization
    bounds = lower and upper bound of the parametrization
    """
    def __init__(self,
                 initial_value: np.ndarray,
                 coarse_x: np.ndarray,
                 coarse_y: np.ndarray,
                 fine_x: np.ndarray,
                 fine_y: np.ndarray,
                 symmetry: np.ndarray = np.array([0, 0]),
                 periodicity: np.ndarray = np.array([0, 0]),
                 periods: np.ndarray = np.array([0, 0]),
                 lower_bound: Union[float, List[float]] = 0,
                 upper_bound: Union[float, List[float]] = 1,
                 bounds: List[float] = None) -> None:
        self.x_z = fine_x
        self.y_z = fine_y
        self.x_p = coarse_x
        self.y_p = coarse_y
        self.beta = 1 / 3  # relaxation factor of the fabrication constraint
        self.k = 4  # factor in the exponential in the sigmoid function used to discretize

        self.geometry_matrix, self.reverse_geometry_matrix = cubic_utils.make_geometry_matrix_cubic(
            (len(coarse_x), len(coarse_y)), symmetry, periodicity, periods)

        # correct the initial value
        if isinstance(initial_value, (float, int, complex)):
            self.vector = initial_value * np.ones(
                self.geometry_matrix.shape[1])
        elif len(initial_value) == self.reverse_geometry_matrix.shape[0]:
            self.vector = initial_value
        elif len(initial_value) == self.reverse_geometry_matrix.shape[1]:
            self.vector = self.reverse_geometry_matrix @ initial_value
        else:
            raise ValueError('Invalid initial value')

        # Make the interpolation matrix.
        #periodicity_phi2f = np.logical_and(periodicity, np.logical_not(periods))
        phi2f, _, _ = cubic_utils.CubicMatrices(self.x_z, self.y_z, self.x_p,
                                                self.y_p, periodicity)
        self.vec2f = phi2f @ self.geometry_matrix

        # Set bounds
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def set_k(self, k):
        ''' set the slope of the sigmoid function. '''
        self.k = k

    def get_structure(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        if self.k:
            return 1 / (1 + np.exp(-self.k * (2 * z_cubic - 1)))
        else:
            return z_cubic

    def calculate_gradient(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        if self.k:
            return sparse.diags(
                2 * self.k * np.exp(-self.k * (2 * z_cubic - 1)) /
                (1 + np.exp(-self.k * (2 * z_cubic - 1)))**2) @ self.vec2f
        else:
            return self.vec2f

    def get_bounds(self):
        vec_len = len(self.vector)
        return ((self.lower_bound, ) * vec_len, (self.upper_bound, ) * vec_len)

    def project(self) -> None:
        self.vector = np.clip(self.vector, self.lower_bound, self.upper_bound)

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        self.vector = vector

    def serialize(self) -> Dict:
        return {
            "vector": self.to_vector(),
            "sigmoid_strength": self.k,
        }

    def deserialize(self, state: Dict) -> None:
        self.from_vector(state["vector"])
        self.k = state["sigmoid_strength"]

    # functions to fit the parametrization
    def fit2eps(self, eps_bg, S, eps):
        from spins.invdes.problem import Fit2Eps, OptimizationProblem
        import spins.invdes.optimization as optim

        # make objective
        obj = Fit2Eps(eps_bg, S, eps)
        obj = OptimizationProblem(obj)

        # optimize continuous
        opt_cont = optim.ScipyOptimizer(method='L-BFGS-B',
                                        options={
                                            'maxiter': 200,
                                            'maxcor': 10
                                        })
        iter_num = 0

        def callback(_):
            nonlocal iter_num
            iter_num += 1

        opt_cont(obj, self, callback=callback)

    # Functions to generate gds
    def generate_polygons(self, dx: float):
        '''
            Generate a list of polygons

            input:
                dx: grid spacing
            output:
                list of the polygons
        '''
        x_z = self.x_z * dx / 2
        y_z = self.y_z * dx / 2
        design_area_fine = np.array([len(x_z), len(y_z)])
        phi = self.vec2f @ self.vector
        phi_mat = phi.reshape(design_area_fine, order='F')

        # Pad the design region with zeros to ensure outer boundary is drawn.
        phi_extended = np.zeros(design_area_fine + 2)
        phi_extended[1:-1, 1:-1] = phi_mat

        x_extended = np.r_[x_z[0] - dx / 2, x_z, x_z[-1] + dx / 2]
        y_extended = np.r_[y_z[0] - dx / 2, y_z, y_z[-1] + dx / 2]

        import matplotlib.pyplot as plt
        cs = plt.contour(x_extended, y_extended, phi_extended - 0.5, [0])
        paths = cs.collections[0].get_paths()

        return [p.to_polygons()[0] for p in paths]


class HermiteParam(Parametrization):
    """ Parametrization that interpolates coarse grid value and derivatives to a finer
        grid that is used as z. The parametrization is defined at the coarse grid by
        f, df/dx, df/dy and d^2f/dxdy. The parametrization vector is thus 4 times len(
        coarse_x)*len(coarse_y)

    initial_value = initial parametrization (on coarse grid) taking into account the
                        the symmetry and periodicity
    coarse_x, coarse_y = coarse grid
    fine_x, fine_y = fine grid
    symmetry = 2 element array that indicates with 0 or 1 if there is symmetry in the x
                    and or y direction (this is imposed on the coarse grid)
    periodicity = 2 element array that indicates with 0 or 1 if the boundaries in the x
                    and/or y direction are the same
    periods = 2 element array that indicates how many periods there are in the x and/or y
                    direction
    lower_bound, upper_bound = lower and upper bound of the parametrization
    bounds = lower and upper bound of the parametrization
    scale = scaling factor for the derivatives
    """
    def __init__(self,
                 initial_value: np.ndarray,
                 coarse_x: np.ndarray,
                 coarse_y: np.ndarray,
                 fine_x: np.ndarray,
                 fine_y: np.ndarray,
                 symmetry: np.ndarray = np.array([0, 0]),
                 periodicity: np.ndarray = np.array([0, 0]),
                 periods: np.ndarray = np.array([0, 0]),
                 lower_bound: Union[float, List[float]] = -np.inf,
                 upper_bound: Union[float, List[float]] = np.inf,
                 bounds: List[float] = None,
                 scale: float = 1.75) -> None:
        self.x_z = fine_x
        self.y_z = fine_y
        self.x_p = coarse_x
        self.y_p = coarse_y
        self.beta = 1 / 3  # relaxation factor of the fabrication constraint
        self.k = 4  # factor in the exponential in the sigmoid function used to discretize
        self.scale_deriv = scale

        self.fine_x_grid, self.fine_y_grid = np.meshgrid(fine_x,
                                                         fine_y,
                                                         indexing='ij')

        self.geometry_matrix, self.reverse_geometry_matrix = cubic_utils.make_geometry_matrix_hermite(
            (len(coarse_x), len(coarse_y)), symmetry, periodicity, periods)

        # correct the initial value
        self.derivative_matrix = cubic_utils.idxdydxy_matrix(
            coarse_x,
            coarse_y,
            deriv_scaling=np.array([
                1, scale * np.diff(fine_x).mean(),
                scale**2 * np.diff(fine_x).mean()**2
            ]))

        # correct the initial value
        if isinstance(initial_value, (float, int, complex)):
            self.vector = initial_value * np.ones(
                self.geometry_matrix.shape[1])
        elif len(initial_value) == self.geometry_matrix.shape[1]:
            self.vector = initial_value
        elif len(initial_value) == self.geometry_matrix.shape[0]:
            self.vector = self.reverse_geometry_matrix @ initial_value
        elif len(initial_value) == self.derivative_matrix.shape[1]:
            self.vector = self.reverse_geometry_matrix @ \
                self.derivative_matrix @ initial_value
        #TODO vcruysse: account for the following cases
        #elif len(initial_value) == symmetry_matrix.shape[1]*4:
        #elif len(initial_value) == symmetry_matrix.shape[1]:
        #elif len(initial_value) == periodic_matrix_n.shape[0]*4:
        #elif len(initial_value) == periodic_matrix_n.shape[0]:
        else:
            raise ValueError('Invalid initial value')

        # Make the interpolation matrix.
        phi2f, _, _ = cubic_utils.CubicMatrices(
            fine_x,
            fine_y,
            coarse_x,
            coarse_y,
            periodicity,
            derivatives=True,
            deriv_scaling=np.array([
                1, scale * np.diff(fine_x).mean(),
                scale**2 * np.diff(fine_x).mean()**2
            ]))
        self.vec2f = phi2f @ self.geometry_matrix

        # Set bounds
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def get_structure(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        return 1 / (1 + np.exp(-self.k * (2 * z_cubic - 1)))

    def calculate_gradient(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        return sparse.diags(2 * self.k * np.exp(-self.k * (2 * z_cubic - 1)) /
                            (1 + np.exp(-self.k *
                                        (2 * z_cubic - 1)))**2) @ self.vec2f

    def set_k(self, k):
        self.k = k

    def project(self) -> None:
        self.vector = np.clip(self.vector, self.lower_bound, self.upper_bound)

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        if isinstance(vector, (float, int, complex)):
            self.vector = vector * np.ones(self.geometry_matrix.shape[1])
        elif len(vector) == self.geometry_matrix.shape[1]:
            self.vector = vector
        elif len(vector) == self.geometry_matrix.shape[0]:
            self.vector = self.reverse_geometry_matrix @ vector
        #TODO vcruysse: account for the following cases
        #elif len(initial_value) == symmetry_matrix.shape[1]*4:
        #elif len(initial_value) == symmetry_matrix.shape[1]:
        #elif len(initial_value) == periodic_matrix_n.shape[0]*4:
        #elif len(initial_value) == periodic_matrix_n.shape[0]:
        else:
            raise ValueError('Invalid initial value')

    def serialize(self) -> Dict:
        return {
            "vector": self.to_vector(),
            "sigmoid_strength": self.k,
        }

    def deserialize(self, state: Dict) -> None:
        self.from_vector(state["vector"])
        self.k = state["sigmoid_strength"]

    # functions to fit the parametrization
    def fit2eps(self, eps_bg, S, eps):
        from spins.invdes.problem import Fit2Eps, OptimizationProblem
        import spins.invdes.optimization as optim

        # make objective
        obj = Fit2Eps(eps_bg, S, eps)
        obj = OptimizationProblem(obj)

        # optimize continuous
        opt_cont = optim.ScipyOptimizer(method='L-BFGS-B',
                                        options={
                                            'maxiter': 200,
                                            'maxcor': 10
                                        })
        iter_num = 0

        def callback(v):
            nonlocal iter_num
            iter_num += 1
            print('fit2eps-continous: ' + str(iter_num))

        opt_cont(obj, self, callback=callback)
