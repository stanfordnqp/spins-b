from typing import List, Tuple

import numpy as np

from spins import fdfd_tools
from spins import gridlock
from spins.invdes import problem
from spins.invdes.problem_graph import creator_em
from spins.invdes.problem_graph import grid_utils
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
from spins.invdes.problem_graph.simspace import SimulationSpace


class PowerTransmission(optplan.Function):
    """Defines a function that measures amount of power passing through plane.

    The amount of power is computed by summing the Poynting vector across
    the desired plane.

    Attributes:
        field: The simulation field to use.
        center: Center of plane to compute power.
        extents: Extents of the plane over which to compute power.
        normal: Normal direction of the plane. This determines the sign of the
            power flowing through the plane.
    """
    type = optplan.define_schema_type("function.poynting.plane_power")
    field = optplan.ReferenceType(optplan.FdfdSimulation)
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()


class PowerTransmissionFunction(problem.OptimizationFunction):
    """Evalutes power passing through a plane.

    The power is computed by summing over the Poynting vector on the plane.
    Specifically, the power is `sum(0.5 * real(E x H*))[axis]` where `axis`
    indicates the component of the Poynting vector to use.

    Currently the plane must be an axis-aligned plane and the permeability
    is assumed to be unity.

    Note that the calculation of Poynting vector does not take into account
    PMLs.
    """

    def __init__(
            self,
            field: creator_em.FdfdSimulation,
            simspace: SimulationSpace,
            wlen: float,
            plane_slice: Tuple[slice, slice, slice],
            axis: int,
            polarity: int,
    ) -> None:
        """Creates a new power plane function.

        Args:
            field: The `FdfdSimulation` to use to calculate power.
            simspace: Simulation space corresponding to the field.
            wlen: Wavelength of the field.
            plane_slice: Represents the locations of the field that are part
                of the plane.
            axis: Which Poynting vector field component to use.
            polarity: Indicates directionality of the plane.
        """
        super().__init__(field)
        self._omega = 2 * np.pi / wlen
        self._dxes = simspace.dxes
        self._plane_slice = plane_slice
        self._axis = axis
        self._polarity = polarity

        # Precompute any operations that can be computed without knowledge of
        # the field.
        self._op_e2h = fdfd_tools.operators.e2h(self._omega, self._dxes)
        self._op_curl_e = fdfd_tools.operators.curl_e(self._dxes)

        # Create a filter that sets a 1 in every position that is included in
        # the computation of the Poynting vector.
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[self._axis][tuple(self._plane_slice)] = 1
        self._filter_vec = fdfd_tools.vec(filter_grid)

    # TODO(logansu): Make it work for arbitrary mu.
    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        efield = input_val[0]

        hfield = self._op_e2h @ efield
        op_e_cross = fdfd_tools.operators.poynting_chew_e_cross(
            efield, self._dxes)
        poynting = 0.5 * np.real(op_e_cross @ np.conj(hfield))
        return self._polarity * np.sum(poynting * self._filter_vec)

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:

        efield = input_vals[0]

        hfield = self._op_e2h @ efield
        op_e_cross = fdfd_tools.operators.poynting_chew_e_cross(
            efield, self._dxes)
        op_h_cross = fdfd_tools.operators.poynting_chew_h_cross(
            hfield, self._dxes)

        # Compute the gradient across all of space.
        mu = 1
        grad_mat = -0.25 * (1 / (1j * self._omega * mu) * op_e_cross.conj() *
                            self._op_curl_e + op_h_cross.conj())
        return [grad_val * self._polarity * self._filter_vec.T @ grad_mat]


@optplan.register_node(PowerTransmission)
def create_power_transmission_function(
        params: PowerTransmission,
        context: workspace.Workspace) -> PowerTransmissionFunction:
    simspace = context.get_object(params.field.simulation_space)
    return PowerTransmissionFunction(
        field=context.get_object(params.field),
        simspace=simspace,
        wlen=params.field.wavelength,
        plane_slice=grid_utils.create_region_slices(
            simspace.edge_coords, params.center, params.extents),
        axis=gridlock.axisvec2axis(params.normal),
        polarity=gridlock.axisvec2polarity(params.normal))
