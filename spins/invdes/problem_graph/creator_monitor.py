from typing import List, Optional

import numpy as np

from spins import fdfd_tools
from spins import gridlock
from spins.invdes import parametrization
from spins.invdes import problem
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import simspace
from spins.invdes.problem_graph import workspace


class Monitor(problem.OptimizationFunction):
    """Defines a monitor.

    Monitors behave exactly as functions but do not implement differentiation.
    This enables using the graph evaluation engine to share computation
    when multiple monitors need to evaluated simultaneously.

    This is mainly used to rename `calculate_objective_function` to
    `get_data`, the latter of which is more appropriate for a monitor.
    """

    def get_data(self, param: parametrization.Parametrization):
        return self.calculate_objective_function(param)


class SimpleMonitor(Monitor):
    """Monitor that does not do any post-processing on function values.

    This monitor simply records output value of an optimization function.
    """

    def __init__(self, function: problem.OptimizationFunction) -> None:
        """Initialize the monitor.

        Args:
            function: Function to monitor.
        """
        super().__init__(function)
        self._function = function

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        return input_vals[0]


@optplan.register_node(optplan.SimpleMonitor)
def create_scalar_monitor(params: optplan.SimpleMonitor,
                          work: workspace.Workspace) -> SimpleMonitor:
    return SimpleMonitor(function=work.get_object(params.function))


class FieldMonitor(Monitor):
    """Monitor for functions with a 3D vector field output."""

    def __init__(self,
                 function: problem.OptimizationFunction,
                 sim_space: simspace.SimulationSpace,
                 slice_point: Optional[List[float]] = None,
                 slice_normal: Optional[List[int]] = None) -> None:
        """Initializes the monitor.

        If `slice_point` and `slice_normal` are set, then a 2D slice is taken
        over the 3D vector field.

        Args:
            function: Field function to monitor.
            sim
            slice_point: Point in the field slice to monitor.
            slice_normal: Normal of the slice that is evaluated.
        """
        super().__init__(function)
        self._function = function
        self._slices = None
        self._sim_space = sim_space
        if slice_point and slice_normal:
            slice_axis = gridlock.axisvec2axis(slice_normal)
            grid = gridlock.Grid(sim_space.edge_coords, num_grids=3)
            slice_ind = grid.pos2ind(slice_point, which_shifts=None).astype(int)
            slice_ind = slice_ind[slice_axis]

            self._slices = 3 * [slice(0, None)]
            self._slices[slice_axis] = slice(slice_ind, slice_ind + 1)

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the  3D vector field.

        Args:
            input_vals: Single-element list containing the field.

        Returns:
            List of all three field components.
        """
        # Get fields.
        fields = fdfd_tools.unvec(input_vals[0], self._sim_space.dims)

        # Make slices.
        if self._slices:
            fields = [field[tuple(self._slices)] for field in fields]

        return fields


@optplan.register_node(optplan.FieldMonitor)
def create_field_monitor(params: optplan.FieldMonitor,
                         work: workspace.Workspace) -> FieldMonitor:
    return FieldMonitor(
        function=work.get_object(params.function),
        sim_space=work.get_object(params.function.simulation_space),
        slice_point=params.center,
        slice_normal=params.normal)
