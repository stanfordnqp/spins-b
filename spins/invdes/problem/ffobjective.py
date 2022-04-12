import abc
import enum
import numpy as np
import scipy.sparse
import scipy.io
import copy
from typing import Callable, List

import spins.invdes.parametrization as invparam
import spins.fdfd_tools as fdfd_tools
from spins.gridlock import Direction

from spins.invdes.problem.objective import OptimizationFunction
from spins.invdes.parametrization import Parametrization
from spins.invdes.problem.simulation import AdjointFdfdSimulation
from spins.invdes.problem.simulation import FdfdSimulation
from spins.invdes.problem.emobjective import EmObjective
from spins.invdes.problem.objective import IndicatorMin

from spins.invdes.problem import farfield


class DirectedPower(EmObjective):

    def __init__(self, sim, grid, FF_cond, E_background=None):
        """ Build an objective that evaluates the power projected in
            far field.

            sim: simulation
            grid: grid used to make simulation
            FF_cond: dictionary that contains th_bound and ph_bound. Both of
            these are 2 element list that define the minimum and maximum angles
            over which you intregrate the power. (th is angle with the z-axis,
            ph is the angle in the xy-plane)

        """
        super().__init__(sim)

        self.grid = grid
        self.E_background = E_background

        self._compute_objective(FF_cond)

    def _compute_objective(self, FF_cond):
        # make the far field sphere or half sphere
        if np.all(np.array(FF_cond['th_bounds']) < np.pi / 2):
            self.points, self.triangles = farfield.make_half_sphere_point(4, 1)
        elif np.all(np.array(FF_cond['th_bounds']) > np.pi / 2):
            self.points, self.triangles = farfield.make_half_sphere_point(4, -1)
        else:
            self.points, self.triangles = farfield.make_sphere_point(4)

        # make the near-to-far-field matrix
        if 'box_center' in FF_cond:
            FF_arg = {
                'points': self.points,
                'omegas': self.sim.omega,
                'grid': self.grid,
                'dxes': self.sim.dxes,
                'box_center': FF_cond['box_center'],
                'box_size': FF_cond['box_size'],
                'eps_0': 1
            }
            self.FF_projection_matrix = farfield.make_near2farfield_box_matrix(
                **FF_arg)
        elif 'pos' in FF_cond:
            FF_arg = {
                'points': self.points,
                'omegas': self.sim.omega,
                'grid': self.grid,
                'dxes': self.sim.dxes,
                'pos': FF_cond['pos'],
                'width': FF_cond['width'],
                'polarity': 1,
                'eps_0': 1
            }
            self.FF_projection_matrix = farfield.make_near2farfield_matrix(
                **FF_arg)

        # make the power integration vector
        n_points = self.points.shape[0]
        sum_matrix = scipy.sparse.hstack([
            scipy.sparse.csr_matrix((n_points, n_points)),
            scipy.sparse.eye(n_points),
            scipy.sparse.eye(n_points)
        ])
        points2triangles = farfield.farfield.points2triangles_averaging_matrix(
            self.points, self.triangles)
        area_vector = area_selection_vector(self.points, self.triangles,
                                            FF_cond['th_bounds'],
                                            FF_cond['ph_bounds'])
        self.directed_power_vector = area_vector @ points2triangles @ sum_matrix

        # initialize direcitivity and FarField
        self.FarField = np.ones(3 * n_points)

    def calculate_partial_df_dx(self, x, z):
        if self.E_background is not None:
            E = x - E_background
        else:
            E = x

        self.FarField = self.FF_projection_matrix @ E
        ddirected_power_viol_dx = 0.5 * self.directed_power_vector @ \
                                  scipy.sparse.diags(np.conj(self.FarField)) @ \
                                  self.FF_projection_matrix

        return ddirected_power_viol_dx

    def calculate_f(self, x, z):
        if self.E_background is not None:
            E = x - E_background
        else:
            E = x

        self.FarField = self.FF_projection_matrix @ x
        directed_power = 0.5 * self.directed_power_vector @ np.abs(
            self.FarField)**2

        return directed_power

    def get_electric_fields(self, param: Parametrization):
        # TODO(logansu): Refactor please
        return fdfd_tools.unvec(
            self.sim.simulate(param.get_structure()), self.sim.get_dims())


class BasicScatter(EmObjective):

    def __init__(self, sim, grid, FF_cond, E_background=None):
        """ Build an objective that evaluated the directivity.
        """
        super().__init__(sim)

        self.grid = grid
        self.pf = 2
        self.E_background = E_background

        self._compute_objective(FF_cond)

    def _compute_objective(self, FF_cond):
        # make the far field sphere
        self.points, self.triangles = farfield.make_sphere_point(4)

        # make the near-to-far-field matrix
        if 'box_center' in FF_cond:
            FF_arg = {
                'points': self.points,
                'omegas': self.sim.omega,
                'grid': self.grid,
                'dxes': self.sim.dxes,
                'box_center': FF_cond['box_center'],
                'box_size': FF_cond['box_size'],
                'eps_0': 1
            }
            self.FF_projection_matrix = farfield.make_near2farfield_box_matrix(
                **FF_arg)
        elif 'pos' in FF_cond:
            FF_arg = {
                'points': self.points,
                'omegas': self.sim.omega,
                'grid': self.grid,
                'dxes': self.sim.dxes,
                'pos': FF_cond['pos'],
                'width': FF_cond['width'],
                'polarity': 1,
                'eps_0': 1
            }
            self.FF_projection_matrix = farfield.make_near2farfield_matrix(
                **FF_arg)

        # make the power integration vector
        n_p = self.points.shape[0]
        sum_matrix = scipy.sparse.hstack([
            scipy.sparse.csr_matrix((n_p, n_p)),
            scipy.sparse.eye(n_p),
            scipy.sparse.eye(n_p)
        ])
        points2triangles = farfield.points2triangles_averaging_matrix(
            self.points, self.triangles)
        scatter_area_vector = farfield.area_selection_vector(
            self.points, self.triangles, [0, np.pi], [-np.pi, np.pi])
        directed_area_vector = farfield.area_selection_vector(
            self.points, self.triangles, FF_cond['th_bounds'],
            FF_cond['ph_bounds'])
        self.scattered_power_vector = scatter_area_vector @ points2triangles @ sum_matrix
        self.directed_power_vector = directed_area_vector @ points2triangles @ sum_matrix

        # set alpha
        self.objective_alpha = 1  #FF_cond['directivity']

        # initialize direcitivity and FarField
        self.directivity = 1
        self.FarField = np.ones(3 * n_p)

    def calculate_partial_df_dx(self, x, z):
        if self.E_background is not None:
            E = x - E_background
        else:
            E = x

        #calculate far field
        FF_mat = self.FF_projection_matrix
        FarField = FF_mat @ E
        dFarField_square_viol_dx = scipy.sparse.diags(np.conj(FarField)) @ \
                                    FF_mat
        #calculate total and directed scattered power
        n_points = self.points.size
        sum_matrix = scipy.sparse.hstack([
            scipy.sparse.csr_matrix((n_points, n_points)),
            scipy.sparse.eye(n_points),
            scipy.sparse.eye(n_points)
        ])
        scattered_power = self.scattered_power_vector @ np.abs(FarField)**2
        directed_power = self.directed_power_vector @ np.abs(FarField)**2
        dscattered_power_viol_dx = self.scattered_power_vector @ \
                                    dFarField_square_viol_dx
        ddirected_power_viol_dx = self.directed_power_vector @ \
                                    dFarField_square_viol_dx
        #calculate directivity
        directivity = directed_power / scattered_power
        ddirectivity_viol_dx = ( ddirected_power_viol_dx*scattered_power - \
                                 directed_power*dscattered_power_viol_dx)/  \
                                 scattered_power**2

        #objection function
        alpha = self.objective_alpha
        f0 = (directivity < alpha) * (alpha - directivity)
        df0_viol_dx = (directivity < alpha) * (-1) * ddirectivity_viol_dx
        df_viol_dx = self.pf * f0**(self.pf - 1) * df0_viol_dx

        return df_viol_dx

    def calculate_f(self, x, z):
        #Test for background
        if self.E_background is not None:
            E = x - E_background
        else:
            E = x
        #calculate far field
        self.FarField = self.FF_projection_matrix @ E

        #calculate total and directed scattered power
        scattered_power = self.scattered_power_vector @ np.abs(self.FarField)**2
        directed_power = self.directed_power_vector @ np.abs(self.FarField)**2

        #calculate directivity
        self.directivity = directed_power / scattered_power

        #objection function
        alpha = self.objective_alpha
        f0 = (self.directivity < alpha) * (alpha - self.directivity)
        f = np.sum(f0**self.pf)

        return f

    def get_electric_fields(self, param: Parametrization):
        # TODO(logansu): Refactor please
        return fdfd_tools.unvec(
            self.sim.simulate(param.get_structure()), self.sim.get_dims())
