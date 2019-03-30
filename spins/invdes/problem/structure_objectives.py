""" Module for defining optimization functions related to the structure.
"""

from typing import List

import numpy as np
import scipy.sparse as sparse

from spins.invdes.parametrization import Parametrization
from spins.invdes.problem.objective import OptimizationFunction


class Fit2Eps(OptimizationFunction):
    """
    Fit a parametrization to an epsilon distribution
        the objective is to minimize: |eps - (eps_bg + S*z)|**2
    """

    def __init__(self, eps_bg: np.ndarray, S: sparse.spmatrix, eps: np.ndarray):
        self.eps_bg = eps_bg  # base epsilon
        self.eps = eps  # target epsilon
        self.S = S  #selection matrix

    def calculate_objective_function(self, param: Parametrization) -> float:
        z = param.get_structure()
        return np.sum(np.abs(self.eps_bg + self.S @ z - self.eps)**2)

    def calculate_gradient(self, param: Parametrization) -> np.ndarray:
        z = param.get_structure()
        err = self.eps_bg + self.S @ z - self.eps
        return np.asfortranarray(
            np.real(np.conj(err) @ self.S @ param.calculate_gradient()))

    def __str__(self) -> str:
        return 'fit2eps()'


class FabricationConstraint(OptimizationFunction):
    """
    Fabrication constraint objective
    This optimization function evaluates the fabrication penalty of the parametrization
    for a certain fabrication size limit.
    """

    def __init__(self, fcon, fabcon_method: int = 2):
        '''
        Arg:
            fcon: the fabrication constraint, i.e. the smallest gap size
            fabcon_method:
                0: only applies the gap constraint,
                1: applies the gap and curvature constraint by evaluating the curvature
                    constraint on the border (only available with BicubicLevelSet)
                2: applies the gap and curvature constraint (curvature is evaluated
                    everywhere) (only available with HermiteLevelSet)
        '''
        self.d = fcon
        self.method = fabcon_method
        self.weight_factor = 1

    def set_weight_factor(self, weight):
        self.weight_factor = np.squeeze(weight)

    def calculate_objective_function(self, param):
        if self.method == 1:
            return self.weight_factor * param.calculate_gap_penalty(
                np.pi / (1.20 * self.d))
        if self.method == 2:
            curv = self.weight_factor * param.calculate_curv_penalty(
                np.pi / (1.15 * self.d))
            gap = self.weight_factor * param.calculate_gap_penalty(
                np.pi / (1.20 * self.d))
            return curv + gap
        else:
            return self.weight_factor * param.calculate_gap_penalty(
                np.pi / (1.20 * self.d))

    def calculate_gradient(self, param):
        if self.method == 1:
            return self.weight_factor * param.calculate_gap_penalty_gradient(
                np.pi / (1.20 * self.d))
        if self.method == 2:
            curv = self.weight_factor * param.calculate_curv_penalty_gradient(
                np.pi / (1.15 * self.d))
            gap = self.weight_factor * param.calculate_gap_penalty_gradient(
                np.pi / (1.20 * self.d))
            return curv + gap
        else:
            return self.weight_factor * param.calculate_gap_penalty_gradient(
                np.pi / (1.20 * self.d))

    def __str__(self):
        return 'FabCon(' + str(self.d) + ')'
