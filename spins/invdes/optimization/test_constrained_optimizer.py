import logging
import pytest

import numpy as np

from spins.invdes.optimization import (AugmentedLagrangianOptimizer,
                                       PenaltyOptimizer)
import spins.invdes.optimization.problems as problems
from spins.invdes.parametrization import DirectParam, Parametrization

logging.getLogger('spins.invdes.optimization.constrained_optimizer').setLevel(
    logging.WARNING)


@pytest.mark.parametrize("opt,param,ans",
                         problems.build_constrained_problem_list())
def test_penalty_optimization(opt, param, ans):
    optimizer = PenaltyOptimizer()

    out_param = optimizer(opt, param)
    np.testing.assert_array_almost_equal(out_param.to_vector(), ans, decimal=2)


@pytest.mark.skip(reason="Test is unstable")
@pytest.mark.parametrize("opt,param,ans",
                         problems.build_constrained_problem_list())
def test_lagrangian_optimization(opt, param, ans):
    optimizer = AugmentedLagrangianOptimizer()
    out_param = optimizer(opt, param, callback=lambda x: print(x.to_vector()))
    np.testing.assert_array_almost_equal(out_param.to_vector(), ans, decimal=4)


if __name__ == '__main__':
    #optimizer = AugmentedLagrangianOptimizer()
    optimizer = PenaltyOptimizer()

    import time
    start = time.time()
    # prob1
    opt, param, ans = problems.build_constrained_ellipsoidal_problem()
    out_param = optimizer(opt, param)
    print(str(ans) + '    ' + str(out_param.get_structure()))
    # prob2
    opt, param, ans = problems.build_constrained_linear_problem(0)
    out_param = optimizer(opt, param)
    print(str(ans) + '    ' + str(out_param.get_structure()))
    # prob3
    opt, param, ans = problems.build_constrained_linear_problem(1)
    out_param = optimizer(opt, param)
    print(str(ans) + '    ' + str(out_param.get_structure()))
    # prob4
    opt, param, ans = problems.build_constrained_quadratic_problem(0)
    out_param = optimizer(opt, param)
    print(str(ans) + '    ' + str(out_param.get_structure()))
    # prob5
    opt, param, ans = problems.build_constrained_quadratic_problem(1)
    out_param = optimizer(opt, param)
    print(str(ans) + '    ' + str(out_param.get_structure()))
    # prob6
    opt, param, ans = problems.build_constrained_quadratic_problem(2)
    out_param = optimizer(opt, param)
    print(str(ans) + '    ' + str(out_param.get_structure()))
    print(time.time() - start)
