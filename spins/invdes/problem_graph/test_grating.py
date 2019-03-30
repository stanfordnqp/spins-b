import numpy as np
import pytest

from spins.invdes import parametrization
from spins.invdes.problem_graph import grating


@pytest.mark.parametrize("cont,disc,min_feature", [
    ([0, 0, 0.4, 1, 1, 0.8, 0, 0], [2.6, 5.8], 2),
    ([0, 0, 0.4, 0.8, 0.8, 0.8, 0, 0], [2.6, 5.8], 2),
    ([0, 0, 0.4, 0.2, 1, 0.8, 0, 0], [3.8, 5.8], 2),
    ([1, 1, 0.6, 0, 0, 0.2, 1, 1], [0, 2.6, 5.8, 8], 2),
])
def test_grating_edge_discretization(cont, disc, min_feature):
    param = parametrization.DirectParam(cont)
    trans = grating.GratingEdgeDiscretization(param, 40 * min_feature, 40)
    param_disc = parametrization.GratingParam([],
                                              num_pixels=len(param.to_vector()))

    trans(param_disc, None)

    np.testing.assert_almost_equal(param_disc.to_vector(), disc)


def test_grating_edge_discretization_min_feature_too_small_raises_error():
    param = parametrization.DirectParam([0, 1, 2, 3])
    with pytest.raises(ValueError, match="feature size must be larger"):
        trans = grating.GratingEdgeDiscretization(param, 30, 40)
