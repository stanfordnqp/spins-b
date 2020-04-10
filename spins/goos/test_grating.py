import numpy as np

from spins import goos
from spins.goos.grating import _get_general_edge_loc_dp

import pytest


def test_grating_one_teeth():
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Constant([0.2, 0.5])
        thickness = goos.Constant([0.8])
        grating = goos.grating.BarcodeGrating(edge_locs,
                                              thickness, [1, 2, 3], [2, 10, 1],
                                              goos.material.Material(index=2),
                                              grating_dir=0)
        grating_boxes = grating.get()

        assert len(grating_boxes) == 1
        np.testing.assert_almost_equal(grating_boxes[0].pos, [0.7, 2, 3])
        np.testing.assert_almost_equal(grating_boxes[0].extents, [0.6, 10, 0.8])


def test_grating_two_teeth():
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Constant([0.2, 0.5, 1])
        thickness = goos.Constant([0.8, 0.5])
        grating = goos.grating.BarcodeGrating(edge_locs,
                                              thickness, [1, 2, 3], [2, 10, 1],
                                              goos.material.Material(index=2),
                                              grating_dir=0)
        grating_boxes = grating.get()

        assert len(grating_boxes) == 2
        np.testing.assert_almost_equal(grating_boxes[0].pos, [0.7, 2, 3])
        np.testing.assert_almost_equal(grating_boxes[0].extents, [0.6, 10, 0.8])
        np.testing.assert_almost_equal(grating_boxes[1].pos, [1.5, 2, 3])
        np.testing.assert_almost_equal(grating_boxes[1].extents, [1, 10, 0.5])


@pytest.mark.parametrize("use_edge_locs,params", [(True, [0, 1, 1.5, 2]),
                                                  (False, [0, 1, 0.5, 0.5])])
def test_pixelated_multietch_grating_xz(use_edge_locs, params):
    # Grating defined in xz-plane.
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Variable(params)
        height_index = goos.Variable([2, 0, 1, 2])
        grating = goos.grating.PixelatedGrating(
            edge_locs,
            height_index=height_index,
            height_fracs=[0, 0.5, 1],
            pos=[0, 0, 0],
            extents=[2, 10, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=0,
            grating_dir_spacing=0.5,
            etch_dir_divs=2,
            use_edge_locs=use_edge_locs).get()

        np.testing.assert_equal(grating.pixel_size, [0.5, 10, 0.110])
        np.testing.assert_equal(grating.extents, [2, 10, 0.220])
        np.testing.assert_almost_equal(grating.array,
                                       [[[1, 1]], [[1, 1]], [[0, 0]], [[0, 1]]],
                                       decimal=4)


def test_pixelated_multietch_grating_xz_polarity():
    use_edge_locs = True
    params = [0, 1, 1.5, 2]
    # Grating defined in xz-plane.
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Variable(params)
        height_index = goos.Variable([2, 0, 1, 2])
        grating = goos.grating.PixelatedGrating(
            edge_locs,
            height_index=height_index,
            height_fracs=[0, 0.5, 1],
            pos=[0, 0, 0],
            extents=[2, 10, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=0,
            grating_dir_spacing=0.5,
            etch_dir_divs=2,
            etch_polarity=-1,
            use_edge_locs=use_edge_locs).get()

        np.testing.assert_equal(grating.pixel_size, [0.5, 10, 0.110])
        np.testing.assert_equal(grating.extents, [2, 10, 0.220])
        np.testing.assert_almost_equal(grating.array,
                                       [[[1, 1]], [[1, 1]], [[0, 0]], [[1, 0]]],
                                       decimal=4)


@pytest.mark.parametrize("use_edge_locs,params", [(True, [0, 1, 1.5, 2]),
                                                  (False, [0, 1, 0.5, 0.5])])
def test_pixelated_multietch_grating_yz(use_edge_locs, params):
    # Grating defined in yz-plane.
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Variable(params)
        height_index = goos.Variable([2, 0, 1, 2])
        grating = goos.grating.PixelatedGrating(
            edge_locs,
            height_index=height_index,
            height_fracs=[0, 0.5, 1],
            pos=[0, 0, 0],
            extents=[10, 2, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=1,
            grating_dir_spacing=0.5,
            etch_dir_divs=2,
            use_edge_locs=use_edge_locs).get()

        np.testing.assert_equal(grating.pixel_size, [10, 0.5, 0.110])
        np.testing.assert_equal(grating.extents, [10, 2, 0.220])
        np.testing.assert_almost_equal(grating.array,
                                       [[[1, 1], [1, 1], [0, 0], [0, 1]]],
                                       decimal=4)


def test_pixelated_multietch_grating_fit_edge_locs():
    # Grating defined in xz-plane.
    with goos.OptimizationPlan() as plan:
        height_index = goos.Variable([2, 0, 1, 2], parameter=True)

        edge_locs_target = goos.Variable([0, 1, 1.5, 2], parameter=True)
        grating_target = goos.grating.PixelatedGrating(
            edge_locs_target,
            height_index=height_index,
            height_fracs=[0, 0.5, 1],
            pos=[0, 0, 0],
            extents=[2, 10, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=0,
            grating_dir_spacing=0.5,
            etch_dir_divs=2,
            use_edge_locs=True)

        edge_locs = goos.Variable([0, 0.8, 1.6, 2])
        grating = goos.grating.PixelatedGrating(
            edge_locs,
            height_index=height_index,
            height_fracs=[0, 0.5, 1],
            pos=[0, 0, 0],
            extents=[2, 10, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=0,
            grating_dir_spacing=0.5,
            etch_dir_divs=2,
            use_edge_locs=True)

        from spins.goos_sim import maxwell
        region = goos.Box3d(center=[0, 0, 0], extents=[3, 11, 0.5])
        mesh = maxwell.UniformMesh(dx=0.1)
        eps_diff = goos.Norm(
            maxwell.RenderShape(
                grating, region=region, mesh=mesh, wavelength=1.55) -
            maxwell.RenderShape(
                grating_target, region=region, mesh=mesh, wavelength=1.55))

        goos.opt.scipy_minimize(eps_diff, "L-BFGS-B", max_iters=20)
        plan.run()

        np.testing.assert_almost_equal(grating.get().array,
                                       grating_target.get().array)


@pytest.mark.parametrize("use_edge_locs,params", [(True, [0, 1, 1.5, 2]),
                                                  (False, [0, 1, 0.5, 0.5])])
def test_pixelated_multietch_grating_xz_full_etch(use_edge_locs, params):
    # Grating defined in xz-plane.
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Variable(params)
        height_index = goos.Variable([0, 1, 0, 1])
        grating = goos.grating.PixelatedGrating(
            edge_locs,
            height_index=height_index,
            height_fracs=[0, 1],
            pos=[0, 0, 0],
            extents=[2, 10, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=0,
            grating_dir_spacing=0.5,
            etch_dir_divs=1,
            use_edge_locs=use_edge_locs).get()

        np.testing.assert_equal(grating.pixel_size, [0.5, 10, 0.220])
        np.testing.assert_equal(grating.extents, [2, 10, 0.220])
        np.testing.assert_almost_equal(grating.array,
                                       [[[0]], [[0]], [[1]], [[0]]],
                                       decimal=4)


@pytest.mark.parametrize(
    "use_edge_locs,params,pol",
    [(True, [0, 1, 1.5, 1.9], 1), (False, [0, 1, 0.5, 0.4], 1),
     (True, [0, 1, 1.5, 1.9], -1), (False, [0, 1, 0.5, 0.4], -1)],
)
def test_pixelated_multietch_grating_xz_gradient(use_edge_locs, params, pol):
    # Brute force calculate the gradient.
    # Grating defined in xz-plane.
    with goos.OptimizationPlan() as plan:
        edge_locs = goos.Variable(params)
        height_index = goos.Variable([0, 1, 0, 1])
        grating = goos.grating.PixelatedGrating(
            edge_locs,
            height_index=height_index,
            height_fracs=[0.4, 1],
            pos=[0, 0, 0],
            extents=[2, 10, 0.220],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.5),
            grating_dir=0,
            grating_dir_spacing=0.5,
            etch_dir_divs=4,
            etch_polarity=pol,
            use_edge_locs=use_edge_locs)

        from spins.goos_sim import maxwell
        shape = maxwell.RenderShape(grating,
                                    region=goos.Box3d(center=[0, 0, 0],
                                                      extents=[4, 10, 0.220]),
                                    mesh=maxwell.UniformMesh(dx=0.25),
                                    wavelength=1550)

        np.random.seed(247)
        obj = goos.dot(shape, np.random.random(shape.get().array.shape))

        val_orig = edge_locs.get().array
        step = 1e-5
        grad_num = np.zeros_like(val_orig)
        for i in range(len(val_orig)):
            delta = np.zeros_like(val_orig)
            delta[i] = 1

            edge_locs.set(val_orig + step * delta)
            f_plus = obj.get(run=True).array

            edge_locs.set(val_orig - step * delta)
            f_minus = obj.get(run=True).array

            grad_num[i] = (f_plus - f_minus) / (2 * step)

        grad_backprop = obj.get_grad([edge_locs])[0].array_grad
        np.testing.assert_allclose(grad_backprop, grad_num)


def test_discretization_multietch():
    vec = [1, 0.8]
    depths = [0, 0.2, 0.8, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 start_depth_ind=3,
                                                 end_depth_ind=3)
    np.testing.assert_almost_equal(edge_locs, [1, 2])
    np.testing.assert_almost_equal(levels, [2, 3])

    vec = [1, 0.8, 0.8, 1]
    depths = [0, 0.2, 0.8, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 start_depth_ind=3,
                                                 end_depth_ind=3)
    np.testing.assert_almost_equal(edge_locs, [1, 3])
    np.testing.assert_almost_equal(levels, [2, 3])

    vec = [1, 1, 0.19, 0.2, 1]
    depths = [0, 0.2, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 start_depth_ind=2,
                                                 end_depth_ind=2)
    np.testing.assert_almost_equal(edge_locs, [2, 4])
    np.testing.assert_almost_equal(levels, [1, 2])

    vec = [1, 0, 0.3, 0.3, 1]
    depths = [0, 0.3, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 start_depth_ind=2,
                                                 end_depth_ind=2)
    np.testing.assert_almost_equal(edge_locs, [1, 2, 4])
    np.testing.assert_almost_equal(levels, [0, 1, 2])

    vec = [1, 1, 0.3, 1, 0, 0, 0.3, 0.3, 1]
    depths = [0, 0.3, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 start_depth_ind=2,
                                                 end_depth_ind=2)
    np.testing.assert_almost_equal(edge_locs, [2, 3, 4, 6, 8])
    np.testing.assert_almost_equal(levels, [1, 2, 0, 1, 2])


def test_discretization_max_feature_size():
    vec = [1, 1, 1, 0]
    depths = [0, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 max_features=2,
                                                 divisions=1,
                                                 start_depth_ind=1,
                                                 end_depth_ind=0)
    np.testing.assert_almost_equal(edge_locs, [2])
    np.testing.assert_almost_equal(levels, [0])

    vec = [1, 1, 1, 0, 0.2, 0.8, 1]
    depths = [0, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 max_features=[2, 3],
                                                 divisions=1,
                                                 start_depth_ind=1,
                                                 end_depth_ind=1)
    np.testing.assert_almost_equal(edge_locs, [3, 5])
    np.testing.assert_almost_equal(levels, [0, 1])

    vec = [1, 1, 0, 0.2, 0.8, 1]
    depths = [0, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=[2, 1],
                                                 max_features=[2, 2],
                                                 divisions=5,
                                                 start_depth_ind=1,
                                                 end_depth_ind=1)
    np.testing.assert_almost_equal(edge_locs, [2, 4])
    np.testing.assert_almost_equal(levels, [0, 1])

    vec = [1, 1, 0, 0, 0.2, 1, 1, 1, 0.4, 0, 0, 1]
    depths = [0, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=[2, 1],
                                                 max_features=[2, 10],
                                                 divisions=1,
                                                 start_depth_ind=1,
                                                 end_depth_ind=1)
    np.testing.assert_almost_equal(edge_locs, [2, 4, 9, 11])
    np.testing.assert_almost_equal(levels, [0, 1, 0, 1])


def test_discretization_regression():
    vec = [0.5, 1, 0.5]
    depths = [0, 0.5, 1]
    edge_locs, levels = _get_general_edge_loc_dp(vec,
                                                 depths,
                                                 min_features=1,
                                                 divisions=1,
                                                 start_depth_ind=1,
                                                 end_depth_ind=1)
    np.testing.assert_almost_equal(edge_locs, [1, 2])
    np.testing.assert_almost_equal(levels, [2, 1])


@pytest.mark.parametrize("use_edge_locs,grating_params", [(True, [110, 300]),
                                                          (False, [110, 190])])
def test_discretization(use_edge_locs, grating_params):
    with goos.OptimizationPlan() as plan:
        cont_var = goos.Variable([1, 1, 0.2, 0, 0.8, 0, 1, 1])
        edge_locs, height_ind, design = goos.grating.discretize_to_pixelated_grating(
            cont_var, [0, 1],
            50,
            1,
            1,
            min_features=100,
            use_edge_locs=use_edge_locs,
            pos=[0, 0, 0],
            extents=[400, 100, 100],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=2),
            grating_dir=0,
            grating_dir_spacing=20)

        plan.run()

        np.testing.assert_allclose(edge_locs.get().array, grating_params)
        np.testing.assert_allclose(height_ind.get().array, [0, 1])
