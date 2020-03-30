import numpy as np

from spins import goos
from spins.goos_sim import maxwell


def test_render_prism():
    with goos.OptimizationPlan() as plan:
        rect = goos.Cuboid(extents=goos.Constant([80, 80, 100]),
                           pos=goos.Constant([0, 0, 0]),
                           material=goos.material.Material(index=3.45))
        render = maxwell.RenderShape(
            rect,
            region=goos.Box3d(center=[0, 0, 0], extents=[160, 200, 40]),
            mesh=maxwell.UniformMesh(dx=40),
            background=goos.material.Material(index=1.0),
            wavelength=1550)

        np.testing.assert_array_almost_equal(
            render.get().array,
            [[[[1.], [1.], [1.], [1.], [1.]],
              [[1.], [1.], [11.9025], [11.9025], [1.]],
              [[1.], [1.], [11.9025], [11.9025], [1.]],
              [[1.], [1.], [1.], [1.], [1.]]],
             [[[1.], [1.], [1.], [1.], [1.]],
              [[1.], [3.725625], [6.45125], [3.725625], [1.]],
              [[1.], [6.45125], [11.9025], [6.45125], [1.]],
              [[1.], [3.725625], [6.45125], [3.725625], [1.]]],
             [[[1.], [1.], [1.], [1.], [1.]],
              [[1.], [1.], [6.45125], [6.45125], [1.]],
              [[1.], [1.], [11.9025], [11.9025], [1.]],
              [[1.], [1.], [6.45125], [6.45125], [1.]]]])


def test_render_cylinder():
    with goos.OptimizationPlan() as plan:
        cyl = goos.Cylinder(pos=goos.Constant([0, 0, 0]),
                            radius=goos.Constant(60),
                            height=goos.Constant(60),
                            material=goos.material.Material(index=3.45))
        render = maxwell.RenderShape(
            cyl,
            region=goos.Box3d(center=[0, 0, 0], extents=[200, 200, 40]),
            mesh=maxwell.UniformMesh(dx=40),
            background=goos.material.Material(index=1.0),
            wavelength=1550)

        np.testing.assert_array_almost_equal(
            render.get().array,
            [[[[1.], [1.], [1.], [1.], [1.]],
              [[1.], [2.20393657], [8.15133225], [8.14708434], [2.20801505]],
              [[1.], [4.81696873], [9.176875], [9.176875], [4.81254501]],
              [[1.], [2.20393657], [8.15133225], [8.14708434], [2.20801505]],
              [[1.], [1.], [1.], [1.], [1.]]],
             [[[1.], [1.], [1.], [1.], [1.]],
              [[1.], [2.20452237], [4.81426614], [2.20733312], [1.]],
              [[1.], [8.1503488], [9.176875], [8.14865466], [1.]],
              [[1.], [8.1503488], [9.176875], [8.14865466], [1.]],
              [[1.], [2.20452237], [4.81426614], [2.20733312], [1.]]],
             [[[1.], [1.], [1.], [1.], [1.]],
              [[1.], [1.06734618], [5.08385966], [5.07819579], [1.07209387]],
              [[1.], [5.0825484], [11.9025], [11.9025], [5.08028954]],
              [[1.], [5.0825484], [11.9025], [11.9025], [5.08028954]],
              [[1.], [1.06734618], [5.08385966], [5.07819579], [1.07209387]]]])


def test_render_pixelated_cont_shape():
    with goos.OptimizationPlan() as plan:

        def initializer(size):
            return [[1, 0], [0.5, 0.75], [0, 0.8], [1, 0.8]]

        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([200, 0, 0]),
            extents=[1000, 1000, 220],
            material=goos.material.Material(index=2),
            material2=goos.material.Material(index=4),
            pixel_size=[250, 500, 220])

        rect = goos.Cuboid(extents=goos.Constant([1000, 1000, 220]),
                           pos=goos.Constant([200, 0, 0]),
                           material=goos.material.Material(index=2))

        render = maxwell.RenderShape(
            design,
            region=goos.Box3d(center=[0, 0, 0], extents=[1500, 1500, 200]),
            mesh=maxwell.UniformMesh(dx=40),
            background=goos.material.Material(index=1.0),
            wavelength=1550)

        np.testing.assert_almost_equal(
            np.real(render.get().array[2][10:, 8, 3]), [
                1., 8.5, 16., 16., 16., 16., 16., 14.5, 10., 10., 10., 10., 10.,
                10., 4., 4., 4., 4., 4., 4., 13., 16., 16., 16., 16., 16., 8.5
            ])
        #goos.util.visualize_eps(render.get().array[2], z=3)


def test_render_pixelated_cont_shape_grad():
    with goos.OptimizationPlan() as plan:

        def initializer(size):
            return [[1, 0], [0.5, 0.75], [0, 0.8], [1, 0.8]]

        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=[200, 0, 0],
            extents=[1000, 1000, 220],
            material=goos.material.Material(index=2),
            material2=goos.material.Material(index=4),
            pixel_size=[250, 500, 220])

        render = maxwell.RenderShape(
            design,
            region=goos.Box3d(center=[0, 0, 0], extents=[1500, 1500, 200]),
            mesh=maxwell.UniformMesh(dx=40),
            background=goos.material.Material(index=1.0),
            wavelength=1550)

        # Compute a random objective function to test gradient.
        np.random.seed(247)
        obj = goos.dot(np.random.random(render.get().array.shape), render)

        adjoint_grad = obj.get_grad([var])[0].array_grad

        # Calculate brute force gradient.
        var_val = var.get().array
        eps = 0.001
        num_grad = np.zeros_like(var_val)
        for i in range(var_val.shape[0]):
            for j in range(var_val.shape[1]):
                temp_val = var_val.copy()
                temp_val[i, j] += eps
                var.set(temp_val)
                fplus = obj.get(run=True).array

                temp_val = var_val.copy()
                temp_val[i, j] -= eps
                var.set(temp_val)
                fminus = obj.get(run=True).array

                num_grad[i, j] = (fplus - fminus) / (2 * eps)

        np.testing.assert_array_almost_equal(adjoint_grad, num_grad)
