import pytest

import numpy as np

from spins import goos


def test_parameter_frozen_but_settable():
    """Checks the parameters are always frozen but are always settable."""
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(2, lower_bounds=0)
        param = goos.Variable(3, parameter=True)

        # No error should be thrown here.
        param.set(4)

        obj = (x + param)**2
        goos.opt.scipy_minimize(obj, max_iters=3, method="L-BFGS-B")
        plan.run()

        # Check that `x` is optimized by `param` is constant.
        assert param.get() == 4
        assert x.get() == 0


def test_power():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(3.0)
        y = x**2

        assert y.get() == 9.0
        assert y.get_grad([x]) == [6.0]


def test_sum_single():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(2.0)
        res = goos.Sum([x])

        assert res.get() == 2
        assert res.get_grad([x]) == [1]


def test_sum_double():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(2.0)
        y = goos.Variable(3.0)
        res = x + y

        assert res.get() == 5
        assert res.get_grad([x, y]) == [1, 1]


def test_sum_double_array():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([2.0, 1.0])
        y = goos.Variable([3.0, -1.0])
        res = goos.dot([1, 1], x + y)

        np.testing.assert_allclose(res.get().array, 5)
        np.testing.assert_allclose(res.get_grad([x])[0].array_grad, [1, 1])
        np.testing.assert_allclose(res.get_grad([y])[0].array_grad, [1, 1])


def test_sum_triple():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(2.0)
        y = goos.Variable(3.0)
        res = x + y + 5

        assert res.get() == 10
        assert res.get_grad([x, y]) == [1, 1]


@pytest.mark.parametrize("val,expected", [(3, 3), (-5, 5),
                                          (1 + 1j, np.sqrt(2))])
def test_abs_eval(val, expected):
    with goos.OptimizationPlan() as plan:
        res = goos.AbsoluteValue(goos.Constant(val))
        assert res.get() == expected


@pytest.mark.parametrize("val,expected", [(3, 1), (-5, -1), (1j, -0.5j)])
def test_abs_grad(val, expected):
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(val)
        res = goos.AbsoluteValue(x)

        assert res.get_grad([x])[0] == expected


def test_product_double():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(3)
        res = x * 2

        assert res.get() == 6
        assert res.get_grad([x])[0] == 2


def test_product_double_array():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([3, 1])
        res = goos.Norm(x * goos.Constant([2, 4]))**2

        np.testing.assert_allclose(res.get().array, 52)
        np.testing.assert_allclose(res.get_grad([x])[0].array_grad, [24, 32])


def test_product_triple():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(3)
        y = goos.Variable(4)
        res = x * 2 * y

        assert res.get() == 24
        assert res.get_grad([x, y])[0] == 8
        assert res.get_grad([x, y])[1] == 6


def test_dot():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([3, 2, 1])
        y = goos.Variable([-1, 1, 4])
        res = goos.dot(x, y)

        assert res.get() == 3
        grad = res.get_grad([x, y])
        np.testing.assert_array_equal(grad[0].array_grad, [-1, 1, 4])
        np.testing.assert_array_equal(grad[1].array_grad, [3, 2, 1])


def test_dot_auto_constant():
    with goos.OptimizationPlan() as plan:
        res = goos.dot([3, 2, 1], [-1, 1, 4])

        assert res.get() == 3


@pytest.mark.parametrize("vec,order,expected_val,expected_grad", [
    ([2], 1, 2, [1]),
    ([-2], 1, 2, [-1]),
    ([2, -1], 1, 3, [1, -1]),
    ([2], 2, 2, [1]),
    ([-2], 2, 2, [-1]),
    ([3, -4], 2, 5, [3 / 5, -4 / 5]),
    ([[[2]]], 1, 2, [[[1]]]),
    ([3 + 4j], 1, 5, [(3 - 4j) / 10]),
    ([0], 1, 0, [0]),
])
def test_norm(vec, order, expected_val, expected_grad):
    with goos.OptimizationPlan() as plan:
        var = goos.Variable(vec)
        norm = goos.Norm(var, order=order)

        assert norm.get() == expected_val
        np.testing.assert_almost_equal(
            norm.get_grad([var])[0].array_grad, expected_grad)


def test_max_two_scalar():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(2)
        y = goos.Variable(3)
        z = goos.max(x, y)

        assert z.get() == 3
        assert z.get_grad([x])[0] == 0
        assert z.get_grad([y])[0] == 1


def test_max_three_scalars():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(2)
        y = goos.Variable(3)
        w = goos.Variable(1)
        z = goos.max(x, y, w)

        assert z.get() == 3
        assert z.get_grad([x])[0] == 0
        assert z.get_grad([y])[0] == 1
        assert z.get_grad([w])[0] == 0


def test_max_two_array():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([2, 1])
        y = goos.Variable([3, 0.5])
        z = goos.dot(goos.max(x, y), [1, -1])

        np.testing.assert_array_equal(z.get().array, 2)
        np.testing.assert_array_equal(z.get_grad([x])[0].array_grad, [0, -1])
        np.testing.assert_array_equal(z.get_grad([y])[0].array_grad, [1, 0])


def test_max_two_array_2d():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([[1, 2], [3, 4]])
        y = goos.Variable([[3, 0.5], [4.5, 6]])
        z = goos.dot([[1, 2], [3, 4]], goos.max(x, y))

        np.testing.assert_array_equal(z.get().array, 44.5)
        np.testing.assert_array_equal(
            z.get_grad([x])[0].array_grad, [[0, 2], [0, 0]])
        np.testing.assert_array_equal(
            z.get_grad([y])[0].array_grad, [[1, 0], [3, 4]])


def test_sigmoid_scalar():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(0.5)
        y = goos.Sigmoid(x)

        np.testing.assert_allclose(y.get().array, 1 / (1 + np.exp(-0.5)))
        np.testing.assert_allclose(
            y.get_grad([x])[0].array_grad,
            np.exp(-0.5) / (1 + np.exp(-0.5))**2)


def test_sigmoid_array():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([0, 1])
        y = goos.dot([1, 2], goos.Sigmoid(x))

        np.testing.assert_allclose(y.get().array, 1.962117)
        np.testing.assert_allclose(
            y.get_grad([x])[0].array_grad, [0.25, 0.39322386648])


def test_sigmoid_with_scalar_ops():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([0, 1])
        y = goos.dot([1, 2], goos.Sigmoid(2 * x - 1))

        np.testing.assert_allclose(y.get().array, 1.7310585786300)
        np.testing.assert_allclose(
            y.get_grad([x])[0].array_grad, [0.39322386648296, 0.7864477329659])


def test_slice():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([[0, 1, 2, 4, 5], [6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                           [21, 22, 23, 24, 25]])

        t = goos.Slice(x, ['c', 'c'])
        np.testing.assert_allclose(t.get().array, 13)
        g = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(t.get_grad([x])[0].array_grad, g)

        t = goos.Slice(x, [[1, 4], 'c'])
        np.testing.assert_allclose(t.get().array, [[8], [13], [18]])
        g = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(t.get_grad([x])[0].array_grad, g)

        t = goos.Slice(x, [3, [1, 3]])
        np.testing.assert_allclose(t.get().array, [[17, 18]])
        g = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(t.get_grad([x])[0].array_grad, g)

        t = goos.Slice(x, [3, None])
        np.testing.assert_allclose(t.get().array, [[16, 17, 18, 19, 20]])
        g = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(t.get_grad([x])[0].array_grad, g)
