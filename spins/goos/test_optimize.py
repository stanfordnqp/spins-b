import numpy as np
import pytest

from spins import goos


def test_optimize_simple():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([1])
        obj = (x + 1)**2 + 3
        goos.opt.scipy_minimize(obj, method="L-BFGS-B")

        plan.run()

        assert x.get().array == -1


def test_optimize_init_var_no_bracket():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(1)
        obj = (x + 1)**2 + 3
        goos.opt.scipy_minimize(obj, method="L-BFGS-B")

        plan.run()

        assert x.get().array == -1


def test_optimize_2d():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([[1, 2], [3, 4]])
        obj = goos.Norm(x - goos.Constant([[3, 2], [-4, 2]]))**2 + 3
        goos.opt.scipy_minimize(obj, method="L-BFGS-B")

        plan.run()

        np.testing.assert_almost_equal(x.get().array, [[3, 2], [-4, 2]])


def test_optimize_freeze_check():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([1])
        y = goos.Variable([1])

        y.freeze()
        obj = (x + y)**2 + 3
        goos.opt.scipy_minimize(obj, method="L-BFGS-B")

        plan.run()

        assert x.get().array == -1
        assert y.get().array == 1


def test_optimize_lower_bounds():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([1], lower_bounds=0)
        obj = (x + 1)**2 + 3
        goos.opt.scipy_minimize(obj, method="L-BFGS-B")

        plan.run()

        assert x.get().array == 0


def test_optimize_upper_bounds():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([1], upper_bounds=5)
        obj = (x + 1)**2 + 3
        goos.opt.scipy_maximize(obj, method="L-BFGS-B")

        plan.run()

        assert x.get().array == 5


def test_optimizer_eq_constraints():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(1)
        y = goos.Variable(2)
        obj = (x * y - 12)**2
        goos.opt.scipy_minimize(obj, constraints_eq=[y - 3], method="SLSQP")

        plan.run()

        np.testing.assert_allclose(x.get().array, 4)


def test_optimize_ineq_constraints():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable(1)
        obj = (x + 1)**2 + 3
        goos.opt.scipy_maximize(obj, constraints_ineq=[x - 5], method="SLSQP")

        plan.run()

        assert x.get().array == 5


@pytest.mark.skip("Gradients not computed properly for vector functions.")
def test_optimize_ineq_constraints_arr():
    with goos.OptimizationPlan() as plan:
        x = goos.Variable([0, 0])
        obj = (goos.dot([1, 1], x) + 1)**2 + 3
        goos.opt.scipy_maximize(obj,
                                constraints_ineq=[x - np.array([2, 1])],
                                method="SLSQP")

        plan.run()

        np.testing.assert_allclose(x.get().array, [2, 1])


def test_optimize_resume(tmp_path):
    plan_dir = tmp_path / "test_plan"
    plan_dir.mkdir()
    with goos.OptimizationPlan(autorun=True, save_path=plan_dir) as plan:
        x = goos.Variable([1], name="x")
        y = goos.Variable([1], name="y")

        y.freeze()
        obj = (x + y)**4 + 3
        goos.opt.scipy_minimize(obj, method="L-BFGS-B", max_iters=10)

        y.thaw()
        y.set(5)

        x_final = x.get()
        plan.save(plan_dir)

    with goos.OptimizationPlan() as plan:
        plan.load(plan_dir)

        x = plan.get_node("x")
        y = plan.get_node("y")

        plan.read_checkpoint(plan_dir / "step5.pkl")

        assert x.get() != x_final
        assert y.get() != 5

        plan.resume()

        assert y.get() == 5
