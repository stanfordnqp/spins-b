import os

from spins import goos


def main(save_folder: str):
    goos.util.setup_logging(save_folder)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        x = goos.Variable(3, name="x")
        y = goos.Variable(1, name="y")

        obj = goos.rename((x + y)**2 + (y - 2)**2, name="obj")

        # First optimize only `x`.
        y.freeze()

        goos.opt.scipy_minimize(obj, "L-BFGS-B", max_iters=10)

        # Now do co-optimization.
        y.thaw()
        goos.opt.scipy_minimize(obj, "L-BFGS-B", max_iters=10)

        plan.save()
        plan.run()

        # More efficient to call `eval_nodes` when evaluating multiple nodes
        # at the same time.
        x_val, y_val, obj_val = plan.eval_nodes([x, y, obj])
        print("x: {}, y: {}, obj: {}".format(x_val.array, y_val.array,
                                             obj_val.array))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("save_folder")

    args = parser.parse_args()

    main(args.save_folder)
