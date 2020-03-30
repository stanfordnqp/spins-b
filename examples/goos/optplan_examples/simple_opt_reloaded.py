import os

from spins import goos


def main(save_folder: str, checkpoint_file: str):
    with goos.OptimizationPlan() as plan:
        # Automatically loads _problem graph_ from the save path.
        plan.load(save_folder)

        # Retrieve our variables.
        x = plan.get_node("x")
        y = plan.get_node("y")
        obj = plan.get_node("obj")

        # Note that we have not loaded any saved variable data yet!
        # `x` is still currently 3.

        # Now load the variable state (checkpoint).
        plan.read_checkpoint(os.path.join(save_folder, checkpoint_file))

        # Show that we have retrieved the values.
        x_val, y_val, obj_val = plan.eval_nodes([x, y, obj])
        print("x: {}, y: {}, obj: {}".format(x_val.array, y_val.array,
                                             obj_val.array))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("save_folder")
    parser.add_argument("checkpoint_file")

    args = parser.parse_args()

    main(args.save_folder, args.checkpoint_file)
