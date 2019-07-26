import numpy as np

from spins import gridlock


def test_draw_slab_2d():
    """Tests that drawing a slab in 2D is correct (no aliasing issues)."""
    edge_coords = [[0, 1.0, 2.0, 3.0, 4, 5, 6], [0, 1.0, 2, 3, 4, 5, 6],
                   [-0.5, 0.5]]
    grid = gridlock.Grid(edge_coords, num_grids=3, initial=10)
    grid.draw_slab(gridlock.Direction.y, 4, 3, 1)
    grid.render()

    np.testing.assert_array_almost_equal(grid.grids[0][2, :, 0],
                                         [10, 10, 10, 1, 1, 1])
    np.testing.assert_array_almost_equal(grid.grids[1][2, :, 0],
                                         [10, 10, 5.5, 1, 1, 5.5])
