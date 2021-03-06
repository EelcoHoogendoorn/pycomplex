
import numpy as np
import numpy.testing as npt
from pycomplex import synthetic


def test_pick_primal_2():
    grid = synthetic.n_cube_grid((20, 30), centering=False).as_22().as_regular()
    idx, bary = grid.pick_primal([[0, 0]])
    idx, bary = grid.pick_primal([[1, 1]])
    idx, bary = grid.pick_primal([[-20, -30]])
    assert idx == 0

    points = np.random.uniform(0, 1, 1000).reshape(-1, 2) * [20, 30]
    idx, bary = grid.pick_primal(points)

    assert np.alltrue(bary >= 0)
    assert np.alltrue(bary <= 1)


def test_metric():
    n = 2
    grid = synthetic.n_cube_grid((3, ) * n, centering=False).as_22().as_regular()
    PM, DM = grid.metric
    print(DM)
    for pm in PM:
        assert np.allclose(pm, 1)
