
from pycomplex import synthetic


def test_cube():
    for n_dim in [1, 2, 3, 4]:
        cube = synthetic.n_cube(n_dim)
        primal = cube.topology
        primal.check_chain()
        dual = primal.dual
        dual.check_chain()


def test_cube_subdivide():
    for n_dim in [1, 2, 3]:
        cube = synthetic.n_cube(n_dim).subdivide()
        primal = cube.topology
        primal.check_chain()
        dual = primal.dual
        dual.check_chain()


def test_simplex():
    for n_dim in [2, 3, 4]:
        simplex = synthetic.n_simplex(n_dim)
        primal = simplex.topology
        primal.check_chain()
        dual = primal.dual
        dual.check_chain()
