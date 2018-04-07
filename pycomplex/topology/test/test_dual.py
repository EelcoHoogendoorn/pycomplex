
from pycomplex import synthetic


def test_cube():
    """Check that we get a valid cochain complex from our n-cube constructor"""
    for n_dim in [1, 2, 3, 4]:
        cube = synthetic.n_cube(n_dim)
        primal = cube.topology
        primal.check_chain()
        dual = primal.dual
        dual.check_chain()


def test_cube_subdivide():
    """Check that we get a valid cochain complex from our n-cube constructor plus cubical subdivision"""
    for n_dim in [1, 2, 3]:
        cube = synthetic.n_cube(n_dim).subdivide_cubical()
        primal = cube.topology
        primal.check_chain()
        dual = primal.dual
        dual.check_chain()


def test_simplex():
    """Check that we get a valid cochain complex from our n-simplex constructor"""
    for n_dim in [2, 3, 4]:
        simplex = synthetic.n_simplex(n_dim)
        primal = simplex.topology
        primal.check_chain()
        dual = primal.dual
        dual.check_chain()
