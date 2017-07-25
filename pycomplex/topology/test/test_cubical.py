import numpy as np
import numpy.testing as npt
import pytest

from pycomplex.synthetic import n_cube, n_cube_grid
from pycomplex.topology.cubical import *
from pycomplex.topology.test.test_base import basic_test
from pycomplex.math import linalg


def test_generate_boundary():
    n_dim = 3
    cube = n_cube(n_dim)
    cubes = cube.topology.elements[-1]
    for d in range(3):
        b = generate_boundary(cubes, degree=d)


def test_cube_parity():
    for n in [1, 2, 3]:
        cubes = n_cube(n).topology.elements[-1]
        npt.assert_array_equal(cube_parity(cubes), [0])

    # test that diagonal flip is not a valid n-cube
    cubes = n_cube(2).topology.elements[-1]
    cubes = np.swapaxes(cubes, 1, 2)
    with pytest.raises(ValueError):
        cube_parity(cubes)


def test_cube():
    for n_dim in [1, 2, 3, 4]:
        cube = n_cube(n_dim)
        basic_test(cube.topology)


def test_projected_hypercube():
    n_dim = 4
    cube = n_cube(n_dim)
    np.random.seed(1)
    cube.vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))
    cube.as_44().plot(plot_dual=True)


def test_quads():
    """
    0 1 2
    3 4 5
    6 7
    """
    n_dim = 2
    quads = [
        [[0, 1], [3, 4]],
        [[1, 2], [4, 5]],
        [[3, 4], [6, 7]],
    ]

    topology = TopologyCubical.from_cubes(quads)

    basic_test(topology)
    b = topology.boundary
    dual = topology.dual
    assert topology.is_oriented


def test_product():
    """Test product topology functionality"""
    quad = n_cube(2)
    line = n_cube(1)
    c = quad.product(line)


def test_to_simplicial():
    """Check that mapping cubical to simplicial retains orientation"""
    n_dim = 3
    cube = n_cube(n_dim).boundary()
    assert not cube.topology.is_oriented
    assert not cube.as_23().to_simplicial().topology.is_oriented
    cube.topology = cube.topology.fix_orientation()
    assert cube.topology.is_oriented
    assert cube.as_23().to_simplicial().topology.is_oriented

