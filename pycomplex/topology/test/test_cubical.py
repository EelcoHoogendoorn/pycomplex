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
        b = generate_cube_boundary(cubes, degree=d)


def test_permutation_map():
    a, b = permutation_map(2, rotations=True)
    assert (len(npi.unique(b)) == 8)
    a, b = permutation_map(2, rotations=False)
    assert (len(npi.unique(b)) == 4)
    a, b = permutation_map(3, rotations=True)
    assert (len(npi.unique(b)) == 48)
    a, b = permutation_map(3, rotations=False)
    assert (len(npi.unique(b)) == 8)


def test_cube_parity():
    for n in [1, 2, 3]:
        cubes = n_cube(n).topology.elements[-1]
        npt.assert_array_equal(relative_cube_parity(cubes), [0])


def test_cube_parity_raises():
    # test that this is not a valid n-cube
    cubes = [[[0, 1], [3, 2]]]
    with pytest.raises(ValueError):
        relative_cube_parity(cubes)


def test_cube():
    for n_dim in [1, 2, 3, 4]:
        print()
        print(n_dim)
        print()
        cube = n_cube(n_dim)
        basic_test(cube.topology)


def test_non_regular():
    cubes = [
        [[0, 1], [2, 3]],
        # [[0, 4], [1, 5]],
        [[5, 1], [4, 0]],
    ]
    t = TopologyCubical.from_elements(cubes)
    basic_test(t)


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
    cube = n_cube(n_dim).boundary
    # assert not cube.topology.is_oriented
    # assert not cube.as_23().to_simplicial().topology.is_oriented
    # cube.topology = cube.topology.fix_orientation()
    assert cube.topology.is_oriented
    assert cube.as_23().to_simplicial().topology.is_oriented

