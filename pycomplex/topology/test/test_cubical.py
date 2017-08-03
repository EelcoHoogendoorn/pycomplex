import numpy as np
import numpy.testing as npt
import pytest

from pycomplex import synthetic
from pycomplex.topology.cubical import *
from pycomplex.topology.test.test_base import basic_test
from pycomplex.math import linalg


def test_generate_boundary():
    n_dim = 3
    cube = synthetic.n_cube(n_dim)
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
        cubes = synthetic.n_cube(n).topology.elements[-1]
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
        cube = synthetic.n_cube(n_dim)
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
    quad = synthetic.n_cube(2)
    line = synthetic.n_cube(1)
    c = quad.product(line)


def test_to_simplicial():
    """Check that mapping cubical to simplicial retains orientation"""
    cube = synthetic.n_cube(3).boundary
    assert cube.topology.is_oriented
    assert cube.as_23().to_simplicial().topology.is_oriented


def test_hole():
    """Check that non-connected boundaries work fine"""
    mesh = synthetic.n_cube_grid((3, 3))
    # make a hole in it
    mask = np.ones((3, 3), dtype=np.int)
    mask[1, 1] = 0
    mesh = mesh.select_subset(mask.flatten())

    # subdivide
    for i in range(2):
        mesh = mesh.subdivide()

    bt = mesh.boundary.topology
    assert not bt.is_connected

    for i in range(mesh.n_dim):
        npt.assert_allclose(
            mesh.primal_position[i][bt.parent_idx[i]],
            mesh.boundary.primal_position[i]
        )


def test_fundamental_domains():
    for n in [2, 3, 4]:
        print()
        print(n)
        print()
        cube = synthetic.n_cube(n)
        cube.vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(n, n)))

        domains = cube.topology.fundamental_domains()
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,1)
        # simplex.plot_domains(ax)
        # simplex.plot(ax, plot_lines=False)
        # plt.show()
        print(domains.shape)
        print(domains)

test_fundamental_domains()