
import numpy as np
import numpy.testing as npt

from pycomplex import synthetic
from pycomplex.topology.simplicial import *
from pycomplex.topology.test.test_base import basic_test
from pycomplex.math import linalg


def test_permutation_map():
    for n in [3]:
        parity, permutation = permutation_map(n)
        print(parity)
        print(permutation)


def test_simplex():
    for n in [1, 2, 3, 4, 5]:
        print()
        print(n)
        print()
        simplex = synthetic.n_simplex(n)
        basic_test(simplex.topology)


def test_triangular():
    tris = [[0, 1, 2]]
    topology = TopologyTriangular.from_simplices(tris)
    basic_test(topology)
    assert topology.is_oriented


def test_tetrahedral():
    tets = [[0, 1, 2, 3]]
    tet = TopologySimplicial.from_simplices(tets)
    basic_test(tet)


def test_simplex_parity():
    for n in [1, 2, 3]:
        simplex = synthetic.n_simplex(n).topology.elements[-1]
        npt.assert_array_equal(relative_simplex_parity(simplex), [0])


def test_boundary():
    tet = synthetic.n_simplex(3)
    tris = tet.boundary
    assert tris.topology.is_closed


def test_simplicial_2():
    faces = [[0, 1, 2], [3, 2, 1]]
    topology = TopologySimplicial.from_simplices(faces)
    assert topology.is_oriented
    assert not topology.is_closed


def test_simplicial_3():
    tets = [[0, 1, 2, 3], [4, 3, 2, 1]]
    topology = TopologySimplicial.from_simplices(tets)
    assert topology.is_oriented
    assert not topology.is_closed


def test_ico():
    sphere = synthetic.icosphere(refinement=0)
    assert sphere.topology.is_closed
    assert sphere.topology.is_oriented


def test_subdivide_cubical():
    """Test that mapping to cubical retains orientation"""
    sphere = synthetic.icosphere(refinement=0).as_euclidian()
    assert sphere.topology.is_oriented
    quads = sphere.as_2().subdivide_cubical()
    quads = sphere.as_2().subdivide_cubical()
    assert quads.topology.is_oriented


def test_from_cubical():
    topology = synthetic.n_cube_grid((2, 3)).as_22().subdivide_simplicial().topology

    assert topology.is_oriented
    assert not topology.is_closed


def test_fundamental_domains():
    for n in [2, 3, 4]:
        simplex = synthetic.n_simplex(n)
        simplex = simplex.copy(vertices = np.dot(simplex.vertices, linalg.orthonormalize(np.random.randn(n, n))))

        domains = simplex.topology.fundamental_domains()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        simplex.plot_domains(ax)
        plt.show()


def test_subdivide_fundamental():
    for n in [2, 3, 4, 5]:
        sphere = synthetic.n_cube_dual(4).as_euclidian()
        domains = sphere.topology.fundamental_domains()
        sub = sphere.topology.subdivide_fundamental()
        print(sub.is_oriented)
