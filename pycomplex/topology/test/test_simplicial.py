
import numpy as np
import numpy.testing as npt
import pytest

from pycomplex import synthetic
from pycomplex.topology.simplicial import *
from pycomplex.topology.test.test_base import basic_test
from pycomplex.math import linalg


def test_permutation_map():
    """Test the permuation map used to relate the orientation of n-simplices"""
    for n in [3]:
        parity, permutation = permutation_map(n)
        print(parity)
        print(permutation)


def test_simplex_parity():
    """Test that we can find the relative parity between simplices"""
    for n in [1, 2, 3]:
        simplex = synthetic.n_simplex(n).topology.elements[-1]
        npt.assert_array_equal(relative_simplex_parity(simplex), [0])


def test_simplex():
    """See if we can construct simplicial topologies of various dimensions"""
    for n in [1, 2, 3, 4, 5]:
        print()
        print(n)
        print()
        simplex = synthetic.n_simplex(n)
        basic_test(simplex.topology)


def test_triangular():
    """Test if we can construct a single triangle topology"""
    tris = [[0, 1, 2]]
    topology = TopologyTriangular.from_simplices(tris)
    basic_test(topology)
    assert topology.is_oriented


def test_tetrahedral():
    """Test if we can construct a single tet topology"""
    tets = [[0, 1, 2, 3]]
    tet = TopologySimplicial.from_simplices(tets)
    basic_test(tet)


def test_boundary():
    """Test that we can construct the boundary topology of a simplex"""
    tet = synthetic.n_simplex(3)
    tris = tet.boundary
    assert tris.topology.is_closed


def test_simplicial_2():
    """Test that we can construct a topology from a pair of triangles"""
    faces = [[0, 1, 2], [3, 2, 1]]
    topology = TopologySimplicial.from_simplices(faces)
    assert topology.is_oriented
    assert not topology.is_closed


def test_simplicial_3():
    """Test that we can construct a topology from a pair of tets"""
    tets = [[0, 1, 2, 3], [4, 3, 2, 1]]
    topology = TopologySimplicial.from_simplices(tets)
    assert topology.is_oriented
    assert not topology.is_closed


def test_icosphere():
    """Test if we can create a triangulated sphere"""
    sphere = synthetic.icosphere(refinement=0)
    assert sphere.topology.is_closed
    assert sphere.topology.is_oriented


def test_sphere_subdivide_cubical():
    """Test that subdividing triangles to cubes retains orientation"""
    sphere = synthetic.icosphere(refinement=0).as_euclidian()
    assert sphere.topology.is_oriented
    quads = sphere.as_3().subdivide_cubical()
    assert quads.topology.is_oriented


def test_from_cubical():
    """Test that we can construct a simplicial topology by subdividing quads"""
    topology = synthetic.n_cube_grid((2, 3)).as_22().subdivide_simplicial().topology

    assert topology.is_oriented
    assert not topology.is_closed


def test_fundamental_domains(show_plot):
    """Test that we can perform fundamental domain subdivision of a topology in arbitrary n-dim"""
    for n in [2, 3, 4]:
        simplex = synthetic.n_simplex(n)
        simplex = simplex.transform(linalg.orthonormalize(np.random.randn(n, n)))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        simplex.plot_domains(ax)
        show_plot()


def test_subdivide_fundamental():
    """Test that we can perform fundamental domain subdivision of a topology in arbitrary n-dim"""
    for n in [2, 3, 4, 5]:
        sphere = synthetic.n_cube_dual(4).as_euclidian()
        sub = sphere.topology.subdivide_fundamental()
        assert sub.is_oriented


def test_subdivide_direct():
    """Test the direct-loop subdivision implementation"""
    triangle = synthetic.n_simplex(2).as_2()
    sub = triangle.topology.subdivide_loop_direct()
    print(sub.transfer_matrices[1])
    sub = sub.subdivide_loop_direct()
    print(sub.is_oriented)


def test_subdivide_direction(show_plot):
    """Test that directionality is inherited as expected"""
    triangle = synthetic.n_simplex(2).as_2()
    sub = triangle.subdivide_loop()
    sub2 = sub.subdivide_loop()

    sub.plot(plot_dual=True, plot_arrow=True)
    sub2.plot(plot_dual=True, plot_arrow=True)
    show_plot()
