
import numpy as np
import numpy.testing as npt

from pycomplex import synthetic
from pycomplex.topology.simplicial import *
from pycomplex.topology.test.test_base import basic_test



def test_permutation_map():

    for n in [3]:
        parity, permutation = permutation_map(n)
        print(parity)
        print(permutation)
# test_permutation_map()
# quit()

def test_simplex():
    for n in [3]:
        print()
        print(n)
        print()
        simplex = synthetic.simplex(n)
        basic_test(simplex.topology)
#
# test_simplex()

def test_triangular():
    tris = [[0, 1, 2]]
    topology = TopologyTriangular.from_simplices(tris)
    basic_test(topology)
    assert topology.is_oriented


def test_tetrahedral():
    tets = [[0, 1, 2, 3]]
    tet = TopologyTetrahedral.from_simplices(tets)
test_tetrahedral()
quit()

def test_simplex_parity():
    for n in [1, 2, 3]:
        simplex = synthetic.simplex(n).topology.elements[-1]
        npt.assert_array_equal(simplex_parity(simplex), [0])


def test_boundary():
    tet = synthetic.simplex(3)
    tris = tet.boundary()


def test_simplicial_2():
    faces = [[0, 1, 2], [3, 2, 1]]
    topology = TopologySimplicial.from_simplices(faces)
    assert topology.is_oriented
    assert not topology.is_closed


def test_simplicial_2_delauney():
    faces = [[0, 1, 2], [3, 2, 1]]
    import scipy.spatial
    hull = scipy.spatial.ConvexHull(np.random.randn(10, 3))
    topology = TopologySimplicial.from_simplices(hull.simplices)


def test_simplicial_3():
    tets = [[0, 1, 2, 3], [4, 3, 2, 1]]
    topology = TopologySimplicial.from_simplices(tets)
    assert topology.is_oriented
    assert not topology.is_closed


def test_ico():
    sphere = synthetic.icosphere(refinement=0)
    assert sphere.topology.is_closed
    assert sphere.topology.is_oriented


def test_to_cubical():
    """Test that mapping to cubical retains orientation"""
    sphere = synthetic.icosphere(refinement=0).as_euclidian()
    assert sphere.topology.is_oriented
    quads = sphere.as_2().to_cubical()
    assert quads.topology.is_oriented


def test_from_cubical():
    topology = synthetic.n_cube_grid((2, 3)).as_22().to_simplicial().topology

    assert topology.is_oriented
    assert not topology.is_closed
