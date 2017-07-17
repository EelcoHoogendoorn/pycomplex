
import numpy as np
from pycomplex.topology.simplicial import TopologySimplicial, TopologyTriangular
from pycomplex import synthetic


def test_simplicial_2():
    faces = [[0, 1, 2], [3, 2, 1]]

    topology = TopologySimplicial.from_simplices(faces)


def test_simplicial_2_delauney():
    faces = [[0, 1, 2], [3, 2, 1]]
    import scipy.spatial
    hull = scipy.spatial.ConvexHull(np.random.randn(10, 3))
    topology = TopologySimplicial.from_simplices(hull.simplices)


def test_simplicial_3():
    tets = [[0, 1, 2, 3], [4, 3, 2, 1]]

    topology = TopologySimplicial.from_simplices(tets)


def test_subdivide():
    topology = synthetic.n_cube_grid((2, 3)).as_22().to_simplicial().topology

    assert topology.is_oriented


def test_ico():
    sphere = synthetic.icosphere(refinement=0)
    print(sphere.topology.triangles)
    print(sphere.topology.is_closed)
    print(sphere.topology.is_oriented)