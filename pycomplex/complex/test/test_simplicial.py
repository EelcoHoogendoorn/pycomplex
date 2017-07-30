
import numpy as np
import numpy.testing as npt

from pycomplex import synthetic
from pycomplex.complex.simplicial import ComplexTriangularEuclidian2, ComplexTriangularEuclidian3
from pycomplex.math import linalg


def test_triangular():
    n_dim = 2
    vertices = [
        [1, 0],
        [0, 2],
        [-1, 0],
    ]

    tris = [
        [0, 1, 2],
    ]

    complex = ComplexTriangularEuclidian2(vertices=vertices, triangles=tris)

    complex.plot(plot_dual=True)


def test_sphere():
    sphere = synthetic.icosahedron().as_euclidian()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    for i in range(3):
        sphere = sphere.subdivide(smooth=True)

    sphere.plot_3d(backface_culling=True)


def test_n_simplex():
    for n_dim in [2, 3, 4, 5]:
        simplex = synthetic.n_simplex(n_dim)
        simplex.vertices = np.dot(simplex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

        assert simplex.topology.is_oriented
        assert simplex.topology.is_connected
        assert simplex.topology.boundary.is_connected
        assert simplex.topology.boundary.is_oriented
        assert simplex.topology.boundary.is_closed

        simplex.plot(plot_dual=True)

test_n_simplex()

def test_subdivided_triangle():
    tri = synthetic.n_simplex(2).as_2().as_2()
    for i in range(5):
        tri = tri.subdivide()
    tri.plot()


def test_delaunay():
    """Triangulate a quad """
    import scipy.spatial
    boundary = synthetic.n_cube(2).boundary
    for i in range(3):
        boundary = boundary.subdivide()

    points = np.concatenate([
        boundary.vertices,
        np.random.uniform(0, 1, (100, 2))
    ], axis=0)

    delaunay = scipy.spatial.Delaunay(points)

    quad = ComplexTriangularEuclidian2(vertices=points, triangles=delaunay.simplices)
    assert quad.topology.is_oriented
    assert quad.topology.is_connected
    assert quad.topology.boundary.is_closed
    assert quad.topology.boundary.is_connected
    assert quad.topology.boundary.is_oriented
    quad.plot(plot_dual=False)
