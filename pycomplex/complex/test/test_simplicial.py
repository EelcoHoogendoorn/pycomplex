
import numpy as np

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


def test_projected_hypersimplex():
    n_dim = 4
    simplex = synthetic.n_simplex(n_dim)
    simplex.vertices = np.dot(simplex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    simplex.plot(plot_dual=True)
    # simplex.boundary().plot()

test_projected_hypersimplex()