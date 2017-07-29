
import numpy as np

from pycomplex import synthetic
from pycomplex.complex.spherical import ComplexSpherical2
from pycomplex.math import linalg


def test_single():
    """Test a single spherical triangle"""
    sphere = ComplexSpherical2(vertices=np.eye(3), simplices=[[0, 1, 2]])
    sphere.plot()


def test_ico():
    """Test a full icosahedron"""
    sphere = synthetic.icosahedron()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    sphere = sphere.subdivide()
    sphere.plot()


def test_ico_subset():
    """Test that a concave boundary works just the same on a sphere"""
    sphere = synthetic.icosahedron()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    triangle_position = sphere.primal_position()[2]
    selection = triangle_position[:, 2] != triangle_position[:,2].max()
    sphere = sphere.select_subset(selection)
    sphere = sphere.subdivide()
    sphere = sphere.subdivide()

    sphere.plot(plot_dual=True, backface_culling=True)


def test_subdivide():
    """Test if subdivision works well for big triangles up to 90deg angle too"""
    sphere = ComplexSpherical2(vertices=linalg.normalized(np.eye(3)), simplices=[[0, 1, 2]])
    sphere = sphere.subdivide()
    sphere = sphere.subdivide()
    sphere = sphere.subdivide()
    sphere.plot(plot_dual=True)


def test_tetra():
    n_dim = 4
    tet = synthetic.n_simplex(n_dim)
    tet.vertices = np.dot(tet.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    tet.boundary().as_spherical().plot(backface_culling=False, plot_dual=False)
    print(np.linalg.norm(tet.vertices, axis=1))
