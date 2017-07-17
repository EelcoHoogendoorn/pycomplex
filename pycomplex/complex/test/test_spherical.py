
import numpy as np

from pycomplex import synthetic
from pycomplex.complex.spherical import ComplexSpherical
from pycomplex.math import linalg


def test_single():
    """Test a single spherical triangle"""
    sphere = ComplexSpherical(vertices=np.eye(3), triangles=[[0, 1, 2]])
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

    selection = np.delete(np.arange(20), sphere.primal_position()[2][:, 2].argmax())
    sphere = ComplexSpherical(vertices=sphere.vertices, triangles=sphere.topology.elements[-1][selection])
    sphere = sphere.subdivide()
    sphere = sphere.subdivide()

    sphere.plot()


def test_subdivide():
    """Test if subdivision works well for big triangles up to 90deg angle too"""
    sphere = ComplexSpherical(vertices=linalg.normalized(np.eye(3)), triangles=[[0, 1, 2]])
    sphere = sphere.subdivide()
    sphere = sphere.subdivide()
    sphere = sphere.subdivide()
    sphere.plot(plot_dual=True)
