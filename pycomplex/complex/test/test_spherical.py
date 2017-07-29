
import numpy as np
import numpy.testing as npt

from pycomplex import synthetic
from pycomplex.complex.spherical import ComplexSpherical2
from pycomplex.math import linalg


def test_single():
    """Test a single spherical triangle"""
    sphere = ComplexSpherical2(vertices=np.eye(3), simplices=[[0, 1, 2]])
    sphere.plot()


def test_icosahedron():
    """Test a full icosahedron"""
    sphere = synthetic.icosahedron()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    for i in range(0):
        sphere = sphere.subdivide()
    sphere.plot(backface_culling=True)


def test_icosahedron_subset():
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


def test_tetrahedron():
    n_dim = 3
    tet = synthetic.n_simplex(n_dim).boundary().as_spherical().as_2()
    tet.topology = tet.topology.fix_orientation()
    tet.vertices = np.dot(tet.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))
    for i in range(0):      # subdivision on a tet gives rather ugly tris
        tet = tet.subdivide()
    tet.plot(backface_culling=True, plot_dual=True)


def test_circle():
    n_dim = 2
    circle = synthetic.n_simplex(n_dim).boundary().as_spherical()
    # circle.topology = circle.topology.fix_orientation()
    circle.plot(backface_culling=False, plot_dual=True)


def test_hexacosichoron():

    complex = synthetic.hexacosichoron()

    deg = complex.topology.vertex_degree()
    npt.assert_equal(deg, 20)
    assert complex.topology.is_oriented
    assert complex.topology.n_elements == [120, 720, 1200, 600]

    edges = complex.topology.elements[1]
    edges = complex.vertices[edges]
    length = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1)
    npt.assert_allclose(length, length[0])

    n_dim = complex.n_dim
    complex.vertices = np.dot(complex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))
    complex.plot(plot_dual=True, backface_culling=False)


def test_n_cube_dual():
    for n_dim in [2, 3, 4, 5]:
        complex = synthetic.n_cube_dual(n_dim)

        complex.vertices = np.dot(complex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))
        complex.plot(plot_dual=True, backface_culling=n_dim==3)
