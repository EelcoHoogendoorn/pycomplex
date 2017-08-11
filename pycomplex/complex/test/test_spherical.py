
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.complex.spherical import ComplexSpherical2
from pycomplex.math import linalg


def test_single():
    """Test a single spherical triangle"""
    sphere = ComplexSpherical2(vertices=np.eye(3), simplices=[[0, 1, 2]])
    sphere = sphere.subdivide()
    fig, ax = plt.subplots(1, 1)
    sphere.plot(ax=ax)
    for i in range(1):
        sphere = sphere.subdivide()
        sphere.plot(ax=ax, primal_color='c', dual_color='m')
    plt.show()


def test_icosahedron():
    """Test a full icosahedron"""
    sphere = synthetic.icosahedron()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    fig, ax = plt.subplots(1, 1)
    sphere.plot(ax=ax, backface_culling=True)
    for i in range(1):
        sphere = sphere.subdivide()
        sphere.plot(ax=ax, primal_color='c', dual_color='m', backface_culling=True)
    plt.show()


def test_overlap():
    # drawing to intuit multigrid transfer operators
    sphere = synthetic.icosahedron().subdivide_fundamental()
    sphere = sphere.select_subset(np.eye(20*6)[0])
    for i in range(2):
        sphere = sphere.subdivide()

    fig, ax = plt.subplots(1, 1)
    sphere.plot(ax=ax)
    sphere = sphere.subdivide()
    sphere.plot(ax=ax, primal_color='c', dual_color='m')
    plt.show()


def test_icosahedron_subset():
    """Test that a concave boundary works just the same on a sphere"""
    sphere = synthetic.icosahedron()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    triangle_position = sphere.primal_position[2]
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
    tet = synthetic.n_simplex(n_dim).boundary.as_spherical().as_2()
    tet = tet.fix_orientation()
    tet.vertices = np.dot(tet.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim)))
    for i in range(0):      # subdivision on a tet gives rather ugly tris
        tet = tet.subdivide()
    tet.plot(backface_culling=True, plot_dual=True)


def test_circle():
    n_dim = 2
    circle = synthetic.n_simplex(n_dim).boundary.as_spherical()
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


def test_picking():
    for n_dim in [2, 3, 4, 5]:
        sphere = synthetic.n_cube_dual(n_dim)
        points = linalg.normalized(np.random.randn(10, n_dim))
        simplex_idx, bary = sphere.pick_primal(points)
        assert np.alltrue(bary >= 0)    # this could fail for points on a boundary
        assert np.allclose(bary.sum(axis=1), 1)
        simplex_idx, bary = sphere.pick_primal(points, simplex_idx)
        assert np.alltrue(bary >= 0)    # this could fail for points on a boundary
        assert np.allclose(bary.sum(axis=1), 1)


def test_picking_alt():
    for n_dim in [2, 3, 4, 5]:
        sphere = synthetic.n_cube_dual(n_dim)
        points = linalg.normalized(np.random.randn(10, n_dim))

        sphere.pick_fundamental(points)

        simplex_idx, bary = sphere.pick_primal(points)
        simplex_idx_alt, bary_alt = sphere.pick_primal_alt(points)

        npt.assert_equal(simplex_idx, simplex_idx_alt)
        npt.assert_allclose(bary, bary_alt)


def test_picking_fundamental():
    sphere = synthetic.icosphere(1)

    # sphere = synthetic.n_cube_dual(3).as_2().subdivide()
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    p = np.linspace(-1, +1, 1024, endpoint=True)
    x, y = np.meshgrid(p, p)
    r2 = x**2 + y**2
    mask = r2 < 1
    z = np.sqrt(1 - np.clip(r2, 0, 1))

    points = np.array([x, y, z]).T
    domain, bary = sphere.pick_fundamental(points.reshape(-1, 3))
    D, B = sphere.pick_fundamental_precomp
    import numpy_indexed as npi
    idx = npi.indices(D.reshape(-1, 3), domain)
    B = B.reshape(-1, 3, 3)[idx]
    color = np.linalg.det(B.astype(np.float32))
    color = color.reshape(len(p), len(p))
    color[np.logical_not(mask)] = 0

    import matplotlib.pyplot as plt
    plt.imshow(color, cmap='bwr')
    plt.show()


def test_fundamental_subdivide():
    sphere = synthetic.icosphere(1)
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    sphere = sphere.subdivide_fundamental()
    sphere.plot(backface_culling=True, plot_vertices=False)
    # FIXME: does not yet work for n > 3
    n = 3
    sphere = synthetic.n_cube_dual(n)
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(n, n)))
    sphere = sphere.subdivide_fundamental()
    sphere.plot(backface_culling=True, plot_vertices=False)
