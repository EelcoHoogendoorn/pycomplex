
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian2
from pycomplex.math import linalg


def test_subdivide_cubical():
    simplex = synthetic.n_simplex(3)
    simplex = simplex.transform(linalg.orthonormalize(np.random.randn(3, 3)))

    cubes = simplex.subdivide_cubical()
    cubes.plot(plot_dual=False)
    plt.show()


def test_subdivide_simplicial():
    for n in [2, 3, 4]:
        simplex = synthetic.n_simplex(n)
        simplex = simplex.transform(linalg.orthonormalize(np.random.randn(n, n)))

        simplices = simplex.subdivide_simplicial()
        assert simplices.topology.is_oriented
        simplices.plot(plot_dual=False)
        plt.show()


def test_subdivide_cubical_many():
    # sphere = synthetic.hexacosichoron().as_euclidian()
    sphere = synthetic.n_cube_dual(n_dim=4).as_euclidian()
    sphere = sphere.transform(linalg.orthonormalize(np.random.randn(4, 4)))

    cubes = sphere.subdivide_cubical().smooth()#.subdivide().smooth()
    cubes.plot(plot_dual=False)
    plt.show()


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
    sphere = sphere.transform(linalg.orthonormalize(np.random.randn(3, 3)))

    for i in range(3):
        sphere = sphere.subdivide_loop(smooth=True)

    sphere.plot_3d(backface_culling=True)
    plt.show()


def test_n_simplex():
    for n_dim in [2, 3, 4, 5, 6, 7]:
        simplex = synthetic.n_simplex(n_dim)
        simplex = simplex.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

        assert simplex.topology.is_oriented
        assert simplex.topology.is_connected
        assert not simplex.topology.is_closed
        assert simplex.topology.boundary.is_connected
        assert simplex.topology.boundary.is_oriented
        assert simplex.topology.boundary.is_closed

        simplex.plot(plot_dual=True)
        plt.show()


def test_subdivided_triangle():
    tri = synthetic.n_simplex(2).as_2().as_2()
    for i in range(5):
        tri = tri.subdivide_loop()


def test_power_dual():
    tri = synthetic.n_simplex(2).as_2().as_2()
    for i in range(2):
        tri = tri.subdivide_loop()
    tri = tri.copy(
        vertices=tri.vertices + np.random.normal(0, 0.06, size=tri.vertices.shape),
        # weights=np.random.uniform(0, tri.primal_metric[1].mean()**2/1, tri.topology.n_elements[0])
    )
    print(tri.is_pairwise_delaunay)
    print(tri.is_well_centered)
    tri = tri.optimize_weights()
    tri.plot()
    fundamental = tri.subdivide_fundamental()


    # fundamental = fundamental.optimize_weights_metric()
    # fundamental = fundamental.optimize_weights()
    fundamental = fundamental.optimize_weights_fundamental()
    print(fundamental.is_pairwise_delaunay)
    print(fundamental.is_well_centered)

    fundamental.plot()
    plt.show()


def test_delaunay():
    """Triangulate a quad """
    import scipy.spatial
    boundary = synthetic.n_cube(2).boundary
    for i in range(3):
        boundary = boundary.subdivide_cubical()

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
    plt.show()


def test_metric():
    sphere = synthetic.icosahedron()

    for i in range(2):
        sphere = sphere.subdivide_loop()

    pm, dm = sphere.metric
    for i, m in enumerate(pm):
        print(i)
        print(m.min(), m.max())
    for i, m in enumerate(dm):
        print(i)
        print(m.min(), m.max())

    sphere = sphere.as_euclidian()

    print()
    pm, dm = sphere.metric

    for i, m in enumerate(pm):
        print(i)
        print(m.min(), m.max())
    for i, m in enumerate(dm):
        print(i)
        print(m.min(), m.max())


def test_metric_2():
    sphere = synthetic.optimal_delaunay_sphere(200, 3).as_2().as_euclidian()
    # assert sphere.is_well_centered

    pm, dm = sphere.metric
    for i, m in enumerate(pm):
        print(i)
        print(m.min(), m.max())
    for i, m in enumerate(dm):
        print(i)
        print(m.min(), m.max())

    sphere.plot_3d(backface_culling=True)
    plt.show()
