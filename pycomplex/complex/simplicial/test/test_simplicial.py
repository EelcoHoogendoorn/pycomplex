"""General tests not very specific to euclidian or spherical metric"""

import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.math import linalg


def test_subdivide_cubical(show_plot):
    """Test the subdivision of a simplex into cubes"""
    n_dim = 3
    simplex = synthetic.n_simplex(n_dim)
    simplex = simplex.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    cubes = simplex.subdivide_cubical().fix_orientation()
    assert cubes.topology.is_oriented
    assert cubes.topology.is_connected
    cubes.plot(plot_dual=False)
    show_plot()


def test_subdivide_simplicial(show_plot):
    """Test the subdivision of a simplex into simplices"""
    for n in [2, 3, 4]:
        simplex = synthetic.n_simplex(n)
        simplex = simplex.transform(linalg.orthonormalize(np.random.randn(n, n)))

        simplices = simplex.subdivide_simplicial()
        assert simplices.topology.is_oriented
        simplices.plot(plot_dual=False)
        show_plot()


def test_subdivide_cubical_many(show_plot):
    # sphere = synthetic.hexacosichoron().as_euclidian()
    n_dim = 3
    sphere = synthetic.n_cube_dual(n_dim=n_dim).as_euclidian()
    sphere = sphere.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    cubes = sphere.subdivide_cubical().smooth().subdivide_cubical().smooth()
    cubes.plot(plot_dual=False)
    show_plot()


def test_sphere(show_plot):
    """Test a simple loop-subdivided sphere"""
    sphere = synthetic.icosahedron().as_euclidian()
    sphere = sphere.transform(linalg.orthonormalize(np.random.randn(3, 3)))

    for i in range(3):
        sphere = sphere.subdivide_loop(smooth=True)

    sphere.plot_3d(backface_culling=True)
    show_plot()


def test_n_simplex(show_plot):
    """Test n-simplices, in a range of dimensions"""
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
        show_plot()


def test_subdivided_triangle(show_plot):
    """Simple test of loop subdivision on a bounded domain"""
    tri = synthetic.n_simplex(2).as_2().as_2()
    for i in range(5):
        tri = tri.subdivide_loop()
    tri.plot(plot_dual=True)
    show_plot()


def test_subdivided_mesh(show_plot):
    """Test involving combination of cube and loop subdivision"""
    surface = synthetic.n_cube(3).boundary.subdivide_fundamental().as_2().as_3()
    surface = surface.smooth()
    for i in range(1):
        surface = surface.subdivide_loop(smooth=True)

    surface = surface.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    surface.plot_3d(plot_dual=True, backface_culling=True)
    show_plot()


def test_power_dual(show_plot):
    """Put power dual functionality through the motions"""
    tri = synthetic.n_simplex(2).as_2().as_2()
    for i in range(2):
        tri = tri.subdivide_loop()
    tri = tri.copy(
        vertices=tri.vertices + np.random.normal(0, 0.03, size=tri.vertices.shape),
        # weights=np.random.uniform(0, tri.primal_metric[1].mean()**2/1, tri.topology.n_elements[0])
    )
    print(tri.is_pairwise_delaunay)
    print(tri.is_well_centered)
    tri = tri#.optimize_weights()
    tri.plot()
    fundamental = tri.subdivide_fundamental()

    # fundamental = fundamental.optimize_weights_metric()
    # fundamental = fundamental.optimize_weights()
    fundamental = fundamental.optimize_weights_fundamental()
    print(fundamental.is_pairwise_delaunay)
    print(fundamental.is_well_centered)

    fundamental.plot()
    show_plot()


def test_metric():
    """Simple test of simplicial metric"""
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
