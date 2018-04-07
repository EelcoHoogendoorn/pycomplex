
import os

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.math import linalg


def test_save_STL(show_plot):
    """Test a roundtrip to stl"""
    sphere = synthetic.optimal_delaunay_sphere(200, 3).as_euclidian().as_2().as_3()

    sphere.save_STL('test.stl')
    sphere = sphere.load_STL('test.stl')
    os.remove('test.stl')

    sphere.as_spherical().plot(backface_culling=True)
    show_plot()


def test_delaunay(show_plot):
    """Triangulate a quad. Does this still have any relevance now that it is part of the synthetic module?"""
    boundary = synthetic.n_cube(2).boundary
    for i in range(3):
        boundary = boundary.subdivide_cubical()

    points = np.concatenate([
        boundary.vertices,
        np.random.uniform(0, 1, (100, 2))
    ], axis=0)

    import scipy.spatial
    delaunay = scipy.spatial.Delaunay(points)

    from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian2
    quad = ComplexTriangularEuclidian2(vertices=points, triangles=delaunay.simplices)
    assert quad.topology.is_oriented
    assert quad.topology.is_connected
    assert quad.topology.boundary.is_closed
    assert quad.topology.boundary.is_connected
    assert quad.topology.boundary.is_oriented
    quad.plot(plot_dual=False)
    show_plot()


def test_metric_2(show_plot):
    sphere = synthetic.optimal_delaunay_sphere(200, 3).as_euclidian().as_2().as_3()
    # assert sphere.is_well_centered

    pm, dm = sphere.metric
    for i, m in enumerate(pm):
        print(i)
        print(m.min(), m.max())
    for i, m in enumerate(dm):
        print(i)
        print(m.min(), m.max())

    sphere.plot_3d(backface_culling=True)
    show_plot()


def test_extrude(show_plot):
    quad = synthetic.delaunay_cube(n_dim=2, density=10)
    from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
    # add a z=0 coordinate
    quad = ComplexTriangularEuclidian3(
        vertices=np.concatenate((quad.vertices, quad.vertices[:,:1]*0), axis=1),
        topology=quad.topology.as_2()
    )

    extruded = quad.extrude(quad.copy(vertices=quad.vertices + [[0, 0, .1]]))
    extruded.transform(linalg.orthonormalize(np.random.randn(3, 3))).plot_3d(backface_culling=True)
    show_plot()


def test_flux_to_vector(show_plot):
    """Test if a constant gradient potential produces constant vectors"""
    quad = synthetic.delaunay_cube(n_dim=2, density=10)
    # potential that is a linear gradient
    phi_p0 = quad.primal_position[0][:, 0]# * quad.primal_position[0][:, 1]

    P01, P12 = quad.topology.matrices
    flux_p1 = P01.T * phi_p0
    # check what happens to non-solenoidal components
    # flux_p1 += np.random.randn(*flux_p1.shape) * 1e-2
    flux_d1 = quad.hodge_DP[1] * flux_p1
    # add zero boundary terms
    flux_d1 = quad.topology.dual.selector[1].T * flux_d1
    velocity_d0 = quad.dual_flux_to_dual_velocity(flux_d1)

    npt.assert_allclose(velocity_d0, [[0, -1]] * len(velocity_d0), atol=1e-9, rtol=1e9)

    quad.plot(plot_dual=False)
    plt.quiver(*quad.primal_position[2].T, *velocity_d0.T)
    show_plot()
