
import os

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from pycomplex import synthetic
from pycomplex.math import linalg
from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3, ComplexTriangularEuclidian2


def test_save_STL():
    sphere = synthetic.optimal_delaunay_sphere(200, 3).as_euclidian().as_2().as_3()


    sphere.save_STL('test.stl')
    sphere = ComplexTriangularEuclidian3.load_STL('test.stl')
    os.remove('test.stl')

    sphere.plot_3d(backface_culling=True)
    plt.show()


def test_delaunay():
    """Triangulate a quad """
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


def test_metric_2():
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
    plt.show()


def test_extrude():
    quad = synthetic.delaunay_cube(n_dim=2)
    from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
    quad = ComplexTriangularEuclidian3(
        vertices=np.concatenate((quad.vertices, quad.vertices[:,:1]*0), axis=1),
        topology=quad.topology.as_2()
    )

    extruded = quad.extrude(quad.copy(vertices=quad.vertices + [[0, 0, 1]]))
    extruded.transform(linalg.orthonormalize(np.random.randn(3, 3))).plot_3d(backface_culling=True)
    plt.show()