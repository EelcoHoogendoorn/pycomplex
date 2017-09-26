
import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.math import linalg


def test_optimal_sphere():
    n_dim = 4
    complex = synthetic.optimal_delaunay_sphere(100, n_dim, iterations=50, weights=False, condition=None)
    # even this fairly weak condition fails in ndim > 3 still...
    complex = complex.optimize_weights()
    print(complex.is_pairwise_delaunay)
    print(complex.is_well_centered)

    # complex.plot(backface_culling=n_dim==3, plot_vertices=False)
    # plt.show()


def test_delaunay_cube():
    n_dim = 2
    complex = synthetic.delaunay_cube(3, n_dim=n_dim)
    complex = complex.copy(vertices = np.dot(complex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim))))

    print(complex.is_pairwise_delaunay)
    print(complex.is_well_centered)
    complex = complex.optimize_weights()

    print(complex.is_pairwise_delaunay)
    print(complex.is_well_centered)

    # complex.plot()
    # plt.show()
