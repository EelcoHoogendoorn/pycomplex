
import numpy as np

from pycomplex import synthetic
from pycomplex.math import linalg


def test_optimal_sphere(show_plot):
    """Test the delaunay tesselation of an n-sphere"""
    n_dim = 4
    complex = synthetic.optimal_delaunay_sphere(100, n_dim, iterations=50, weights=False, condition=None)
    # even this fairly weak condition often fails in ndim > 3 still...
    complex = complex.optimize_weights()
    print(complex.is_pairwise_delaunay)
    print(complex.is_well_centered)

    complex.plot(backface_culling=n_dim==3, plot_vertices=False)
    show_plot()


def test_delaunay_cube(show_plot):
    """Test the delaunay tesselation of an n-cube"""
    n_dim = 2
    complex = synthetic.delaunay_cube(density=4, n_dim=n_dim)
    complex = complex.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    print(complex.is_pairwise_delaunay)
    print(complex.is_well_centered)
    complex = complex.optimize_weights()

    print(complex.is_pairwise_delaunay)
    print(complex.is_well_centered)

    complex.plot()
    show_plot()
