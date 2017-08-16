
import numpy as np
from pycomplex import synthetic


def test_optimal_sphere():
    n_dim = 4
    complex = synthetic.optimal_delaunay_sphere(100, n_dim, iterations=50, weights=False, condition=None)
    # even this fairly weak condition fails in ndim > 3 still...
    complex = complex.optimize_weights_metric()
    print(complex.is_pairwise_delaunay)

    complex.plot(backface_culling=n_dim==3, plot_vertices=False)
    import matplotlib.pyplot as plt
    plt.show()

test_optimal_sphere()