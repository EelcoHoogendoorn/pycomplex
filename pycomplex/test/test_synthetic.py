
from pycomplex import synthetic


def test_optimal_sphere():
    n_dim = 3
    complex = synthetic.optimal_delaunay_sphere(800, n_dim)
    complex.plot(backface_culling=n_dim==3, plot_vertices=False)
    import matplotlib.pyplot as plt
    plt.show()

test_optimal_sphere()