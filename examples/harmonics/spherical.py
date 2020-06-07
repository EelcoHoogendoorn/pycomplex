"""Example script to compute and visualize spherical harmonics

This is a simple sanity check of a complex and its metrics
"""

import matplotlib.pyplot as plt

import pycomplex.synthetic
from examples.harmonics import get_harmonics_0, get_harmonics_2


if __name__ == '__main__':
    # construct a spherical complex
    sphere = pycomplex.synthetic.icosphere(refinement=4)
    assert sphere.topology.is_oriented

    from time import clock
    t = clock()
    v = get_harmonics_0(sphere, amg=True, K=150, tol=1e-6)
    print(clock() - t)
    # plot a spherical harmonic
    sphere.as_euclidian().plot_primal_0_form(v[:, -1])

    if False:
        t = clock()
        v = get_harmonics_2(sphere, amg=True, K=250, tol=1e-6)
        print(clock() - t)
        # plot a spherical harmonic
        sphere.as_euclidian().plot_primal_2_form(v[:, -1])
        # sphere.as_euclidian().plot_dual_0_form_interpolated(v[:, -1])
        sphere.as_euclidian().plot_dual_0_form_interpolated(v[:, -1], weighted=False)

    plt.show()
