"""Example script to compute and visualize spherical harmonics

This is a simple sanity check of a complex and its metrics
"""

import pycomplex.synthetic
from examples.harmonics import get_harmonics_0, get_harmonics_2


if __name__=='__main__':
    # construct a spherical complex
    sphere = pycomplex.synthetic.icosphere(refinement=4)
    sphere.metric()

    v = get_harmonics_0(sphere)
    # plot a spherical harmonic
    sphere.as_euclidian().plot_primal_0_form(v[:, -1])

    assert sphere.topology.is_oriented
    v = get_harmonics_2(sphere)
    # plot a spherical harmonic
    sphere.as_euclidian().plot_primal_2_form(v[:, -1])