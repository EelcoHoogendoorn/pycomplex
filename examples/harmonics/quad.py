"""Harmonics of a square"""

import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from examples.harmonics import get_harmonics_0, get_harmonics_2


if False:
    quad = synthetic.n_cube(2)
    for i in range(5):
        quad = quad.subdivide()
else:
    quad = synthetic.n_cube_grid((32, 32), False)

quad = quad.as_22().as_regular()


v = get_harmonics_0(quad)
f0 = v[:, 8]
# do plotting via mapping to simplicial complex
tris = quad.subdivide_simplicial()
f0 = tris.topology.transfer_operators[0] * f0
tris.as_2().plot_primal_0_form(f0, plot_contour=False)

v = get_harmonics_2(quad)
f2 = v[:, 12]
# do plotting via mapping to simplicial complex
tris = quad.subdivide_simplicial()
f2 = tris.topology.transfer_operators[2] * f2
tris.as_2().plot_dual_0_form_interpolated(f2, weighted=False, plot_contour=False, shading='gouraud')
tris.as_2().plot_primal_2_form(f2)

plt.show()