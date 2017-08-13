"""Harmonics on a real number line

Indeed reproduces all the harmonic functions on the real number line.
Nothing too exciting, but added this to test some aspects of the nd-generalizations of this library

"""

import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from examples.harmonics import get_harmonics_0


line = synthetic.n_cube_grid([1000], centering=False).as_11().as_regular()

if True:
    # demonstrate that our sinusoids are not sensitive to vertex distribution
    line = line.copy(vertices=np.power(line.vertices + 1, 0.8))

v = get_harmonics_0(line)

line.plot_primal_0_form(v[:, 10])

plt.show()