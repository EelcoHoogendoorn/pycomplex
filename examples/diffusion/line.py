"""Diffusion on the real number line

This showcases the sigma parameter on the diffusor; and that it works as intended

"""

import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from examples.diffusion.explicit import Diffusor


line = synthetic.n_cube_grid([1000]).as_11().as_regular()

f0 = line.topology.chain(0)
f0[500] = 1
d = Diffusor(line)
sigma = 30
f0 = d.integrate_explicit_sigma(f0, sigma=sigma)

# plot the result
fig, ax = plt.subplots(1, 1)
line.plot_primal_0_form(f0 / f0.max(), ax, color='r')
# plot a reference line to compare to; note that numerical solution is slightly over-smoothed
r0 = np.exp(-line.vertices[:,0] ** 2 / sigma**2 / 2)
line.plot_primal_0_form(r0, ax, color='g')
plt.show()
