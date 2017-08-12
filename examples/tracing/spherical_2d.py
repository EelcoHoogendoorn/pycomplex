"""2d spherical raytracer, constructed as visuale analogue of the 3d case"""

import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.math import linalg


space = synthetic.icosahedron()

coordinate = linalg.orthonormalize(np.random.randn(3, 3))
p, x, y = coordinate
dx, dy = [linalg.power(linalg.rotation_from_plane(p, d), 1. / 90) for d in [x, y]]

stepsize = 2  # ray step size in degrees

fov = 1  # higher values give a wider field of view
resolution = 30,
gx = np.linspace(-stepsize, +stepsize, num=resolution[0], endpoint=True)
ray_step = np.einsum('xij,jk->xik', linalg.power(dx, gx * fov), linalg.power(dy, +stepsize))
max_distance = 180  # trace rays half around the universe

space.plot()

ray = p
for distance in np.arange(0, max_distance, stepsize):
    ray = np.einsum('...ij,...j->...i', ray_step, ray)
    plt.scatter(*ray[:, :2].T)

plt.show()
