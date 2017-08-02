"""Ray casting over the hexacosichoron.

I added this to stress-test complex.primal_pick, and out of general curiosity.
It is not every day that you get to visualize curved spaces from within.

It is interesting to note that all great circles appear as straight lines,
and all spherical tetrahedra look perfectly flat.
There are two things that I can tell that give away that this isnt a euclidian space.
First of all, objects at the opposite pole of the universe appear enlarged,
as a consequence of the rays being traced converging there again after traversing a half sphere.
And also, a tiling with regular tetrahedra of euclidian space is impossible,
so whatever it is we are looking at here isnt that.

"""

import numpy as np
from pycomplex import synthetic
from pycomplex.math import linalg

space = synthetic.hexacosichoron()

# generate a random starting position and orientation
coordinate = linalg.orthonormalize(np.random.randn(4, 4))
p, x, y, z = coordinate

# stepping of each ray is a rotation matrix
dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1./90) for d in [x, y, z]]

stepsize = .2       # ray step size in degrees
max_distance = 180  # go around half the universe
fov = 1             # higher values give a wider field of view
resolution = (256, 256)     # in pixels

# build up projection plane
gx = np.linspace(-stepsize, +stepsize, num=resolution[0], endpoint=True)
gy = np.linspace(-stepsize, +stepsize, num=resolution[1], endpoint=True)
ray_step = np.einsum('xij,yjk,kl->xyil', linalg.power(dx, gx * fov), linalg.power(dy, gy * fov), linalg.power(dz, +stepsize))

# all rays start from p
ray = p
simplex = None

# accumulate what simplex we hit, and how far away
pick = -np.ones(resolution, np.int16)
depth = np.zeros(resolution, np.float)

for distance in np.arange(0, max_distance, stepsize):
    print(distance)
    # this is simply the stepping operation
    ray = np.einsum('...ij,...j->...i', ray_step, ray)
    simplex, bary = space.pick_primal(ray.reshape(-1, 4), simplex=simplex)  # caching simplex makes a huge speed difference!
    bary = bary.reshape(resolution + (4,))
    # try and see if we hit an edge
    edge_hit = (bary < 0.01).sum(axis=2) >= 2
    draw = np.logical_and(edge_hit, pick==-1)
    pick[draw] = simplex.reshape(resolution)[draw]
    depth[draw] = distance


# plot the result
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

fig, ax = plt.subplots(1, 1)

cmap = plt.get_cmap('hsv')
colors = ScalarMappable(cmap=cmap).to_rgba(pick)
colors[pick==-1, :3] = 0
alpha = np.exp(-depth / 100)
colors[:, :, :3] *= alpha[:, :, None]

plt.imshow(colors)
plt.show()
