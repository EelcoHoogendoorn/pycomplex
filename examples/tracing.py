"""Ray casting over the hexacosichoron.

I added this to stress-test complex.primal_pick, and out of general curiosity.
It is not every day that you get to visualize curved spaces from within.

It is interesting to note that all great circles appear as straight lines,
and all spherical tetrahedra look perfectly flat. Looking down upon a 2d spherical triangle
you can see it isnt flat, but you are not getting any of that sense with these similarly curved tetrahedra,
since the rays that are going the tracing are in on the joke, and doing the exact same curving as the 'straight' edges do.

There are two things that I can tell that give away that this isnt a euclidian space:
First of all, objects at the opposite pole of the universe appear enlarged,
as a consequence of the rays being traced converging there again after traversing a half sphere.
And also, a tiling with regular tetrahedra of euclidian space is impossible,
so whatever it is we are looking at here isnt that. Still, this isnt something that jumps out at you.

"""

import numpy as np
from pycomplex import synthetic
from pycomplex.math import linalg
import time


def render_frame(p, x, y, z):
    # stepping of each ray is a rotation matrix
    dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1./90) for d in [x, y, z]]

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

    t = time.clock()
    for distance in np.arange(0, max_distance, stepsize):
        # print(distance)
        # this is simply the stepping operation
        ray = np.einsum('...ij,...j->...i', ray_step, ray)
        simplex, bary = space.pick_primal_alt(ray.reshape(-1, 4), simplex=simplex)  # caching simplex makes a huge speed difference!
        bary = bary.reshape(resolution + (4,))
        # try and see if we hit an edge
        edge_hit = (bary < 0.01).sum(axis=2) >= 2
        draw = np.logical_and(edge_hit, pick==-1)
        pick[draw] = simplex.reshape(resolution)[draw]
        depth[draw] = distance
    print(time.clock() - t)

    # plot the result
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(1, 1)

    cmap = plt.get_cmap('hsv')
    colors = ScalarMappable(cmap=cmap).to_rgba(pick)
    colors[pick==-1, :3] = 0
    alpha = np.exp(-depth / 100)
    colors[:, :, :3] *= alpha[:, :, None]


    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # from matplotlib.figure import Figure

    # fig = plt.Figure(figsize=colors.shape[:-1], dpi=1, frameon=False)
    # canvas = FigureCanvas(fig)
    # fig = plt.Figure(figsize=(2, 2), dpi=300)
    fig.figimage(colors, origin='upper')
    shape = np.array(colors.shape[:-1])
    dpi = fig.get_dpi()
    fig.set_size_inches((shape / dpi) + 0.1, forward=True)

    # fig.tight_layout()

    # plt.imshow(colors)
    plt.axis('off')


if __name__ == '__main__':
    space = synthetic.hexacosichoron()

    stepsize = .2  # ray step size in degrees
    max_distance = 180  # trace rays half around the universe
    fov = 1  # higher values give a wider field of view
    resolution = (512, 512)  # in pixels

    from pycomplex.util import save_animation
    path = r'c:\development\examples\hexacosichoron_13'

    # generate a random starting position and orientation
    coordinate = linalg.orthonormalize(np.random.randn(4, 4))
    p, x, y, z = coordinate
    dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1. / 90) for d in [x, y, z]]

    for i in save_animation(path, frames=360):
        # make a step forward along the z axis
        coordinate = np.einsum('...ij,...j->...i', dz, coordinate)
        render_frame(*coordinate)
