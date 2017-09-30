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

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from pycomplex import synthetic
from pycomplex.math import linalg

from examples.util import save_animation


def render_frame(p, x, y, z, plot_primal=False):
    # stepping of each ray is a rotation matrix
    dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1./90) for d in [x, y, z]]

    # build up projection plane
    gx = np.linspace(-stepsize, +stepsize, num=resolution[0], endpoint=True)
    gy = np.linspace(-stepsize, +stepsize, num=resolution[1], endpoint=True)
    # normalize step magnitude, so distance coloring is more correct?
    ray_step = np.einsum('xij,yjk,kl->xyil', linalg.power(dx, gx * fov), linalg.power(dy, gy * fov), linalg.power(dz, +stepsize))
    # normalize stepsizes
    s = np.rad2deg(linalg.angle_from_rotation(ray_step))
    ray_step = linalg.power(ray_step, stepsize / s)

    # all rays start from p
    ray = p
    simplex_idx = None
    domain_idx = None

    # accumulate what simplex we hit, and how far away
    pick = -np.ones(resolution, np.int16)
    depth = np.zeros(resolution, np.float)

    t = time.clock()
    for distance in np.arange(0, max_distance, stepsize):
        print(distance)
        # print(distance)
        # this is simply the stepping operation
        ray = np.einsum('...ij,...j->...i', ray_step, ray)
        if plot_primal:
            # visualize primal edges
            simplex_idx, bary = space.pick_primal(ray.reshape(-1, 4), simplex_idx=simplex_idx)  # caching simplex makes a huge speed difference!
            max_idx = space.topology.n_elements[-1]

            bary = bary.reshape(resolution + (4,))
            # try and see if we hit an edge
            edge_hit = (bary < 0.01).sum(axis=-1) >= 2
        else:
            # visualize dual edges instead
            domain_idx, bary, domains = space.pick_fundamental(ray.reshape(-1, 4), domain_idx=domain_idx)
            simplex_idx = domains[:, 0]
            max_idx = space.topology.n_elements[0]
            bary = bary.reshape(resolution + (4,))
            edge_hit = bary[..., :-2].sum(axis=-1) <= 0.04 # last two baries are dual cell center and its boundary

        draw = np.logical_and(edge_hit, pick==-1)
        pick[draw] = simplex_idx.reshape(resolution)[draw]
        depth[draw] = distance
    print(time.clock() - t)

    # plot the result
    fig, ax = plt.subplots(1, 1)
    cmap = plt.get_cmap('hsv')
    colors = ScalarMappable(cmap=cmap).to_rgba(pick / max_idx, norm=False)
    colors[pick==-1, :3] = 0
    alpha = np.exp(-depth / 150)
    colors[:, :, :3] *= alpha[:, :, None]

    fig.figimage(colors, origin='upper')
    shape = np.array(colors.shape[:-1])
    dpi = fig.get_dpi()
    fig.set_size_inches((shape / dpi) + 0.1, forward=True)

    plt.axis('off')


if __name__ == '__main__':

    if False:
        # visualize a randomly tesselated space
        space = synthetic.optimal_delaunay_sphere(n_dim=4, n_points=200, iterations=20)
        space = space.optimize_weights()

        stepsize = .2  # ray step size in degrees
        max_distance = 180  # trace rays all around the universe
        fov = 1  # higher values give a wider field of view
        resolution = (512, 512)  # in pixels

        # generate a random starting position and orientation
        coordinate = linalg.orthonormalize(np.random.randn(4, 4))
        p, x, y, z = coordinate
        dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1. / 90) for d in [x, y, z]]

        coordinate = np.einsum('...ij,...j->...i', dz, coordinate)
        render_frame(*coordinate)
        plt.show()
        quit()

    elif False:
        path = r'../output/random_sphere_5'
        space = synthetic.optimal_delaunay_sphere(n_dim=4, n_points=300, iterations=50)
        space = space.optimize_weights()

        stepsize = .2  # ray step size in degrees
        max_distance = 360  # trace rays half around the universe
        fov = 1  # higher values give a wider field of view
        resolution = (512, 512)  # in pixels

        # generate a random starting position and orientation
        coordinate = linalg.orthonormalize(np.random.randn(4, 4))
        p, x, y, z = coordinate
        dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1. / 90) for d in [x, y, z]]

        for i in save_animation(path, frames=360):
            # make a step forward along the z axis
            coordinate = np.einsum('...ij,...j->...i', dz, coordinate)
            render_frame(*coordinate)

    else:
        space = synthetic.hexacosichoron()
        space = space.copy(vertices=linalg.normalized(space.vertices + np.random.normal(scale=0.03, size=space.vertices.shape)))
        space = space.optimize_weights()
        assert space.is_well_centered

        stepsize = .2  # ray step size in degrees
        max_distance = 180  # trace rays half around the universe
        fov = 1  # higher values give a wider field of view
        resolution = (256, 256)  # in pixels

        path = r'../output/hexacosichoron_0'

        # generate a random starting position and orientation
        coordinate = linalg.orthonormalize(np.random.randn(4, 4))
        p, x, y, z = coordinate
        dx, dy, dz = [linalg.power(linalg.rotation_from_plane(p, d), 1. / 90) for d in [x, y, z]]

        for i in save_animation(path, frames=360):
            # make a step forward along the z axis
            coordinate = np.einsum('...ij,...j->...i', dz, coordinate)
            render_frame(*coordinate, plot_primal=False)
