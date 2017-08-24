"""Visualize harmonics on an irregular simpicial grid

This demonstrates the simplex picking functionality
"""


import matplotlib.pyplot as plt
from pycomplex import synthetic
from examples.harmonics import *
from pycomplex.math import linalg


# sphere = synthetic.optimal_delaunay_sphere(100, 3, weights=False)
sphere = synthetic.icosphere(refinement=4).copy(radius=30)
sphere = sphere.optimize_weights_metric()

print(sphere.is_well_centered)
print(sphere.is_pairwise_delaunay)

from examples.diffusion.perlin_noise import perlin_noise

p0 = perlin_noise(
    sphere,
    [
        (.5, .5),
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
    ]
)

# # generate some interesting pattern
# v = get_harmonics_0(quad)
# f0 = v[:, 8]
# vmin, vmax = f0.min(), f0.max()

# pick a rectangular grid of values
N = 1024
points = np.moveaxis(np.indices((N, N)), 0, -1) / (N - 1) * 2 - 1
z = np.sqrt(np.clip(1 - linalg.dot(points, points), 0, 1))
points = np.concatenate([points, z[..., None]], axis=-1)


# if False:
#     # primal pick compared to primal plot
#     quad.plot_primal_0_form(f0, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
#     quad.plot(ax=plt.gca())
#     # plt.show()
#
#
#     tri_idx, bary = quad.pick_primal(points.reshape(-1, 2))
#     plt.figure()
#     plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))
#     plt.figure()
#     plt.imshow(tri_idx.reshape(N, N))
#
#     plt.figure()
#     img = quad.sample_primal_0(f0, points.reshape(-1, 2)).reshape(N, N)
#     plt.imshow(img.T[::-1])
#
#
#     plt.show()


if True:
    # # interpolate primal 0 form to test dual 0-form plotting; easier than implementing harmonics with BCs's
    d0 = sphere.topology.averaging_operators_0[2] * p0
    # sphere.as_euclidian().plot_primal_2_form(d0)
    # plt.show()


    sphere.as_euclidian().plot_dual_0_form_interpolated(d0, weighted=True, plot_contour=False, shading='gouraud')#, vmin=vmin, vmax=vmax)
    # quad.plot_dual_0_form_interpolated(d0, weighted=False, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    # quad.plot()#ax=plt.gca())

    domain_idx, bary, domain = sphere.pick_fundamental(points.reshape(-1, 3))

    plt.figure()
    plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))

    plt.figure()
    img = sphere.sample_dual_0(d0, points.reshape(-1, 3)).reshape(N, N)
    plt.imshow(img.T[::-1])


    plt.show()
