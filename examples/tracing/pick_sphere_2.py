"""This demonstrates the spherical simplex picking functionality
"""


import matplotlib.pyplot as plt
from pycomplex import synthetic
from examples.harmonics import *
from pycomplex.math import linalg


# sphere = synthetic.optimal_delaunay_sphere(100, 3, weights=False)
sphere = synthetic.icosphere(refinement=2).copy(radius=30)
# sphere = sphere.optimize_weights_metric()

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

vmin, vmax = p0.min(), p0.max()
print(vmin, vmax)

# pick a rectangular grid of values
N = 128
points = np.moveaxis(np.indices((N, N)), 0, -1) / (N - 1) * 2 - 1
z = np.sqrt(np.clip(1 - linalg.dot(points, points), 0, 1))
points = np.concatenate([points, z[..., None]], axis=-1)


if False:
    # primal pick compared to primal plot

    sphere.as_euclidian().plot_primal_0_form(p0, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    sphere.plot(ax=plt.gca())
    # plt.show()

    tri_idx, bary = sphere.pick_primal(points.reshape(-1, 3))
    plt.figure()
    plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))
    plt.figure()
    plt.imshow(tri_idx.reshape(N, N))

    plt.figure()
    img = sphere.sample_primal_0(p0, points.reshape(-1, 3)).reshape(N, N)
    plt.imshow(img.T[::-1])

    plt.show()


if True:
    # # interpolate primal 0 form to test dual 0-form plotting
    d0 = sphere.sample_primal_0(p0, sphere.dual_position[0])

    sphere.as_euclidian().plot_primal_2_form(d0)
    plt.show()

    sphere.as_euclidian().plot_dual_0_form_interpolated(d0, weighted=True, plot_contour=False, shading='gouraud')#, vmin=vmin, vmax=vmax)
    # quad.plot()#ax=plt.gca())

    domain_idx, bary, domain = sphere.pick_fundamental(points.reshape(-1, 3))
    plt.hist(bary.flatten(), bins=50)
    plt.show()

    plt.figure()
    img = np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0)
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)

    plt.figure()
    img = sphere.sample_dual_0(d0, points.reshape(-1, 3)).reshape(N, N).T[::-1]
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)


    plt.show()
