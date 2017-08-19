"""Visualize harmonics on an irregular simpicial grid

This demonstrates the simplex picking functionality
"""


import matplotlib.pyplot as plt
from pycomplex import synthetic
from examples.harmonics import *

while True:
    quad = synthetic.delaunay_cube(8, 2, iterations=30)
    quad = quad.optimize_weights_metric()
    if quad.is_well_centered:
        break
quad = quad.as_2().as_2()
print(quad.is_well_centered)
print(quad.is_pairwise_delaunay)


# generate some interesting pattern
v = get_harmonics_0(quad)
f0 = v[:, 8]
vmin, vmax = f0.min(), f0.max()

# pick a rectangular grid of values
N = 128
points = np.moveaxis(np.indices((N, N)), 0, -1) / (N - 1)

if False:
    # primal pick compared to primal plot
    quad.plot_primal_0_form(f0, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    quad.plot(ax=plt.gca())
    # plt.show()


    tri_idx, bary = quad.pick_primal(points.reshape(-1, 2))
    plt.figure()
    plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))
    plt.figure()
    plt.imshow(tri_idx.reshape(N, N))

    plt.figure()
    img = quad.sample_primal_0(f0, points.reshape(-1, 2)).reshape(N, N)
    plt.imshow(img.T[::-1])


    plt.show()


if True:
    # interpolate primal 0 form to test dual 0-form plotting; easier than implementing harmonics with BCs's
    d0 = quad.topology.averaging_operators_0[2] * f0
    # quad.plot_primal_2_form(d0)
    # plt.show()
    db = quad.topology.averaging_operators_0[1] * f0
    d0 = np.concatenate([d0, db[quad.topology.boundary.parent_idx[1]]])


    # quad.plot_dual_0_form_interpolated(d0, weighted=True, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    # quad.plot_dual_0_form_interpolated(d0, weighted=False, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    # quad.plot()#ax=plt.gca())

    domain_idx, bary, domain = quad.pick_fundamental(points.reshape(-1, 2))

    plt.figure()
    plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))

    plt.figure()
    img = quad.sample_dual_0(d0, points.reshape(-1, 2)).reshape(N, N)
    plt.imshow(img.T[::-1])


    plt.show()
