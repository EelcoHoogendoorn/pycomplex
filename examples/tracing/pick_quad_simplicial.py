"""Visualize harmonics on an irregular simpicial grid

This demonstrates the simplex picking functionality
"""


import matplotlib.pyplot as plt
from pycomplex import synthetic
from examples.harmonics import *

while True:
    complex = synthetic.delaunay_cube(8, 2, iterations=30)
    # smooth while holding boundary constant
    # FIXME: need more utility functions for this; too much boilerplate for such a simple pattern
    chain_0 = complex.topology.chain(0, fill=0)
    chain_0[complex.boundary.topology.parent_idx[0]] = 1
    chain_1 = complex.topology.chain(1, fill=0)
    chain_1[complex.boundary.topology.parent_idx[1]] = 1
    creases = {0: chain_0, 1: chain_1}
    for i in range(2):
        complex = complex.as_2().subdivide(smooth=True, creases=creases)
        for d, c in creases.items():
            creases[d] = complex.topology.transfer_matrices[d] * c
        complex = complex.optimize_weights()


    # complex = complex.optimize_weights_metric()
    if complex.is_well_centered:
        break
    print('retry')

complex = complex.as_2().as_2()
print(complex.is_well_centered)
print(complex.is_pairwise_delaunay)


# generate some interesting pattern
v = get_harmonics_0(complex)
p0 = v[:, 8]
vmin, vmax = p0.min(), p0.max()
print('computed harmonic')

# pick a rectangular grid of values
N = 1024
points = np.moveaxis(np.indices((N, N)), 0, -1) / (N - 1)

if True:
    # primal pick compared to primal plot
    # complex.plot_primal_0_form(f0, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    # complex.plot(ax=plt.gca())
    # plt.show()


    tri_idx, bary = complex.pick_primal(points.reshape(-1, 2))

    plt.figure()
    plt.hist(bary.flatten(), bins=1000)

    plt.figure()
    plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))
    plt.figure()
    plt.imshow(tri_idx.reshape(N, N))

    plt.figure()
    img = complex.sample_primal_0(p0, points.reshape(-1, 2)).reshape(N, N)
    plt.imshow(img.T[::-1])


    plt.show()


if True:
    # interpolate primal 0 form to test dual 0-form plotting
    d0 = complex.sample_primal_0(p0, complex.dual_position[0])


    complex.plot_dual_0_form_interpolated(d0, weighted=True, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    # quad.plot_dual_0_form_interpolated(d0, weighted=False, plot_contour=False, shading='gouraud', vmin=vmin, vmax=vmax)
    # quad.plot()#ax=plt.gca())

    domain_idx, bary, domain = complex.pick_fundamental(points.reshape(-1, 2))
    plt.figure()
    plt.hist(bary.flatten(), bins=1000)


    plt.figure()
    plt.imshow(np.flip(np.moveaxis(bary.reshape(N, N, 3), 0, 1), axis=0))

    plt.figure()
    img = complex.sample_dual_0(d0, points.reshape(-1, 2)).reshape(N, N)
    plt.imshow(img.T[::-1])


    plt.show()
