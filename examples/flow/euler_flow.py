"""
Euler equations:
    Dω / dt = 0     (material derivative of vorticity is zero; or constant along streamlines)
    div(u) = 0      (incompressible)
    ω = curl(u)

This example is an implementation of the excellent paper 'Stable, Circulation-Preserving, Simplicial Fluids'
We depart from it in some aspects; primarily, the use of subdomain-interpolation rather than barycentric interpolation.
This should have as advantage that it is easier to implement efficiently in python, and leads to a simpler formulation at the boundary.
Moreover, we seek to extend the method to cubical grids too, not just simplicial ones.

References
----------
[1] http://www.geometry.caltech.edu/pubs/ETKSD07.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

from pycomplex import synthetic
from pycomplex.math import linalg
from pycomplex.topology import sparse_to_elements

sphere = synthetic.icosphere(refinement=3)
dt = 1

if False:
    sphere.plot()

# get initial conditions; solve for incompressible irrotational field
from examples.harmonics import get_harmonics_1, get_harmonics_0
# H = get_harmonics_1(sphere)[:, 0]
# or maybe incompressible but with rotations?
H = get_harmonics_0(sphere)[:, -2]

from examples.diffusion.explicit import Diffusor
H = Diffusor(sphere).integrate_explicit_sigma(np.random.randn(sphere.topology.n_elements[0]), 0.2)
H /= 30


def advect_vorticity_precompute(sphere):
    T01, T12 = sphere.topology.matrices

    dual_edge_vector = T12 * sphere.dual_position[0]
    # there exists a 3-vector at the dual vertex,
    # which projected on the dual edges forms the dual fluxes. on a sphere the third component is not determined,
    # dirs.dot(v) = tangent_edge_flux
    B = sphere.topology._boundary[-1]
    O = sphere.topology._orientation[-1]
    # tangent edges per primal n-element
    tangent_edges = linalg.normalized(dual_edge_vector)[B] * O[..., None]
    # compute pseudoinverse, to quickly construct velocities at dual vertices
    u, s, v = np.linalg.svd(tangent_edges)
    s = 1 / s
    s[:, 2] = 0
    pinv = np.einsum('...ij,...j,...jk', u, s, v)

    # operators to average something defined on a dual vertex to all dual elements
    average_1 = np.abs(T12) / 2
    q = np.abs(T01) * np.abs(T12)
    average_2 = scipy.sparse.diags(1. / np.array(q.sum(axis=1)).flatten()) * q
    dual_averages = [1, average_1, average_2]

    def dual_flux_to_dual_velocity(flux_d1):
        tangent_flux = (flux_d1 / sphere.dual_metric[1])[B] * O
        velocity_d0 = np.einsum('...ji,...j->...i', pinv, tangent_flux)
        return velocity_d0

    return dual_flux_to_dual_velocity, dual_averages


def advect_vorticity(sphere, flux_p1, dt):
    T01, T12 = sphere.topology.matrices
    flux_d1 = sphere.hodge_DP[1] * flux_p1

    to_dual_velocity, averages = advect_vorticity_precompute(sphere)

    velocity_d0 = to_dual_velocity(flux_d1)
    print(velocity_d0.max())

    # extend dual 0 form to all other dual elements by averaging
    velocity_dual = [a * velocity_d0 for a in averages]

    # advect the dual mesh
    advected_d0 = sphere.dual_position[0] + velocity_d0 * dt
    advected_d0 = linalg.normalized(advected_d0)


    # sample at all advected dual vertices, average at the mid of dual edge, and dot with advected dual edge vector

    domain, bary = sphere.pick_fundamental(advected_d0)
    # do interpolation over fundamental domain
    velocity_sampled_d0 = sum([velocity_dual[::-1][i][domain[:, i]] * bary[:, [i]] for i in range(sphere.n_dim)])
    # integrate the tangent flux of the advected mesh
    velocity_sampled_d1 = averages[1] * velocity_sampled_d0

    advected_edge = T12 * advected_d0
    flux_advected = linalg.dot(velocity_sampled_d1, advected_edge)

    # do streamfunction solve to recover incompressible component of advected flux

    # advected vorticity is the curl of advected flux
    vorticity_advected = T01 * flux_advected
    import scipy.sparse
    laplacian = T01 * (scipy.sparse.diags(sphere.hodge_DP[1]) * T01.T)
    phi = scipy.sparse.linalg.minres(laplacian, vorticity_advected, tol=1e-12)[0]

    advected_projected_flux = T01.T * phi
    return sphere.hodge_PD[0] * vorticity_advected, advected_projected_flux, phi


T01, T12 = sphere.topology.matrices
curl = T01.T
flux_p1 = curl * H
# flux_d1 = sphere.hodge_DP[1] * flux_p1
# old_vorticity = T01 * flux_d1

phi = H

while (True):
    vorticity, flux_p1, phi = advect_vorticity(sphere, flux_p1, dt=1)
    sphere.as_euclidian().as_3().plot_primal_0_form(phi, plot_contour=True, cmap='jet',vmin=-4e-2, vmax=+4e-2)
    # sphere.as_euclidian().as_3().plot_primal_0_form(vorticity, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-1e-0, vmax=+1e-0)
    plt.show()
# solve for a new flow field

# potentially add BFECC step
# do momentum diffusion, if desired

