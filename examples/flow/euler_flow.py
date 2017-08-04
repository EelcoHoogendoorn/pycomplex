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

if False:
    sphere.as_euclidian().as_3().plot_primal_0_form(H, plot_contour=False, cmap='bwr', shading='gouraud')
    plt.show()

def advect_vorticity(sphere, flux_p1, dt):
    T01, T12 = sphere.topology.matrices
    curl = T01.T
    # interpolate flux to primal-2-form
    flux_d1 = sphere.hodge_DP[1] * flux_p1
    # compute dual edge vectors
    de = T12 * sphere.dual_position[0]
    de_n = de / linalg.dot(de, de)[:, None] # normalized dual edge vectors
    # de_n = linalg.normalized(de)
    # now there should be a solving step; there exists a 3-vector at the dual vertex,
    # which projected on the dual edges forms the dual fluxes. on a sphere the third component is not determined,
    # dirs.dot(v) = tangent_edge_flux
    B = sphere.topology._boundary[-1]
    O = sphere.topology._orientation[-1]
    tangent_edge = de_n[B] * O[..., None]

    u, s, v = np.linalg.svd(tangent_edge)
    s = 1 / s
    s[:, 2] = 0
    pinv = np.einsum('...ij,...j,...jk', u, s, v)

    tangent_flux = flux_d1[B] * O
    velocity_d0 = np.einsum('...ij,...j->...i', pinv, tangent_flux)

    # A, B = sparse_to_elements(T01), sparse_to_elements(T12)
    # cascade down from flux at primal simplex to lower forms
    velocity_d1 = np.abs(T12) * velocity_d0 / 2  # mean over vertices of dual edge
    T0N = sphere.topology.matrix(-1, 0)
    velocity_d2 = (T0N * velocity_d1) / sphere.topology.vertex_degree()[:, None]  # mean over vertices of dual face
    velocity = [velocity_d2, velocity_d1, velocity_d0]

    # advect the dual mesh
    advected_d0 = sphere.dual_position[0] + velocity_d0 * dt

    # integrate the tangent flux of the advected mesh
    advected_d1 = np.abs(T12) * advected_d0 / 2 # mean over dual edges

    # sample at the midpoints of all advected dual edges, multiplied with advected dual length

    de = T12 * advected_d0
    domain, bary = sphere.pick_fundamental(advected_d1)
    # do interpolation over fundamental domain
    samples = sum([velocity[i][domain[:, i]] * bary[:, [i]] for i in range(sphere.n_dim)])
    samples = linalg.dot(samples, de)

    # advected vorticity is the curl of this
    advected_vorticity = T01 * samples
    return advected_vorticity


T01, T12 = sphere.topology.matrices
curl = T01.T
flux_p1 = curl * H
old_vorticity = T01 * flux_p1
new_vorticity = advect_vorticity(sphere, flux_p1, dt=dt)

# solve for a new flow field
# potentially add BFECC step
# do momentum diffusion, if desired

