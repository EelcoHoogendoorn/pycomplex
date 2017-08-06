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
from cached_property import cached_property

from pycomplex import synthetic
from pycomplex.math import linalg
from pycomplex.topology import sparse_to_elements


class Advector(object):

    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def advect_vorticity_precompute(self):
        T01, T12 = self.complex.topology.matrices
        D1D0 = T12

        dual_edge_vector = D1D0 * self.complex.dual_position[0]
        # for incompressible flows on simplicial topologies, there exists a 3-vector at the dual vertex,
        # which projected on the dual edges forms the dual fluxes. on a sphere the third component is not determined
        # approximate inverse would still make sense in cubical topology however
        # dirs.dot(v) = tangent_edge_flux
        B = self.complex.topology._boundary[-1]
        O = self.complex.topology._orientation[-1]
        # tangent edges per primal n-element
        tangent_edges = linalg.normalized(dual_edge_vector)[B] * O[..., None]
        # compute pseudoinverse, to quickly construct velocities at dual vertices
        u, s, v = np.linalg.svd(tangent_edges)
        s = 1 / s
        s[:, self.complex.topology.n_dim:] = 0
        pinv = np.einsum('...ij,...j,...jk', u, s, v)

        def dual_flux_to_dual_velocity(flux_d1):
            # compute velocity component in the direction of the dual edge
            tangent_flux = (flux_d1 / self.complex.dual_metric[1])[B] * O
            # given these flows incident on the dual vertex, reconstruct the velocity vector there
            velocity_d0 = np.einsum('...ji,...j->...i', pinv, tangent_flux)
            return velocity_d0

        return dual_flux_to_dual_velocity, self.complex.topology.dual.averaging_operators()

    def sample_dual_0(self, d0, points):
        # FIXME: make this a method on Complex?
        _, dual_averages = self.advect_vorticity_precompute
        # extend dual 0 form to all other dual elements by averaging
        velocity_dual = [a * d0 for a in dual_averages]
        domain, bary = self.complex.pick_fundamental(points)
        # do interpolation over fundamental domain
        return sum([velocity_dual[::-1][i][domain[:, i]] * bary[:, [i]]
                    for i in range(self.complex.topology.n_dim + 1)])
    def sample_primal_0(self, p0, points):
        element, bary = self.complex.pick_primal(points)
        IN0 = self.complex.topology.incidence[-1, 0]
        verts = IN0[element]
        return (p0[verts] * bary[:, None, :]).sum(axis=1)

    @cached_property
    def constrain_divergence_precompute(self):
        import scipy.sparse
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        D1P1 = scipy.sparse.diags(sphere.hodge_DP[1])
        # FIXME: this only works in the 2d case
        laplacian = D2D1 * D1P1 * P1P0
        return laplacian

    def constrain_divergence(self, flux_d1):
        # do streamfunction solve to recover incompressible component of (advected) flux
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        P0D2 = sphere.hodge_PD[0]

        vorticity_d2 = D2D1 * flux_d1
        laplacian = self.constrain_divergence_precompute
        phi = scipy.sparse.linalg.minres(laplacian, vorticity_d2, tol=1e-12)[0]

        return P0D2 * vorticity_d2, P1P0 * phi, phi

    def advect_vorticity(self, flux_p1, dt):
        """The main method of vorticity advection"""
        T01, T12 = self.complex.topology.matrices
        D1D0 = T12
        flux_d1 = self.complex.hodge_DP[-2] * flux_p1

        to_dual_velocity, dual_averages = self.advect_vorticity_precompute

        velocity_d0 = to_dual_velocity(flux_d1)
        print(velocity_d0.max())

        # advect the dual mesh
        advected_d0 = self.complex.dual_position[0] + velocity_d0 * dt
        advected_d0 = linalg.normalized(advected_d0)    # FIXME: this line is specific to working on a spherical complex!

        # sample at all advected dual vertices, average at the mid of dual edge, and dot with advected dual edge vector
        velocity_sampled_d0 = self.sample_dual_0(velocity_d0, advected_d0)

        # integrate the tangent flux of the advected mesh
        velocity_sampled_d1 = dual_averages[1] * velocity_sampled_d0
        advected_edge = D1D0 * advected_d0
        flux_d1_advected = linalg.dot(velocity_sampled_d1, advected_edge)

        vorticity_p0, projected_flux_p1, phi_p0 = self.constrain_divergence(flux_d1_advected)
        return vorticity_p0, projected_flux_p1, phi_p0


if __name__ == "__main__":
    sphere = synthetic.icosphere(refinement=4)
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

    T01, T12 = sphere.topology.matrices
    curl = T01.T
    flux_p1 = curl * H
    # flux_d1 = sphere.hodge_DP[1] * flux_p1
    # old_vorticity = T01 * flux_d1

    phi_p0 = H
    advector = Advector(sphere)
    while (True):
        for i in range(10):
            vorticity_p0, flux_p1, phi_p0 = advector.advect_vorticity(flux_p1, dt=1)
        # sphere.as_euclidian().as_3().plot_primal_0_form(phi_p0, plot_contour=True, cmap='jet', vmin=-4e-2, vmax=+4e-2)
        sphere.as_euclidian().as_3().plot_primal_0_form(
            vorticity_p0, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-5e-1, vmax=+5e-1)
        plt.show()

    # potentially add BFECC step
    # do momentum diffusion, if desired

