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

Notes
-----
mesh-edges are visible in the current implentation, during advection
is this our averaging barycentric interpolant being funny?
can we re-weight our averaging op such that it reconstructs all linear functions?
yeah; weighting should not be linear; but rather proportional to the bary weights of that element!
we can express this as a proportionality between the sum of fundamental-domain area incident to
both the dual n-element and the dual 0-element
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from cached_property import cached_property

from pycomplex import synthetic
from pycomplex.math import linalg
from pycomplex.util import save_animation
from pycomplex.complex.spherical import ComplexSpherical
from examples.advection import Advector


class VorticityAdvector(Advector):

    @cached_property
    def constrain_divergence_precompute(self):
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        D1P1 = scipy.sparse.diags(sphere.hodge_DP[1])
        # FIXME: this only works in the 2d case
        laplacian = D2D1 * D1P1 * P1P0
        return laplacian

    @cached_property
    def pressure_projection_precompute(self):
        TnN = self.complex.topology.matrices[-1]
        hodge = scipy.sparse.diags(sphere.hodge_PD[-2])
        laplacian = TnN.T * hodge * TnN
        return laplacian

    def pressure_projection(self, flux_d1):
        TnN = self.complex.topology.matrices[-1]
        hodge = sphere.hodge_PD[-2]

        div = TnN.T * (hodge * flux_d1)
        laplacian = self.pressure_projection_precompute
        P_d0 = scipy.sparse.linalg.minres(laplacian, div, tol=1e-12)[0]

        return flux_d1 - TnN * P_d0

    def constrain_divergence(self, flux_d1):
        # do streamfunction solve to recover incompressible component of (advected) flux
        # why not just use pressure projection? need vorticity form anyway if we seek to do diffusion?
        # otherwise we can do full time-dependent stokes; best for generality
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        D1P1 = sphere.hodge_DP[1]
        P0D2 = sphere.hodge_PD[0]

        vorticity_d2 = D2D1 * flux_d1
        laplacian = self.constrain_divergence_precompute
        phi_p0 = scipy.sparse.linalg.minres(laplacian, vorticity_d2, tol=1e-12)[0]

        return D1P1 * (P1P0 * phi_p0)

    def advect_vorticity(self, flux_d1, dt):
        """The main method of vorticity advection"""
        T01, T12 = self.complex.topology.matrices
        D1D0 = T12

        velocity_d0_inner = self.dual_flux_to_dual_velocity(flux_d1)
        # implicitly set boundary velocities to all zero
        velocity_d0 = np.zeros((self.complex.topology.dual.n_elements[0], velocity_d0_inner.shape[1]))
        velocity_d0[:len(velocity_d0_inner)] = velocity_d0_inner

        # advect the dual mesh
        advected_d0 = self.complex.dual_position[0] + velocity_d0 * dt
        if isinstance(self.complex, ComplexSpherical):
            advected_d0 = linalg.normalized(advected_d0)    # FIXME: this line is specific to working on a spherical complex!

        # sample at all advected dual vertices, average at the mid of dual edge, and dot with advected dual edge vector
        velocity_sampled_d0 = self.complex.sample_dual_0(velocity_d0, advected_d0)

        # integrate the tangent flux of the advected mesh
        velocity_sampled_d1 = self.complex.cached_averages[1] * velocity_sampled_d0
        advected_edge = D1D0 * advected_d0
        flux_d1_advected = linalg.dot(velocity_sampled_d1, advected_edge)

        return self.pressure_projection(flux_d1_advected)
        # return self.constrain_divergence(flux_d1_advected)


if __name__ == "__main__":
    sphere = synthetic.icosphere(refinement=4)
    dt = 1

    if False:
        sphere.plot(plot_vertices=False, plot_dual=True)
        # quit
    # # get initial conditions; solve for incompressible irrotational field
    from examples.harmonics import get_harmonics_1, get_harmonics_0
    # # H = get_harmonics_1(sphere)[:, 1]
    # or maybe incompressible but with rotations?
    H = get_harmonics_0(sphere)[:, 2]
    #
    # from examples.diffusion.explicit import Diffusor
    # F = Diffusor(sphere).integrate_explicit_sigma(np.random.randn(sphere.topology.n_elements[0]), 0.2)
    # F /= 30
    # F = F + H

    from examples.diffusion.planet_perlin import perlin_noise
    H = perlin_noise(
        sphere,
        [
            # (.05, .05),
            (.1, .1),
            (.2, .2),
            (.4, .4),
            (.8, .8),
        ]
    ) / 100 + H * 8


    T01, T12 = sphere.topology.matrices
    curl = T01.T
    flux_p1 = curl * H
    flux_d1 = sphere.hodge_DP[1] * flux_p1

    # phi_p0 = H
    advector = VorticityAdvector(sphere)

    # test that integrating over zero time does almost nothing
    advected_0 = advector.advect_vorticity(flux_d1, dt=0)
    print(np.abs(advected_0 - flux_d1).max())
    print(np.abs(flux_d1).max())
    assert np.allclose(advected_0, flux_d1, atol=1e-6)

    path = r'c:\development\examples\euler_30'
    # path = None
    def advect(flux_d1, dt):
        return advector.advect_vorticity(flux_d1, dt)

    from examples.advection import MacCormack, BFECC

    for i in save_animation(path, frames=200):
        for r in range(4):
            # flux_d1 = BFECC(advect, flux_d1, dt=2)
            flux_d1 = advect(flux_d1, dt=2)
        # sphere.as_euclidian().as_3().plot_primal_0_form(phi_p0, plot_contour=True, cmap='jet', vmin=-2e-2, vmax=+2e-2)

        vorticity_p0 = sphere.hodge_PD[0] * (T01 * flux_d1)

        sphere.as_euclidian().as_3().plot_primal_0_form(
            vorticity_p0, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)
        plt.axis('off')

    # do momentum diffusion, if desired

