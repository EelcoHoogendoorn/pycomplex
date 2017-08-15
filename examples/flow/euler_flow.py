"""
Euler equations:
    Dω / dt = 0     (material derivative of vorticity is zero; or constant along streamlines)
    div(u) = 0      (incompressible)
    ω = curl(u)

This example is an implementation of the excellent paper 'Stable, Circulation-Preserving, Simplicial Fluids'
We depart from it in some aspects; primarily, the use of subdomain-interpolation rather than barycentric interpolation.
This should have as advantage that it is easier to implement efficiently in python, and leads to a simpler formulation at the boundary.
Moreover, we extend the method to cubical grids too, not just simplicial ones.

References
----------
[1] http://www.geometry.caltech.edu/pubs/ETKSD07.pdf

Notes
-----
mesh-edges are visible in the current implementation, during advection over the sphere
not sure what is the cause of this; the euclidian approximations made on the sphere when integrating flux perhaps?
means it should respond posiively to grid refinement and it does not seem to
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from cached_property import cached_property

from examples.diffusion.explicit import Diffusor
from examples.flow.advection import Advector
from examples.util import save_animation
from pycomplex import synthetic
from pycomplex.complex.spherical import ComplexSpherical
from pycomplex.math import linalg


class VorticityAdvector(Advector):

    def __init__(self, complex, diffusion=None):
        super(VorticityAdvector, self).__init__(complex)
        self.diffusion = diffusion

    @cached_property
    def pressure_projection_precompute(self):
        # FIXME: this leaves all pressure boundary terms implicitly at zero. need control over bc's.
        TnN = self.complex.topology.matrices[-1]
        hodge = scipy.sparse.diags(self.complex.hodge_PD[-2])
        laplacian = TnN.T * hodge * TnN
        return laplacian

    def pressure_projection(self, flux_d1):
        TnN = self.complex.topology.matrices[-1]
        hodge = self.complex.hodge_PD[-2]

        div = TnN.T * (hodge * flux_d1)
        laplacian = self.pressure_projection_precompute
        P_d0 = scipy.sparse.linalg.minres(laplacian, div, tol=1e-12)[0]

        return flux_d1 - TnN * P_d0

    @cached_property
    def constrain_divergence_precompute(self):
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        D1P1 = scipy.sparse.diags(self.complex.hodge_DP[1])
        # FIXME: this only works in the 2d case
        laplacian = D2D1 * D1P1 * P1P0
        return laplacian

    def constrain_divergence(self, flux_d1):
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        D1P1 = self.complex.hodge_DP[1]

        vorticity_d2 = D2D1 * flux_d1
        laplacian = self.constrain_divergence_precompute
        phi_p0 = scipy.sparse.linalg.minres(laplacian, vorticity_d2, tol=1e-12)[0]

        return D1P1 * (P1P0 * phi_p0)

    @cached_property
    def constrain_divergence_precompute_boundary(self):
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T * self.complex.topology.selector[0].T  # this multiply pins all boundary streamfunction implicitly at zero, enforcing zero boundary flux
        D2D1 = P1P0.T
        D1P1 = scipy.sparse.diags(self.complex.hodge_DP[1])
        laplacian = D2D1 * D1P1 * P1P0
        return laplacian

    @cached_property
    def vorticity_diffusor(self):
        return Diffusor(self.complex)

    def constrain_divergence_boundary(self, flux_d1, return_phi=False):
        """

        Parameters
        ----------
        flux_d1 : ndarray, [n_dual_edges], float
            dual 1-form, excluding values on the dual boundary

        """
        T01, T12 = self.complex.topology.matrices
        P1P0 = T01.T
        D2D1 = T01
        D1P1 = self.complex.hodge_DP[1]
        D2P0 = self.complex.hodge_DP[0]
        P0D2 = self.complex.hodge_PD[0]

        vorticity_d2 = D2D1 * flux_d1
        if self.diffusion:
            vorticity_d2 = D2P0 * self.vorticity_diffusor.integrate_explicit(P0D2 * vorticity_d2, dt=self.diffusion)

        vorticity_d2 = self.complex.topology.selector[0] * vorticity_d2
        laplacian = self.constrain_divergence_precompute_boundary
        phi_p0 = scipy.sparse.linalg.minres(laplacian, vorticity_d2, tol=1e-14)[0]
        # add the boundary zeros back in
        phi_p0 = self.complex.topology.selector[0].T * phi_p0

        if return_phi:
            return phi_p0
        else:
            return D1P1 * (P1P0 * phi_p0)


    def time_dependent_stokes(self):
        """Do full time-dependent stokes step,
        to account for diffusion, project out divergent components, and include body forces.

        Much more principles approach than either vorticity or pressure based step only
        """

    def advect_vorticity(self, flux_d1, dt, force=None):
        """The main method of vorticity advection

        Parameters
        ----------
        flux_d1 : ndarray, [n_dual_edges], float
            dual 1-form, including values on the dual boundary
        dt : float
            timestep
        force : ndarray, [n_dual_edges], float, optional
            dual 1-form, including values on the dual boundary

        Returns
        ----------
        flux_d1 : ndarray, [n_dual_edges], float
            self-advected dual 1-form, including values on the dual boundary

        """
        D01, D12 = self.complex.topology.dual.matrices_2
        D1D0 = D01.T

        velocity_d0 = self.dual_flux_to_dual_velocity(flux_d1)

        # advect the dual mesh
        advected_d0 = self.complex.dual_position[0] + velocity_d0 * dt
        if isinstance(self.complex, ComplexSpherical):
            advected_d0 = linalg.normalized(advected_d0)

        # sample at all advected dual vertices, average at the mid of dual edge, and dot with advected dual edge vector
        velocity_sampled_d0 = self.complex.sample_dual_0(velocity_d0, advected_d0)

        # integrate the tangent flux of the advected mesh.
        # FIXME: average_dual here is overkill, but want to reuse boundary sampling
        velocity_sampled_d1 = self.complex.average_dual(velocity_sampled_d0)[1]
        # velocity_sampled_d1 = self.complex.cached_averages[1] * velocity_sampled_d0
        advected_edge = self.complex.topology.dual.matrices[0].T * advected_d0
        flux_d1_advected = linalg.dot(velocity_sampled_d1, advected_edge)        # this does not include flux around boundary edges; but currently all zero anyway
        # drop
        flux_d1_advected = self.complex.topology.dual.selector[1] * flux_d1_advected

        if force is not None:
            # add force impulse, if given
            flux_d1_advected += force * dt

        if self.complex.boundary:
            return self.complex.topology.dual.selector[1].T * self.constrain_divergence_boundary(flux_d1_advected)
        else:
            # on boundary-free domains, using pressure projection is simpler, since it does not
            return self.complex.topology.dual.selector[1].T * self.pressure_projection(flux_d1_advected)


if __name__ == "__main__":
    dt = .1

    complex_type = 'sphere'

    if complex_type == 'sphere':
        complex = synthetic.icosphere(refinement=5)

        # FIXME: for reasons not quite clear to me irregular mesh with same vert-count is a fair bit slower. probably the condition number?
        # complex = synthetic.optimal_delaunay_sphere(complex.topology.n_elements[0], 3, iterations=10)
        complex = complex.optimize_weights()


        if False:
            complex.plot(backface_culling=True)
            plt.show()

    if complex_type == 'grid':
        complex = synthetic.n_cube_grid((2, 1), False)
        for i in range(6):
            complex = complex.subdivide()

        complex = complex.as_22().as_regular()
        complex.topology.check_chain()
        tris = complex.to_simplicial()


    T01, T12 = complex.topology.matrices
    curl = T01.T
    D01, D12 = complex.topology.dual.matrices_2
    curl = T01.T
    D2D1 = D12.T

    from examples.harmonics import get_harmonics_0, get_harmonics_2

    if False:
        # generate a smooth incompressible flow field using harmonics

        # H = get_harmonics_0(complex)
        H_d0 = get_harmonics_2(complex)[:, 2]

        A = complex.topology.dual.averaging_operators_0()
        H_p0 = complex.hodge_PD[0] * (A[2] * H_d0)
        H_p0[complex.boundary.topology.parent_idx[0]] = 0

        if False:
            form = tris.topology.transfer_operators[0] * H_p0
            tris.as_2().plot_primal_0_form(form)
            plt.show()

        # form = tris.topology.transfer_operators[0] * H[:, 2]
        # tris.as_2().plot_dual_2_form_interpolated(
        #     form, plot_contour=False, cmap='terrain', shading='gouraud')
        # plt.show()

        flux_d1 = complex.hodge_DP[1] * (curl * (H_p0)) / 1000

    else:
        # use perlin noise for more chaotic flow pattern
        H = get_harmonics_0(complex)[:, 2]
        from examples.diffusion.perlin_noise import perlin_noise
        H = perlin_noise(
            complex,
            [
                (.05, .05),
                (.1, .1),
                (.2, .2),
                (.4, .4),
                (.8, .8),
            ]
        ) / 100 + H * 0

        flux_p1 = curl * H
        flux_d1 = complex.hodge_DP[1] * flux_p1

    # set boundary tangent flux to zero
    flux_d1 = complex.topology.dual.selector[1].T * flux_d1

    # set up vorticity advector
    advector = VorticityAdvector(complex)
    # test that integrating over zero time does almost nothing
    advected_0 = advector.advect_vorticity(flux_d1, dt=0)
    print(np.abs(advected_0 - flux_d1).max())
    print(np.abs(flux_d1).max())
    # assert np.allclose(advected_0, flux_d1, atol=1e-6)

    path = r'c:\development\examples\euler_44'
    # path = None
    def advect(flux_d1, dt):
        return advector.advect_vorticity(flux_d1, dt)

    from examples.flow.advection import BFECC

    for i in save_animation(path, frames=200, overwrite=True):
        for r in range(4):
            flux_d1 = BFECC(advect, flux_d1, dt=1)
            # flux_d1 = advect(flux_d1, dt=2)

        vorticity_p0 = complex.hodge_PD[0] * (D2D1 * flux_d1)
        if complex.boundary is not None:
            vorticity_p0[complex.boundary.topology.parent_idx[0]] = 0   # dont care about shear in boundary layer

        if complex_type == 'sphere':
            complex.as_euclidian().as_3().plot_primal_0_form(
                vorticity_p0, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)
        if complex_type == 'grid':
            if True:
                form = tris.topology.transfer_operators[0] * vorticity_p0
                tris.as_2().plot_primal_0_form(
                    form, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)
            else:
                phi_0 = advector.constrain_divergence_boundary(complex.topology.dual.selector[1] * flux_d1, return_phi=True)
                form = tris.topology.transfer_operators[0] * phi_0
                tris.as_2().plot_primal_0_form(
                    form, plot_contour=True, cmap='jet', vmin=-2e-3, vmax=+2e-3)

        plt.axis('off')


