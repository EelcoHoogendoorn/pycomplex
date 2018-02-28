
# -*- coding: utf-8 -*-

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

!! Seems to be a numerical unstability relating to pseudoinverse of dual velocity reconstruction
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from cached_property import cached_property

from examples.diffusion.explicit import Diffusor
from examples.flow.advection import Advector
from examples.util import save_animation
from pycomplex import synthetic
from pycomplex.complex.simplicial.spherical import ComplexSpherical
from pycomplex.math import linalg


class VorticityAdvector(Advector):
    # FIXME: is there any point in the class dirivation?

    def __init__(self, complex, diffusion=None):
        """

        Parameters
        ----------
        complex : BaseComplex
        diffusion : float, optional
        """
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

        velocity_d0 = self.complex.dual_flux_to_dual_velocity(flux_d1)


        # advect the dual mesh
        advected_d0 = self.complex.dual_position[0] + velocity_d0 * dt
        if isinstance(self.complex, ComplexSpherical):
            advected_d0 = linalg.normalized(advected_d0)

        # print(dt)
        if False:
            print(flux_d1.min(), flux_d1.max())
            print(velocity_d0.min(), velocity_d0.max())
            # import matplotlib.pyplot as plt
            # self.complex.plot(plot_dual=False)
            # self.complex.copy(vertices=advected_p0).plot(plot_dual=False)
            plt.figure()
            plt.quiver(self.complex.primal_position[2][:, 0], self.complex.primal_position[2][:, 1], velocity_d0[:, 0], velocity_d0[:, 1])
            plt.axis('equal')


        # sample at all advected dual vertices, average at the mid of dual edge, and dot with advected dual edge vector
        # FIXME:
        velocity_sampled_d0 = self.complex.sample_dual_0(velocity_d0, advected_d0)
        # FIXME: test if sampling with dt=0 gives us back our original velocity_d0

        # integrate the tangent flux of the advected mesh.
        velocity_sampled_d1 = self.complex.weighted_average_operators[1] * (velocity_sampled_d0)
        # velocity_sampled_d1 = self.complex.topology.average_operators[1] * (velocity_sampled_d0)
        advected_edge = self.complex.topology.dual.matrices[0].T * advected_d0
        # taking dot is euclidian integral
        flux_d1_advected = linalg.dot(velocity_sampled_d1, advected_edge)        # this does not include flux around boundary edges; but currently all zero anyway

        if False:
            # visualize flowfield after
            print(flux_d1_advected.min(), flux_d1_advected.max())
            velocity_d0_ = self.complex.dual_flux_to_dual_velocity(flux_d1_advected)
            print(velocity_d0_.min(), velocity_d0_.max())
            plt.figure()
            plt.quiver(*self.complex.primal_position[2].T, *velocity_d0_.T)
            plt.axis('equal')
            # plt.show()

        # drop boundary terms
        flux_d1_advected = self.complex.topology.dual.selector[1] * flux_d1_advected



        if force is not None:
            # add force impulse, if given
            flux_d1_advected += force * dt

        if self.complex.boundary is not None:
            F = self.complex.topology.dual.selector[1].T * self.constrain_divergence_boundary(flux_d1_advected)

            if False:
                print(F.min(), F.max())
                velocity_d0_ = self.complex.dual_flux_to_dual_velocity(F)
                print(velocity_d0_.min(), velocity_d0_.max())
                plt.figure()
                plt.quiver(*self.complex.primal_position[2].T, velocity_d0_.T)
                plt.axis('equal')
                plt.show()
            return F

        else:
            # on boundary-free domains, using pressure projection is simpler, since it does not require seperate treatnent of harmonic component
            return self.complex.topology.dual.selector[1].T * self.pressure_projection(flux_d1_advected)


if __name__ == "__main__":
    dt = .1

    # np.seterr(all='raise')
    # complex_type = 'grid'
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
            complex = complex.subdivide_cubical()

        complex = complex.as_22().as_regular()
        complex.topology.check_chain()
        tris = complex.subdivide_simplicial()
    if complex_type == 'simplex_quad':
        # the only difference between this and 'simplicial fluids', should be the dual interpolation
        while True:
            complex = synthetic.delaunay_cube(density=10, n_dim=2, iterations=50)

            # smooth while holding boundary constant
            # FIXME: need more utility functions for this; too much boilerplate for such a simple pattern
            chain_0 = complex.topology.chain(0, fill=0)
            chain_0[complex.boundary.topology.parent_idx[0]] = 1
            chain_1 = complex.topology.chain(1, fill=0)
            chain_1[complex.boundary.topology.parent_idx[1]] = 1
            creases = {0: chain_0, 1: chain_1}
            for i in range(2):
                complex = complex.as_2().subdivide_loop(smooth=True, creases=creases)
                for d, c in creases.items():
                    creases[d] = complex.topology.transfer_matrices[d] * c

            # FIXME: is weighted complex compatible at all with simplicial fluids?
            # complex = complex.optimize_weights_metric()
            print(complex.is_well_centered)
            if complex.is_well_centered:
                break
        # complex.plot()
        # plt.show()

    if complex_type == 'simplex':
        complex = synthetic.n_simplex(2).as_2().as_2()
        for i in range(6):
            complex = complex.subdivide_loop()
        complex.plot()
        plt.show()



    T01, T12 = complex.topology.matrices
    curl = T01.T
    D01, D12 = complex.topology.dual.matrices_2
    curl = T01.T
    D2D1 = D12.T

    from examples.harmonics import get_harmonics_0, get_harmonics_2

    if False:
        # generate a smooth incompressible flow field using harmonics

        H_p0 = get_harmonics_0(complex, zero_boundary=True)[:, 2]


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
        # H = get_harmonics_0(complex, zero_boundary=True)[:, 6]
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
        ) / 300 * 1 #+ H / 2000

        flux_p1 = curl * H
        flux_d1 = complex.hodge_DP[1] * flux_p1

    # set boundary tangent flux to zero
    flux_d1 = complex.topology.dual.selector[1].T * flux_d1

    # set up vorticity advector
    advector = VorticityAdvector(complex)
    # test that integrating over zero time does almost nothing
    flux_d1 = advector.advect_vorticity(flux_d1, dt=0)
    advected_0 = advector.advect_vorticity(flux_d1, dt=0)
    # print(np.abs(advected_0 - flux_d1).max())
    # print(np.abs(flux_d1).max())
    # FIXME: conservation is still far from achieved
    # assert np.allclose(advected_0, flux_d1, atol=1e-6)

    path = r'../output/euler_0'
    # path = None
    def advect(flux_d1, dt):
        return advector.advect_vorticity(flux_d1, dt)

    from examples.flow.advection import BFECC, MacCormack

    for i in save_animation(path, frames=200, overwrite=True):
        for r in range(4):
            flux_d1 = BFECC(advect, flux_d1, dt=1)
            # flux_d1 = advect(flux_d1, dt=1)

        vorticity_p0 = complex.hodge_PD[0] * (D2D1 * flux_d1)
        if complex.boundary is not None:
            vorticity_p0[complex.boundary.topology.parent_idx[0]] = 0   # dont care about shear in boundary layer

        if complex_type == 'sphere':
            complex.as_euclidian().as_3().plot_primal_0_form(
                vorticity_p0, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)
        if complex_type == 'simplex_quad':
            complex.as_2().as_2().plot_primal_0_form(
                vorticity_p0, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)
        if complex_type == 'simplex':
            complex.as_2().as_2().plot_primal_0_form(
                vorticity_p0, plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)
        if complex_type == 'grid':
            tris.as_2().plot_primal_0_form(
                tris.topology.transfer_operators[0] * vorticity_p0,
                plot_contour=False, cmap='bwr', shading='gouraud', vmin=-2e-1, vmax=+2e-1)

        plt.axis('off')
