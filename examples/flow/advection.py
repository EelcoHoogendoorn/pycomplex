
# -*- coding: utf-8 -*-

"""Advection equations are a good illustration of the mesh picking functionality"""

# FIXME: remove any dependencies on 2d complexes

import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property

from examples.util import save_animation
from pycomplex import synthetic
# from pycomplex.complex.simplicial.spherical import ComplexSpherical
from pycomplex.math import linalg


class Advector(object):

    def __init__(self, complex):
        """

        Parameters
        ----------
        complex: BaseComplex
            must have a `sample_dual_0` method
        """
        self.complex = complex

    @cached_property
    def dual_flux_to_dual_velocity(self):
        """

        Returns
        -------
        callable that maps a dual-1-form representing tangent flux,
        to a vector field represented at dual vertices in the embedding space
        """
        # assert self.complex.is_pairwise_delaunay    # required since we are dividing by dual edge lengths; does not work for cubes yet
        # T01, T12 = self.complex.topology.matrices
        D01, D12 = self.complex.topology.dual.matrices_2
        D1D0 = D01.T

        # from pycomplex.geometry import euclidian
        # gradients = euclidian.simplex_gradients(self.complex.vertices[self.complex.topology.elements[-1]])
        # u, s, v = np.linalg.svd(gradients)
        # s = 1 / s
        # pinv = np.einsum('...ij,...j,...jk->...ki', u[..., :s.shape[-1]], s, v)
        # # gradients.dot(velocity) = normal_flux

        # solve using normal equations instead?
        # grad.shape = 3, 2
        # normal = np.einsum('...ji,...jk->...ik', gradients, gradients)
        # inv = np.linalg.inv(normal)
        # pinv = np.einsum('...ij,...kj->...ik', inv, gradients)

        # check = np.einsum('...ij,...jk->...ik', pinv, gradients)

        # from pycomplex.geometry import euclidian
        # gradients = euclidian.simplex_gradients(self.complex.vertices[self.complex.topology.elements[-1]])
        # u, s, v = np.linalg.svd(gradients)
        # s = 1 / s
        # pinv = np.einsum('...ij,...j,...jk->...ki', u[..., :], s, v)
        # # check = np.einsum('...ij,...jk->...ik', pinv, gradients)




        # # for incompressible flows on simplicial topologies, there exists a 3-vector at the dual vertex,
        # # which projected on the dual edges forms the dual fluxes. on a sphere the third component is not determined
        # # approximate inverse would still make sense in cubical topology however
        # # tangent_directions.dot(velocity) = tangent_velocity_component
        B = self.complex.topology._boundary[-1]
        O = self.complex.topology._orientation[-1]
        B = B.reshape(len(B), -1)
        O = O.reshape(len(O), -1)
        # # tangent edges per primal n-element
        # # FIXME: need to enforce incompressibility constraint. more easily done using face normals
        dual_vertex = self.complex.dual_position[0]
        dual_edge_vector = D1D0 * dual_vertex
        tangent_directions = (dual_edge_vector)[B] * O[..., None]
        # compute pseudoinverse, to quickly construct velocities at dual vertices
        # for a regular grid, this should be just an averaging operator in both dims
        u, s, v = np.linalg.svd(tangent_directions)
        s = 1 / s
        # s[:, self.complex.topology.n_dim:] = 0
        pinv = np.einsum('...ij,...j,...jk->...ki', u[..., :s.shape[-1]], s, v)

        def dual_flux_to_dual_velocity(flux_d1):
            flux_d1 = self.complex.topology.dual.selector[1] * flux_d1
            # compute velocity component in the direction of the dual edge
            # tangent_velocity_component = (flux_d1 )[B] * O
            normal_flux = (self.complex.hodge_PD[1] * flux_d1)[B] * O
            print(normal_flux.shape)
            # given these flows incident on the dual vertex, reconstruct the velocity vector there
            velocity_d0 = np.einsum('...ij,...j->...i', pinv, normal_flux)

            # # project out part not on the sphere
            # if isinstance(self.complex, ComplexSpherical):
            #     velocity_d0 = velocity_d0 - dual_vertex * (velocity_d0 * dual_vertex).sum(axis=1, keepdims=True)

            # rec_flux = np.einsum('...ij,...j->...i', gradients, velocity_d0)
            # assert np.allclose(rec_flux, normal_flux, atol=1e-6)
            # cast away dual boundary flux, then pad velocity with zeros... not quite right, should use the boundary information
            velocity_d0 = self.complex.topology.dual.selector[-1].T * velocity_d0

            return velocity_d0

        return dual_flux_to_dual_velocity

    def advect_p0(self, flux_d1, field_p0, dt):
        velocity_d0 = self.dual_flux_to_dual_velocity(flux_d1)
        mesh_p0 = self.complex.primal_position[0]
        # is there any merit to higher order integration of this step?
        advected_p0 = mesh_p0 + self.complex.sample_dual_0(velocity_d0, mesh_p0) * dt
        # FIXME: sample is overkill here; could just average directly
        # No, need correct boundary handling; which is lacking atm btw


        if False:
            # import matplotlib.pyplot as plt
            self.complex.plot(plot_dual=False)
            self.complex.copy(vertices=advected_p0).plot(plot_dual=False)
            plt.figure()
            plt.quiver(self.complex.primal_position[2][:, 0], self.complex.primal_position[2][:, 1], velocity_d0[:, 0], velocity_d0[:, 1])
            plt.axis('equal')
            plt.show()


        return self.complex.sample_primal_0(field_p0, advected_p0)

    def advect_d0(self, flux_d1, field_d0, dt):
        velocity_d0 = self.dual_flux_to_dual_velocity(flux_d1)
        mesh_d0 = self.complex.dual_position[0]
        advected_d0 = mesh_d0 + velocity_d0 * dt
        return self.complex.sample_dual_0(field_d0, advected_d0)


def BFECC(advector, state, dt):
    remapped = advector(advector(state, dt), -dt)
    err = remapped - state
    final = advector(state - err / 2, dt)
    return final


def MacCormack(advector, state, dt):
    # http://physbam.stanford.edu/~fedkiw/papers/stanford2006-09.pdf
    # FIXME: add limiter restricted to range of values in upwind sampling domain?
    forward = advector(state, dt)
    backward = advector(forward, -dt)
    return forward + (state - backward) / 2



if __name__ == "__main__":
    # advect the texture for constant flow field to illustrate advection
    dt = 1

    complex_type = 'sphere'

    if complex_type == 'sphere':
        complex = synthetic.icosphere(refinement=5)
        if False:
            complex.plot()

    if complex_type == 'grid':
        complex = synthetic.n_cube_grid((1, 1), False)
        for i in range(5):
            complex = complex.subdivide_cubical()

        complex = complex.as_22().as_regular()
        complex.topology.check_chain()
        tris = complex.subdivide_simplicial()

    if complex_type == 'simplex_grid':
        while True:
            complex = synthetic.delaunay_cube(30, 2, iterations=50)

            # smooth while holding boundary constant
            # FIXME: need more utility functions for this; too much boilerplate for such a simple pattern
            chain_0 = complex.topology.chain(0, fill=0)
            chain_0[complex.boundary.topology.parent_idx[0]] = 1
            chain_1 = complex.topology.chain(1, fill=0)
            chain_1[complex.boundary.topology.parent_idx[1]] = 1
            creases = {0: chain_0, 1: chain_1}
            for i in range(0):
                complex = complex.as_2().subdivide_loop(smooth=True, creases=creases)
                for d, c in creases.items():
                    creases[d] = complex.topology.transfer_matrices[d] * c

            complex = complex.optimize_weights_metric()
            print(complex.is_well_centered)
            if complex.is_well_centered:
                break


    # generate an interesting texture using RD
    from examples.diffusion.perlin_noise import perlin_noise
    texture_p0 = perlin_noise(
        complex,
        [
            (.05, .05),
            (.1, .1),
            (.2, .2),
            (.4, .4),
            # (.8, .8),
        ]
    ) / 100
    vmin, vmax = texture_p0.min(), texture_p0.max()

    # generate a smooth incompressible flow field using harmonics
    from examples.harmonics import get_harmonics_0, get_harmonics_2

    H_p0 = get_harmonics_0(complex, zero_boundary=True)[:, 5]
    T01, T12 = complex.topology.matrices
    curl = T01.T
    # complex.plot_primal_0_form(H_p0)
    # plt.show()

    # if False:
    #     H_d0 = get_harmonics_2(complex)[:, 2]
    #     complex.as_2().as_2().plot_dual_0_form_interpolated(complex.topology.dual.selector[-1].T * H_d0, weighted=True)
    #     plt.show()
    #
    #     A = complex.topology.averaging_operators_N
    #     H_p0 = complex.hodge_PD[0] * (A[0] * H_d0)
    #     # H_p0[complex.boundary.topology.parent_idx[0]] = 0
    #
    #     if True:
    #         # form = tris.topology.transfer_operators[0] * H_p0
    #         # tris.as_2().plot_primal_0_form(form)
    #         complex.as_2().as_2().plot_primal_0_form(H_p0)
    #         plt.show()
    #
    #     # form = tris.topology.transfer_operators[0] * H[:, 2]
    #     # tris.as_2().plot_dual_2_form_interpolated(
    #     #     form, plot_contour=False, cmap='terrain', shading='gouraud')
    #     # plt.show()

    flux_d1 = complex.hodge_DP[1] * (curl * (H_p0)) / 400
    flux_d1 = complex.topology.dual.selector[1].T * flux_d1

    path = r'../output/advection_0'

    advector = Advector(complex)
    def advect(p0, dt):
        return advector.advect_p0(flux_d1, p0, dt)

    for i in save_animation(path, frames=50, overwrite=True):
        for r in range(1):
            # texture_p0 = MacCormack(advect, texture_p0, dt=1)
            texture_p0 = advect(texture_p0, dt=1)

        if complex_type == 'sphere':
            complex.as_euclidian().as_3().plot_primal_0_form(
                texture_p0, plot_contour=False, cmap='terrain', shading='gouraud', vmin=vmin, vmax=vmax)
        if complex_type == 'grid':
            form = tris.topology.transfer_operators[0] * texture_p0
            tris.as_2().plot_primal_0_form(
                form, plot_contour=False, cmap='terrain', shading='gouraud', vmin=vmin, vmax=vmax)
        if complex_type == 'simplex_grid':
            complex.as_2().as_2().plot_primal_0_form(
                texture_p0, plot_contour=False, cmap='terrain', shading='gouraud', vmin=vmin, vmax=vmax)

        plt.axis('off')
