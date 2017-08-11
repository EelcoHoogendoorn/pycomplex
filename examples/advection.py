"""Advection equations are a good illustration of the mesh picking functionality"""

import numpy as np
from cached_property import cached_property
import scipy.sparse
import matplotlib.pyplot as plt
from pycomplex.util import save_animation

from pycomplex import synthetic
from pycomplex.math import linalg
from pycomplex.complex.spherical import ComplexSpherical


class Advector(object):

    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def dual_flux_to_dual_velocity(self):
        # T01, T12 = self.complex.topology.matrices
        D01, D12 = self.complex.topology.dual.matrices_2
        D1D0 = D01.T

        dual_vertex = self.complex.dual_position[0]
        dual_edge_vector = D1D0 * dual_vertex
        # for incompressible flows on simplicial topologies, there exists a 3-vector at the dual vertex,
        # which projected on the dual edges forms the dual fluxes. on a sphere the third component is not determined
        # approximate inverse would still make sense in cubical topology however
        # tangent_directions.dot(velocity) = tangent_velocity_component
        B = self.complex.topology._boundary[-1]
        O = self.complex.topology._orientation[-1]
        B = B.reshape(len(B), -1)
        O = O.reshape(len(O), -1)
        # tangent edges per primal n-element
        tangent_directions = linalg.normalized(dual_edge_vector)[B] #* O[..., None]
        # compute pseudoinverse, to quickly construct velocities at dual vertices
        # for a regular grid, this should be just an averaging operator in both dims
        u, s, v = np.linalg.svd(tangent_directions)
        s = 1 / s
        # s[:, self.complex.topology.n_dim:] = 0
        pinv = np.einsum('...ij,...j,...jk->...ki', u[..., :s.shape[-1]], s, v)

        def dual_flux_to_dual_velocity(flux_d1):
            flux_d1 = self.complex.topology.dual.selector[1] * flux_d1
            # compute velocity component in the direction of the dual edge
            tangent_velocity_component = (flux_d1 / self.complex.dual_metric[1])[B] #* O
            # given these flows incident on the dual vertex, reconstruct the velocity vector there
            velocity_d0 = np.einsum('...ij,...j->...i', pinv, tangent_velocity_component)

            # project out part not on the sphere
            if isinstance(self.complex, ComplexSpherical):
                velocity_d0 = velocity_d0 - dual_vertex * (velocity_d0 * dual_vertex).sum(axis=1, keepdims=True)

            # cast away dual boundary flux, then pad velocity with zeros... not quite right, should use the boundary
            velocity_d0 = self.complex.topology.dual.selector[-1].T * velocity_d0

            return velocity_d0

        return dual_flux_to_dual_velocity


    def advect_p0(self, flux_d1, field_p0, dt):
        velocity_d0 = self.dual_flux_to_dual_velocity(flux_d1)
        mesh_p0 = self.complex.primal_position[0]
        # is there any merit to higher order integration of this step?
        advected_p0 = mesh_p0 + self.complex.sample_dual_0(velocity_d0, mesh_p0) * dt
        # FIXME: sample is overkill here; could just average directly
        # self.complex.copy(vertices=advected_p0).plot()
        # import matplotlib.pyplot as plt
        # plt.show()
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
    return forward + (state - backward) / 3



if __name__ == "__main__":
    # advect the texture for constant flow field to illustrate advection
    dt = 1

    complex_type = 'grid'

    if complex_type == 'sphere':
        complex = synthetic.icosphere(refinement=5)
        if False:
            complex.plot()

    if complex_type == 'grid':
        complex = synthetic.n_cube_grid((1, 1), False)
        for i in range(5):
            complex = complex.subdivide()

        complex = complex.as_22().as_regular()
        complex.topology.check_chain()
        tris = complex.to_simplicial()


    # generate an interesting texture using RD
    from examples.diffusion.planet_perlin import perlin_noise
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
    H = get_harmonics_0(complex)
    T01, T12 = complex.topology.matrices
    curl = T01.T

    if True:
        H_d0 = get_harmonics_2(complex)[:, 2]

        A = complex.topology.dual.averaging_operators()
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

    flux_d1 = complex.hodge_DP[1] * (curl * (H_p0)) / 100


    path = r'c:\development\examples\advection_14'

    advector = Advector(complex)
    def advect(p0, dt):
        return advector.advect_p0(flux_d1, p0, dt)

    for i in save_animation(path, frames=50, overwrite=True):
        for r in range(1):
            # texture_p0 = MacCormack(advect, texture_p0, dt=20)
            texture_p0 = advect(texture_p0, dt=1)

        if complex_type == 'sphere':
            complex.as_euclidian().as_3().plot_primal_0_form(
                texture_p0, plot_contour=False, cmap='terrain', shading='gouraud', vmin=vmin, vmax=vmax)
        if complex_type == 'grid':
            form = tris.topology.transfer_operators[0] * texture_p0
            tris.as_2().plot_primal_0_form(
                form, plot_contour=False, cmap='terrain', shading='gouraud', vmin=vmin, vmax=vmax)

        plt.axis('off')
