"""Advection equations are a good illustration of the mesh picking functionality"""

import numpy as np
from cached_property import cached_property
import scipy.sparse
import matplotlib.pyplot as plt
from pycomplex.util import save_animation

from pycomplex import synthetic
from pycomplex.math import linalg


class Advector(object):

    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def dual_flux_to_dual_velocity(self):
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
        tangent_edges = linalg.normalized(dual_edge_vector)[B] #* O[..., None]
        # compute pseudoinverse, to quickly construct velocities at dual vertices
        u, s, v = np.linalg.svd(tangent_edges)
        s = 1 / s
        # s[:, self.complex.topology.n_dim:] = 0
        pinv = np.einsum('...ij,...j,...jk->...ki', u, s, v)

        def dual_flux_to_dual_velocity(flux_d1):
            # compute velocity component in the direction of the dual edge
            tangent_velocity_component = (flux_d1 / self.complex.dual_metric[1])[B] #* O
            # given these flows incident on the dual vertex, reconstruct the velocity vector there
            velocity_d0 = np.einsum('...ij,...j->...i', pinv, tangent_velocity_component)
            return velocity_d0

        return dual_flux_to_dual_velocity

    @cached_property
    def dual_averages(self):
         return self.complex.weighted_average_operators()

    def sample_dual_0(self, d0, points):
        # FIXME: make this a method on Complex? would need to cache the averaging operators
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = [a * d0 for a in self.dual_averages]
        domain, bary = self.complex.pick_fundamental(points)
        # do interpolation over fundamental domain
        return sum([dual_forms[::-1][i][domain[:, i]] * bary[:, [i]]
                    for i in range(self.complex.topology.n_dim + 1)])
    def sample_primal_0(self, p0, points):
        element, bary = self.complex.pick_primal(points)
        IN0 = self.complex.topology.incidence[-1, 0]
        verts = IN0[element]
        return (p0[verts] * bary).sum(axis=1)

    def advect_p0(self, flux_d1, field_p0, dt):
        velocity_d0 = self.dual_flux_to_dual_velocity(flux_d1)
        mesh_p0 = self.complex.primal_position[0]
        # is there any merit to higher order integration of this step?
        advected_p0 = mesh_p0 + self.sample_dual_0(velocity_d0, mesh_p0) * dt   # FIXME: sample is overkill here; could just average directly
        return self.sample_primal_0(field_p0, advected_p0)

    def advect_d0(self, flux_d1, field_d0, dt):
        velocity_d0 = self.dual_flux_to_dual_velocity(flux_d1)
        mesh_d0 = self.complex.dual_position[0]
        advected_d0 = mesh_d0 + velocity_d0 * dt
        return self.sample_dual_0(field_d0, advected_d0)

    # def __call__(self, flux, state, dt):
    #     return self.advect_p0()


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

    sphere = synthetic.icosphere(refinement=5)
    if False:
        sphere.plot()
    dt = 1

    # generate an interesting texture using RD
    from examples.diffusion.planet_perlin import perlin_noise
    texture_p0 = perlin_noise(sphere,
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
    from examples.harmonics import get_harmonics_0
    H = get_harmonics_0(sphere)
    T01, T12 = sphere.topology.matrices
    curl = T01.T
    flux_d1 = sphere.hodge_DP[1] * (curl * (H[:, -2] + H[:, 2] * 10))


    path = r'c:\development\examples\advection_5'

    advector = Advector(sphere)
    def advect(p0, dt):
        return advector.advect_p0(flux_d1, p0, dt)

    for i in save_animation(path, frames=50):
        for r in range(2):
            texture_p0 = BFECC(advect, texture_p0, dt=20)
            # texture_p0 = MacCormack(texture_p0, dt=4)
        # sphere.as_euclidian().as_3().plot_primal_0_form(phi_p0, plot_contour=True, cmap='jet', vmin=-2e-2, vmax=+2e-2)
        sphere.as_euclidian().as_3().plot_primal_0_form(
            texture_p0, plot_contour=False, cmap='terrain', shading='gouraud', vmin=vmin, vmax=vmax)
        plt.axis('off')
        # plt.show()

