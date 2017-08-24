"""Transverse electric electromagnetics

E is a primal 1-form
H is a primal 0-form

d/dt E = curl(H)
d/dt H = curl(E)


References
----------
https://jyx.jyu.fi/dspace/bitstream/handle/123456789/44630/978-951-39-5951-7_vaitos26112014.pdf?sequence=1

Todo
----
Add some nontrivial materials; dielectrics and so on
"""

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from cached_property import cached_property

from pycomplex import synthetic


def make_mesh():
    mesh = synthetic.n_cube(2, centering=True).as_22().as_regular()
    # subdivide
    for i in range(6):
        mesh = mesh.subdivide_cubical()

    return mesh


class Transverse(object):
    """Transverse electric field class"""

    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def operators(self):
        dE = P1P0.multiply(self.step)
        dH = (P0D2 * D2D1 * D1P1).multiply(self.step)
        return dE.tocsr(), dH.tocsr()

    def leapfrog(self, E_p1, H_p0):
        """single leapfrog step"""
        dE, dH = self.operators
        E_p1 = E_p1 + dE * H_p0
        H_p0 = H_p0 - dH * E_p1
        return E_p1, H_p0

    @cached_property
    def step(self):
        return 1. / np.sqrt(self.largest_eigenvalue / 2)

    @cached_property
    def largest_eigenvalue(self):
        return Diffusor(self.complex).largest_eigenvalue


if __name__ == "__main__":
    from examples.util import save_animation

    complex_type = 'sphere'

    if complex_type == 'grid':
        complex, all, left, bottom, current, closed = make_mesh()
        # complex.plot(plot_dual=False, plot_vertices=False)
        tris = complex.as_22().subdivide_simplicial()
    if complex_type == 'sphere':
        complex = synthetic.icosphere(refinement=6)

    transverse = Transverse(complex)


    T01, T12 = complex.topology.matrices
    P1P0 = T01.T
    D2D1 = T01
    P2P1 = T12.T
    D1D0 = T12

    P0D2, P1D1, P2D0 = [scipy.sparse.diags(h) for h in complex.hodge_PD]
    D2P0, D1P1, D0P2 = [scipy.sparse.diags(h) for h in complex.hodge_DP]


    H_p0 = complex.topology.form(0)
    E_p1 = complex.topology.form(1)

    # idx = complex.pick_dual(complex.box.mean(axis=0, keepdims=True))
    # H_p0[idx] = 1
    idx = complex.pick_dual(np.random.randn(10, 3))
    H_p0[idx] = 1

    from examples.diffusion.explicit import Diffusor
    H_p0 = Diffusor(complex).integrate_explicit_sigma(H_p0, sigma=0.02) * 5
    # H_p0 = complex.vertices[:, 0]


    path = r'c:\development\examples\transverse_electric_1'

    for i in save_animation(path, frames=500, overwrite=True):

        E_p1, H_p0 = transverse.leapfrog(E_p1, H_p0)

        if complex_type == 'grid':
            form = tris.topology.transfer_operators[0] * H_p0
            tris.as_2().plot_primal_0_form(
                form, plot_contour=False, cmap='seismic', shading='gouraud', vmin=-0.5, vmax=0.5)
        if complex_type == 'sphere':
            complex.as_euclidian().as_3().plot_primal_0_form(
                H_p0, plot_contour=False, cmap='seismic', shading='gouraud', vmin=-0.5, vmax=0.5)

        plt.axis('off')
