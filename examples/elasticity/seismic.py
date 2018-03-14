"""
Set up scenario showing s and p waves in isotropic elastic material,
as a function of position and velocity fields p and v
and material parameters ρ, μ and λ

(μ dδ + λ δd) p = d/dt (v * ρ)
              v = d/dt p

Do simple explicit timestepping of this equation

We can let material properties go to small values to simulate free boundary,
or to large values to simulate fixed boundary

https://authors.library.caltech.edu/62269/2/LMHTD15.pdf
In this paper fixed boundaries are simulated by hodge going to zero
"""
from cached_property import cached_property

import numpy as np
import scipy.sparse

from examples.multigrid.equation import Equation


class Elastic(Equation):
    """Object to manage second order vectorial wave equation over dual 1-forms"""

    def __init__(self, complex, m, l, r):
        """

        Parameters
        ----------
        complex : Complex
            Complex to simulate over
        m : n-2-chain
            shear modulus field
        l : n-chain
            compressive modulus field
        r : n-1 chain
            density field
        """
        self.complex = complex
        self.m = m
        self.l = l
        self.r = r
        self.laplacian, self.mass, self.inverse_mass = self.operators

    @cached_property
    def operators(self):
        """Laplacian acting on dual 1-forms"""
        complex = self.complex
        m, l, r = [scipy.sparse.diags(p) for p in [self.m, self.l, self.r]]
        DPl, DPm, DPr = [scipy.sparse.diags(h) for h in complex.hodge_DP[-3:]]
        PDl, PDm, PDr = [scipy.sparse.diags(h) for h in complex.hodge_PD[-3:]]
        Pl, Pr = [t.T for t in complex.topology.matrices[-2:]]
        Dl, Dr = np.transpose(Pl), np.transpose(Pr)

        A = Pl * PDl * m * Dl + PDm * Dr * DPr * l * Pr * PDm

        # FIXME: is product of primal/dual metric a good mass term?
        # or is plain hodge the way to go?
        # P, D = complex.metric
        # mass = (P[::-1][1] * D[1]) * r
        mass = PDm * r
        B = mass
        BI = scipy.sparse.diags(1/(mass*complex.topology.chain(-2, 1, np.float))) # scipy.sparse.diags(1/mass)
        # BI = scipy.sparse.linalg.inv(B)
        return A.tocsr(), B.tocsc(), BI.tocsc()

    def operate(self, x):
        return (self.inverse_mass * (self.laplacian * x))

    def explicit_step(self, p, v, fraction=1):
        """Forward Euler timestep

        Parameters
        ----------
        p : dual 1-form
            position
        v : dual 1-form
            velocity
        fraction : float, optional
            fraction == 1 is the stepsize that will exactly zero out the biggest eigenvector
            Values over 2 will be unstable

        Returns
        -------
        field : ndarray, [n_vertices], float
            primal 0-form
        """
        p = p + v * fraction
        v = v - self.operate(p) * (fraction / self.largest_eigenvalue)
        return p, v

    def integrate_explicit(self, p, v, dt):
        distance = dt * 2 * np.pi
        steps = int(np.ceil(distance))
        fraction = distance / steps
        for i in range(steps):
            p, v = self.explicit_step(p, v, fraction)
        return p, v

    @cached_property
    def integrate_eigen_precompute(self):
        return self.eigen_basis(K=150, amg=True, tol=1e-14)
    def integrate_eigen(self, p, v, dt):
        V, eigs = self.integrate_eigen_precompute
        pe = np.dot(V.T, self.mass * p) #/ eigs
        ve = np.dot(V.T, self.mass * v) #/ eigs

        c = pe + ve * 1j
        c = c * np.exp(np.pi * 2j * np.sqrt(eigs) / np.sqrt(self.largest_eigenvalue) * dt)
        pe = np.real(c)
        ve = np.imag(c)

        return np.dot(V, pe), np.dot(V, ve)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    kind = 'regular'

    # if kind == 'sphere':
    #     from pycomplex import synthetic
    #     complex = synthetic.icosphere(refinement=6)
    #     complex = complex.copy(radius=16)

    if kind == 'simplicial':
        from pycomplex import synthetic
        complex = synthetic.delaunay_cube(density=64, n_dim=2)
        complex = complex.copy(vertices=complex.vertices - 0.5).optimize_weights().as_2().as_2()

    if kind == 'regular':
        from pycomplex import synthetic
        complex = synthetic.n_cube_grid((1, 1)).as_22().as_regular()
        for i in range(6):
            complex = complex.subdivide_cubical()

    # set up circular domain; everything outside the circle are 'air cells'
    def circle(p, sigma=0.5, radius=0.4):
        sigma = sigma * complex.metric[1][1].mean()
        return scipy.special.erfc((np.linalg.norm(p, axis=1) - radius) / sigma) / 2
    def step(p, sigma=0.5, pos=[0, 0], dir=[-1, 0]):
        sigma = sigma * complex.metric[1][1].mean()
        return scipy.special.erfc(np.dot(p - pos, dir) / sigma) / 2
    def rect(p, s=0.5):
        return step(p, s, pos=[0.4, 0], dir=[1, 0]) * step(p, s, pos=[-0.4, 0], dir=[-1, 0]) * \
               step(p, s, pos=[0, 0.05], dir=[0, 1]) * step(p, s, pos=[0, -0.05], dir=[0, -1])


    # set up scenario
    pp = complex.primal_position[0]
    if True:
        d = circle(pp) + 0.001
    else:
        d = rect(pp) + 0.001
    # d = np.ones_like(d)
    powers = 1., 1., 1.
    m, r, l = [(o * np.power(d, p)) for o, p in zip(complex.topology.averaging_operators_0[-3:], powers)]
    # r = np.ones_like(r)
    m *= 0.4     # mu is shear stiffness
    if False:
        complex.plot_primal_0_form(d, cmap='jet', plot_contour=False)
        plt.show()

    equation = Elastic(complex, m=m, l=l, r=r)
    print(equation.largest_eigenvalue)

    if True:
        # set up impulse; do in velocity space? need to add velocity to flux mapping
        print(complex.box)
        p = complex.topology.chain(1, dtype=np.float)
        v = complex.topology.chain(1, dtype=np.float)
        idx = 0
        idx = np.argmin(np.linalg.norm(complex.dual_position[1] - [0.05, 0.35 + 0.05], axis=1))
        p[idx] = .03
        # smooth impulse a little since the high frequency components are visually distracting
        for i in range(30):
            p = p - equation.operate(p) / equation.largest_eigenvalue


    def plot_flux(fd1):
        complex.plot_primal_0_form(m - 0.5, levels=3, cmap=None)
        ax = plt.gca()
        complex.plot_dual_flux(fd1, plot_lines=True, ax=ax)

        ax.set_xlim(*complex.box[:, 0])
        ax.set_ylim(*complex.box[:, 1])
        plt.axis('off')


    # toggle between eigenmodes or time stepping
    if False:
        # output eigenmodes
        path = r'../output/seismic_modes_0'
        from examples.util import save_animation
        V, v = equation.eigen_basis(K=150, amg=True, tol=1e-14)
        print(v)
        for i in save_animation(path, frames=len(v), overwrite=True):
            plot_flux(V[:, i] * r / 1e2)

    elif True:
        # time integration using explicit integration
        path = r'../output/seismic_0'
        from examples.util import save_animation
        for i in save_animation(path, frames=200, overwrite=True):
            p, v = equation.integrate_explicit(p, v, 1)
            plot_flux(p * r)
    else:
        # time integration using eigen basis
        path = r'../output/seismic_1'
        from examples.util import save_animation
        for i in save_animation(path, frames=200, overwrite=True):
            p, v = equation.integrate_eigen(p, v, 1)
            plot_flux(p * r)
