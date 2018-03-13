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


class Elastic(object):
    """Object to manage second order vectorial wave equation over dual 1-forms"""

    def __init__(self, complex, m, l, r):
        """

        Parameters
        ----------
        complex : Complex
        m : n-2-chain
            shear modulus
        l : n-chain
            compressive modulus
        r : n-1 chain
            density
        """
        self.complex = complex
        self.m = m
        self.l = l
        self.r = r
        self.laplacian, self.mass, self.inverse_mass_operator = self.laplacian()

    def laplacian(self):
        """Laplacian acting on dual 1-forms"""
        complex = self.complex
        m, l, r = self.m, self.l, self.r
        m, l = scipy.sparse.diags(m), scipy.sparse.diags(l)
        DPl, DPm, DPr = [scipy.sparse.diags(h) for h in complex.hodge_DP[-3:]]
        PDl, PDm, PDr = [scipy.sparse.diags(h) for h in complex.hodge_PD[-3:]]
        Pl, Pr = [t.T for t in complex.topology.matrices[-2:]]
        Dl, Dr = np.transpose(Pl), np.transpose(Pr)

        A = Pl * PDl * m * Dl + PDm * Dr * DPr * l * Pr * PDm

        # FIXME: is mass just hodge for non-scalar forms? dont think so...; more like product of primal/dual metric
        P, D = complex.metric
        mass = (P[::-1][1] * D[1]) * r
        B = scipy.sparse.diags(mass)
        BI = scipy.sparse.diags(1/mass)

        return A.tocsr(), B.tocsc(), BI.tocsc()

    @cached_property
    def largest_eigenvalue(self):
        # compute largest eigenvalue, for optimally scaled explicit timestepping
        return scipy.sparse.linalg.eigsh(
            self.laplacian,
            M=self.mass,
            k=1, which='LM', tol=1e-6, return_eigenvectors=False)

    def operate(self, x):
        return (self.inverse_mass_operator * (self.laplacian * x))
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
        p = p + v
        v = v - self.operate(p) * (fraction / self.largest_eigenvalue)
        return p, v

    def integrate_explicit(self, field, dt):
        """Integrate diffusion equation over a timestep dt

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form
        dt : float
            timestep

        Returns
        -------
        field : ndarray, [n_vertices], float
            diffused primal 0-form

        """
        distance = self.largest_eigenvalue * dt
        steps = int(np.ceil(distance))
        fraction = distance / steps
        for i in range(steps):
            field = self.explicit_step(field, fraction)
        return field



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    kind = 'regular'

    # if kind == 'sphere':
    #     from pycomplex import synthetic
    #     complex = synthetic.icosphere(refinement=6)
    #     complex = complex.copy(radius=16)
    if kind == 'regular':
        from pycomplex import synthetic
        complex = synthetic.n_cube_grid((1, 1)).as_22().as_regular()
        for i in range(6):
            complex = complex.subdivide_cubical()
        def circle(p, sigma, radius=0.4):
            return scipy.special.erfc((np.linalg.norm(p, axis=1) - radius) / sigma) / 2
        pp = complex.primal_position[0]
        d = circle(pp, sigma=complex.metric[1][1].mean() / 8) + 0.01
        d = np.ones_like(d)
        m, r, l = [(o * d) for o in complex.topology.averaging_operators_0[-3:]]
        # r = np.ones_like(r)
        m *= .4


    equation = Elastic(complex, m, l, r)
    print(equation.largest_eigenvalue)

    if False:
        field = np.random.rand(complex.topology.n_elements[0])
    else:
        print(complex.box)
        p = complex.topology.chain(1, dtype=np.float)
        v = complex.topology.chain(1, dtype=np.float)
        idx = 0
        idx = np.argmin(np.linalg.norm(complex.vertices - [0, 0], axis=1))
        p[idx] = .02
        for i in range(30):
            p = p - equation.operate(p) / equation.largest_eigenvalue



    path = r'../output/seismic_0'
    from examples.util import save_animation
    for i in save_animation(path, frames=200, overwrite=True):
        for i in range(3):
            p, v = equation.explicit_step(p, v, 1)

        if kind == 'regular':
            # map flux to primal velocity; kinda ugly
            S = complex.topology.dual.selector
            dp = S[-1] * complex.dual_flux_to_dual_velocity(S[1].T * p)
            dp = complex.topology.averaging_operators_N[0] * dp
            # plot warped mesh
            complex.copy(vertices=complex.primal_position[0] + dp * d[:, None]).plot(plot_dual=False)


        plt.axis('off')
