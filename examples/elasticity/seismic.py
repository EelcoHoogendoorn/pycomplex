
# -*- coding: utf-8 -*-

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
In this paper fixed boundaries are simulated by primal metrics going to zero

Note that with air-cell type method, we do not get `pure` null space vectors
This is ok for physical reasons, but eigensolver stability is a bit concerning.
Perhaps feeding it the idealized null space explicitly would help?
or would geometric multigrid render this a non-issue?

think more about implications of air-elements on the micro-level
if we had a hard boundary coinciding with primal elements, what would happen?
alternation of primal-metric only makes result undefined, actually. edge has to be either out or in.
if it is just in, flux through it should be equal to opposing boundary;
which indeed it will be, considering all orthogonal fluxes wil be forced to zero

is this a benefit of going to zero though? what about shear on a free boundary?
is behavior continuous under such discontinuous jumps?

symmetric setup right now is strange; we modify the rotation at each vertex, which has no physical analogue
would be more natural to multiply with density-like factor on edges at the end of the shear force calculation.
making this symmetric requires a similar term (division by density-like term) in the rotation-definition.
this means defacto that we are modelling momentum rather than displacement;
might be good for numerical stability? small unknown magnitude in air; directs solver attention better

changed the unknowns from velocity to momentum; seems to help some with stability, but no miracles

given that we have a fully working vector equation now,
we have a basis to experiment with geometric multigrid

we could model it as a pure first order problem, including rotation and pressure variables
this is numerically more robust, as known from incompressible stokes problem,
and might not be worse in terms of operation count in a multigrid context.
probably not great for amg though

havnt tried feeding nullspace to amg yet; could be interesting

note that we could make cube spacing anisotropic; might make for more efficient air cells?
or is this the same as letting shear go to zero?
"""

from cached_property import cached_property

import numpy as np
import scipy.sparse

import pycomplex
from examples.multigrid.equation import SymmetricEquation


class Elastic(SymmetricEquation):
    """Object to manage second order vectorial wave equation over dual 1-forms"""

    def __init__(self, complex, m, l, r):
        """

        Parameters
        ----------
        complex : Complex
            Complex to simulate over
        m : n-2-chain
            shear modulus field (mu, G)
        l : n-chain
            compressive modulus field (lambda)
        r : n-1 chain
            density field (rho)
        """
        self.complex = complex
        self.mu = m
        self.lamb = l
        self.rho = r
        self.laplacian, self.mass, self.inverse_mass = self.operators
        self.k = complex.n_dim - 1

    @cached_property
    def operators(self):
        """Laplacian acting on dual 1-forms

        quantity being modelled is effectively momentum, rather than velocity
        this helps in that it makes any linear solvers focus on errors within the domain of interest

        Note that for simulating free boundaries, domain bc where vorticity rather than tangent flux
        is zero will likely converge faster. However the current formulation with dual-boundary==0 is the simplest.
        Alternatively, could use a toroidal domain

        Note that we could derive this operator from a first-order system as well by elimination
        """
        complex = self.complex
        l, m, r = [scipy.sparse.diags(p) for p in [self.mu, self.rho, self.lamb]] # NOTE: maps physical variaables to left-mid-right scheme
        li, mi, ri = [pycomplex.sparse.inv_diag(p) for p in [l, m, r]]
        DPl, DPm, DPr = [scipy.sparse.diags(h) for h in complex.hodge_DP[-3:]]
        PDl, PDm, PDr = [scipy.sparse.diags(h) for h in complex.hodge_PD[-3:]]
        Pl, Pr = [t.T for t in complex.topology.matrices[-2:]]
        Dl, Dr = np.transpose(Pl), np.transpose(Pr)

        # left term is shear, right term is pressure
        # FIXME: is zero-vorticity bc just a matter of replacing left term with dual?
        # FIXME: also need to wrap right term in selectors
        # more interesting still; need a mass term for dual boundary unknowns too. any ideas here?
        A = mi * Pl * PDl * l * Dl * mi + \
            mi * PDm * Dr * DPr * r * Pr * PDm * mi
        # A = (A + A.T) * 0.5   # some numerical asymmetry during construction; does not seem to make a different to solver tho

        if False:
            # experimental gravity wave term; not a success
            G = r * PDm * Dr * DPr * self.lamb   # lamb is density at primal n-cubes; G gradient of density
            G = scipy.sparse.diags(G * -1e-3)
            A = A + G

        # FIXME: is product of primal/dual metric a good mass term?
        # or is plain hodge the way to go?
        # seems like plain hodge is superior as judged on triangles;
        # this suggests we shouldnt think of this term as 'mass' alone
        # P, D = complex.metric
        # mass = (P[::-1][1] * D[1]) * r
        mass = PDm
        B = mass.todia()
        BI = pycomplex.sparse.inv_diag(B)
        return A.tocsr(), B.tocsc(), BI.tocsc()

    @cached_property
    def null_space(self):
        """Depends on bcs.

        if constant rotation patterns is in nullspace, why constant compression isnt?
        depends on the topology of the domain; constant compression can be null on an annulus
        """
        return

    def operate(self, x):
        return (self.inverse_mass * (self.laplacian * x))

    def explicit_step(self, p, v, fraction=1):
        """Forward Euler timestep of second order wave equation

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
            dual 1-form
        """
        p = p + v * fraction
        v = v - self.operate(p) * (fraction / self.largest_eigenvalue)
        return p, v

    def integrate_explicit(self, p, v, dt):
        """Forward Euler explicit integration of second order wave equation"""
        distance = dt * 2 * np.pi
        steps = int(np.ceil(distance))
        fraction = distance / steps
        for i in range(steps):
            p, v = self.explicit_step(p, v, fraction)
        return p, v

    @cached_property
    def integrate_eigen_precompute(self):
        return self.eigen_basis(K=80, amg=True, tol=1e-14)
    def integrate_eigen(self, p, v, dt):
        """Eigenbasis integration of second order wave equation"""
        V, eigs = self.integrate_eigen_precompute
        pe = np.dot(V.T, self.mass * p) #/ eigs
        ve = np.dot(V.T, self.mass * v) #/ eigs

        c = pe + ve * 1j
        c = c * np.exp(np.pi * 2j * np.sqrt(eigs) / np.sqrt(self.largest_eigenvalue) * dt)
        pe = np.real(c)
        ve = np.imag(c)

        return np.dot(V, pe), np.dot(V, ve)


    def solve(self, y):
        """Solve elastic linear system in eigenbasis

        Parameters
        ----------
        y : ndarray

        Returns
        -------
        x : ndarray
        """
        A, B, BI = self.operators

        def poisson(y, v):
            null_rank = np.abs(v / self.largest_eigenvalue) < 1e-9

            x = np.zeros_like(y)
            # poisson linear solve is simple division in eigenspace. skip nullspace
            # swivel dimensions to start binding broadcasting dimensions from the left
            x[null_rank:] = (y[null_rank:].T / v[null_rank:].T).T
            return x

        return self._solve_eigen(B * y, poisson)

    @cached_property
    def transfer(self):
        """Transfer operator of state variables between fine and coarse representation

        Returns
        -------
        sparse matrix
            entry [c, f] is the overlap between fine and coarse dual n-cells

        """
        return self.complex.multigrid_transfers[self.k]

    @cached_property
    def restrictor(self):
        """Maps dual 1-form from fine to coarse"""
        A, B, BI = self.operators
        # map to 'intermediate' space between primal and dual
        fine = scipy.sparse.diags(1. / self.complex.dual_metric[self.k])
        coarse = scipy.sparse.diags(self.complex.parent.dual_metric[self.k])
        return coarse * pycomplex.sparse.normalize_l1(self.transfer.T, axis=0) * fine
    def restrict(self, fine):
        """Restrict solution from fine to coarse"""
        return self.restrictor * fine
    @cached_property
    def interpolator(self):
        """Maps dual 1-form from coarse to fine"""
        A, B, BI = self.operators
        # convert to dual; then transfer, then back to primal. why again?
        # getting correct operator in regular grids relies on not doing this
        fine = scipy.sparse.diags(self.complex.hodge_PD[0])
        coarse = scipy.sparse.diags(self.complex.parent.hodge_DP[0])
        return pycomplex.sparse.normalize_l1(self.transfer, axis=1)
    def interpolate(self, coarse):
        """Interpolate solution from coarse to fine"""
        return self.interpolator * coarse


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
        # NOTE: no hierarchy so not compatible with geometric mg

    if kind == 'regular':
        from pycomplex import synthetic
        complex = synthetic.n_cube_grid((1, 1)).as_22().as_regular()
        hierarchy = [complex]
        for i in range(6):
            complex = complex.subdivide_cubical()
            hierarchy.append(complex)

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
    air_density = 1e-3      # mode calculation gets numerical problems with lower densities. 1e-4 is feasible with current configuration
    # FIXME: investigate how much the choice of sigma, or boundary layer really matters. does it influence the spectrum, for instance?
    # sigma 0.5->0.2 messes badly with the rotational symmetry of our circle
    # sigma 0.5 -> 1.0 takes rotation mode from 37 to 29;
    # hard to imagine a significant impact on body modes, but surface modes are probably affected
    # also has impact on first pinching mode; at least on relative position; should check absolute eigenvalues
    if True:
        d = circle(pp) + air_density
    else:
        d = rect(pp) + air_density
    # d = np.ones_like(d)
    powers = 1., 1., 1.
    m, r, l = [(o * np.power(d, p)) for o, p in zip(complex.topology.averaging_operators_0[-3:], powers)]
    # r = np.ones_like(r)
    # m += 1e-4     # different stiffness/density ratios in air dont seem to help much either
    # r += 1e-1
    # l += 1e-4

    # m *= 0.1     # mu is shear stiffness. low values impact stability of eigensolve; 0.4 ratio seems about an upper bound
    # l *= 0.1

    if False:
        complex.plot_primal_0_form(d, cmap='jet', plot_contour=False)
        plt.show()

    # FIXME: to do proper geometric mg, need to coarsen anisotropy fields as well
    # FIXME: might be better to start prototyping vectorial mg in a simpler context
    # FIXME: or should we just implement petrov-galerkin smoothing?
    # equations = [Elastic(c) for c in hierarchy]
    equation = Elastic(complex, m=m, l=l, r=r)
    print(equation.largest_eigenvalue)

    if True:
        # set up impulse; do in velocity space? need to add velocity to flux mapping
        print(complex.box)
        p = complex.topology.chain(-2, dtype=np.float)
        v = complex.topology.chain(-2, dtype=np.float)
        idx = 0
        # idx = np.argmin(np.linalg.norm(complex.dual_position[1] - [0.05, 0.35 + 0.05], axis=1))
        idx = np.argmin(np.linalg.norm(complex.primal_position[-2] - [0.015, 0.05], axis=1))
        v[idx] = .03
        # smooth impulse a little since the high frequency components are visually distracting
        for i in range(300):
            v = v - equation.operate(p) / equation.largest_eigenvalue

    def segment(levelset, value=None):
        """Segment levelset == value"""
        if value is None:
            value = (levelset.min() + levelset.max()) / 2

        tris = complex.subdivide_simplicial()
        c0 = tris.topology.transfer_operators[0] * (levelset - value)
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        triang = tri.Triangulation(*tris.vertices[:, :2].T, triangles=tris.topology.triangles)

        fig, ax = plt.subplots(1, 1)

        levels = np.linspace(c0.min()-1e-6, c0.max()+1e-6, 3, endpoint=True)
        c = ax.tricontour(triang, c0, colors='k', levels=levels)
        contour = c.allsegs[0][0][:-1]  # this is how a single contour can be extracted
        plt.close()

        from pycomplex.complex.simplicial.euclidian import ComplexSimplicialEuclidian

        vidx = np.arange(len(contour))
        edges = np.array([vidx, np.roll(vidx, 1)]).T
        surface = ComplexSimplicialEuclidian(vertices=contour, simplices=edges)
        return surface


    def plot_flux(fd1):
        complex.plot_primal_0_form(m - 0.5, levels=3, cmap=None)
        ax = plt.gca()
        complex.plot_dual_flux(fd1, plot_lines=True, ax=ax)

        ax.set_xlim(*complex.box[:, 0])
        ax.set_ylim(*complex.box[:, 1])
        plt.axis('off')


    # toggle between different outputs
    output = 'modes'
    if output == 'modes':
        # output eigenmodes
        path = r'../output/seismic_modes_1'
        from examples.util import save_animation
        from time import time
        t = time()
        # FIXME: using preconditioning influences spectrum? could be an error of lobpcg
        # however, modes without preconditioning are completely useless, so hard to say
        # for simple isotropic square domain, pattern persists: adding amg slows down by factor two,
        # but yields purer modes, despite being forced to lower tolerance
        V, v = equation.eigen_basis(K=50, preconditioner='amg', tol=1e-12)
        print('eigen solve time:', time() - t)
        print(v)
        # quit()
        for i in save_animation(path, frames=len(v), overwrite=True):
            plot_flux(V[:, i] * (r**0) * 1e-2)

    elif output == 'explicit_integration':
        # time integration using explicit integration
        path = r'../output/seismic_0'
        from examples.util import save_animation
        for i in save_animation(path, frames=200, overwrite=True):
            p, v = equation.integrate_explicit(p, v, 1)
            plot_flux(p * r)
    elif output == 'eigen_integration':
        # time integration using eigen basis
        path = r'../output/seismic_1'
        from examples.util import save_animation
        for i in save_animation(path, frames=200, overwrite=True):
            p, v = equation.integrate_eigen(p, v, 1)
            plot_flux(p * r)
    elif output == 'travelling':
        # traveling wave;
        # note that 18+19 on the sphere have prograde motion;
        # is this a bug, a consequence of sphere geometry,
        # or due to the gradient boundary rather than sharp edge?
        path = r'../output/seismic_travel_1'
        from examples.util import save_animation
        t = np.linspace(0, 2*np.pi, 50)
        V, v = equation.eigen_basis(K=80, amg=True, tol=1e-16)
        c = V[:, 18] + 1j * V[:, 19]
        # c = V[:, 5] + 1j * V[:, 4]

        for i in save_animation(path, frames=len(t), overwrite=True):
            p = c * np.exp(1j * t[i])
            plot_flux(p.real * r / 1e2)

    elif output == 'surface':
        # perform animation by mapping eigenvectors back to surface
        # FIXME: counterintuitive influence of air-density. at higher density, we get oscilation of planet
        # probably since below certain treshold, pseudorigid body motions are numerically-null-space?
        V, v = equation.eigen_basis(K=50, amg=True, tol=1e-14)
        M = equation.mass * np.ones(len(V))
        V = V * np.sqrt(M)[:, None]
        # np.dot(V.T, V) == I, after this division by sqrt(M)
        vn = v / equation.largest_eigenvalue

        surface = segment(m)

        # # map eigen solution to the surface
        S = complex.topology.dual.selector_interior[-2]
        Vs = complex.sample_dual_0(complex.dual_flux_to_dual_velocity(S.T * V), surface.vertices)

        # set up on grid too, so we can diffuse?
        d = np.zeros_like(Vs[..., 0])
        v = np.zeros_like(d)
        # v[len(d)//4-30:len(d)//4+30, 0] = 1e-3
        d = Vs[..., 0:12].sum(axis=-1) * 2e-3   # excited a number of lower modes
        # d[0, 0] = 1e-0


        Q = np.linalg.inv(np.einsum('vnk, vnl', Vs, Vs))

        def integrate_eigen(d, v, dt, damping=10):
            # FIXME: need to add d/v mass multiplication; does current scheme of moving it to vectors work?
            # FIXME: vectors are no longer orthonormal in Vs matrix
            # FIXME: therefore, this mapping isnt identity when dt=0; need explicit invert of reduced matrix?
            # NOTE: seems to work alright, but not sure if it is correct
            pe = np.einsum('vnk, vn -> k', Vs, d)  # / eigs
            ve = np.einsum('vnk, vn -> k', Vs, v)  # / eigs

            pe = np.dot(Q, pe)
            ve = np.dot(Q, ve)
            c = pe + ve * 1j
            c = c * np.exp(np.pi * 2j * np.sqrt(vn) * dt) * np.exp(-vn * damping * dt)
            pe = np.real(c)
            ve = np.imag(c)

            return np.einsum('vnk, k -> vn', Vs, pe), np.einsum('vnk, k -> vn', Vs, ve)


        path = r'../output/seismic_surf_0'
        from examples.util import save_animation
        for i in save_animation(path, frames=200, overwrite=True):
            d, v = integrate_eigen(d, v, 50)
            print(np.linalg.norm(d))
            surface.copy(vertices=surface.vertices + d).plot(plot_dual=False)
            # plt.show()
            ax = plt.gca()
            ax.set_xlim(*complex.box[:, 0])
            ax.set_ylim(*complex.box[:, 1])
            plt.axis('off')
