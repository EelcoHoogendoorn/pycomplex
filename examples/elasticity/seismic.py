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

"""

from cached_property import cached_property

import numpy as np
import scipy.sparse

from examples.multigrid.equation import Equation


def inv_diag(d):
    return scipy.sparse.diags(1 / (d * np.ones(d.shape[1], d.dtype)))


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

        if False:
            # experimental gravity wave term; not a success
            G = r * PDm * Dr * DPr * self.l   # l is density at primal n-cubes; G gradient of density
            G = scipy.sparse.diags(G * -1e-3)
            A = A + G

        # FIXME: is product of primal/dual metric a good mass term?
        # or is plain hodge the way to go?
        # P, D = complex.metric
        # mass = (P[::-1][1] * D[1]) * r
        mass = PDm * r
        B = mass
        BI = inv_diag(mass)
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
    if False:
        d = circle(pp) + 0.001
    else:
        d = rect(pp) + 0.001
    # d = np.ones_like(d)
    powers = 1., 1., 1.
    m, r, l = [(o * np.power(d, p)) for o, p in zip(complex.topology.averaging_operators_0[-3:], powers)]
    # r = np.ones_like(r)
    m *= 0.4     # mu is shear stiffness. low values impact stability of eigensolve
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
    output = 'surface'
    if output == 'modes':
        # output eigenmodes
        path = r'../output/seismic_modes_0'
        from examples.util import save_animation
        V, v = equation.eigen_basis(K=50, amg=True, tol=1e-14)
        print(v)
        for i in save_animation(path, frames=len(v), overwrite=True):
            plot_flux(V[:, i] * r * 1e-2)

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
        path = r'../output/seismic_travel_0'
        from examples.util import save_animation
        t = np.linspace(0, 2*np.pi, 50)
        V, v = equation.eigen_basis(K=50, amg=True, tol=1e-14)
        c = V[:, 18] + 1j * V[:, 19]

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
        S = complex.topology.dual.selector[-2]
        Vs = complex.sample_dual_0(complex.dual_flux_to_dual_velocity(S.T * V), surface.vertices)

        d = np.zeros_like(Vs[..., 0])
        d[len(d)//4, 0] = 1e-2
        # d[0, 0] = 1e-0
        v = np.zeros_like(d)

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
            d, v = integrate_eigen(d, v, 10)
            print(np.linalg.norm(d))
            surface.copy(vertices=surface.vertices + d).plot(plot_dual=False)
            # plt.show()
            ax = plt.gca()
            ax.set_xlim(*complex.box[:, 0])
            ax.set_ylim(*complex.box[:, 1])
