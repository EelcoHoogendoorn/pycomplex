"""Time-dependent diffusion is interesting because it arguably represents
the simplest 'introduction to vector calculus' possible, while still doing something physical and useful.

"""

import numpy as np
import scipy.sparse


class Diffusor(object):
    def __init__(self, complex):
        self.complex = complex
        self.laplacian, self.mass, self.inverse_mass_operator = self.laplacian_0()
        self.precompute()

    def laplacian_0(self):
        complex = self.complex
        T01 = complex.topology.matrices[0]
        grad = T01.T
        div = T01

        D1P1 = scipy.sparse.diags(complex.hodge_DP[1])
        P0D2 = scipy.sparse.diags(complex.hodge_PD[0])

        # construct our laplacian
        laplacian = div * D1P1 * grad
        mass = complex.hodge_DP[0]
        return laplacian, mass, P0D2

    def precompute(self):
        # compute largest eigenvalue, for optimally scaled explicit timestepping
        self.largest_eigenvalue = scipy.sparse.linalg.eigsh(
            self.laplacian,
            M=scipy.sparse.diags(self.mass),
            k=1, which='LM', tol=1e-6, return_eigenvectors=False)
        print(self.largest_eigenvalue)

    # def eigen(self):
    #     """Compute small magnitude eigencomponents; those that are hard to solve explicitly"""
    #     D2P0 = scipy.sparse.diags(complex.D2P0)
    #     values, vectors = scipy.sparse.linalg.eigsh(
    #         self.laplacian, M=D2P0, k=64, which='SM', tol=1e-6, return_eigenvectors=True)
    #     return values, vectors

    def explicit_step(self, field, fraction=1):
        """Forward Euler timestep

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form
        fraction : float, optional
            fraction == 1 is the stepsize that will exactly zero out the biggest eigenvector
            Values over 2 will be unstable

        Returns
        -------
        field : ndarray, [n_vertices], float
            primal 0-form
        """
        return field - (self.inverse_mass_operator * (self.laplacian * field)) * (fraction / self.largest_eigenvalue)

    def integrate_explicit(self, field, dt):
        """Integrate diffusion equation over a timestep dt"""
        distance = self.largest_eigenvalue * dt
        steps = int(np.ceil(distance))
        fraction = distance / steps
        for i in range(steps):
            field = self.explicit_step(field, fraction)
        return field

    def integrate_explicit_sigma(self, field, sigma):
        """Integrate for such a length of time,
         as to be equivalent to a gaussian blur with the given sigma"""
        dt = sigma ** 2 / np.sqrt(np.pi)
        return self.integrate_explicit(field, dt)


if __name__ == '__main__':
    kind = 'letter'

    if kind == 'sphere':
        from pycomplex import synthetic
        complex = synthetic.icosphere(refinement=6)
        complex = complex.copy(radius=16)
        # complex.metric(radius=16)
    if kind == 'regular':
        from pycomplex import synthetic
        complex = synthetic.n_cube_grid((32, 32)).as_22().as_regular()
        for i in range(2):
            complex = complex.subdivide()
        # complex.metric()
    if kind == 'letter':
        from examples.subdivision import letter_a
        complex = letter_a.create_letter(4).to_simplicial().as_3()
        complex = complex.copy(vertices=complex.vertices * 10)
        # complex.metric()
        # complex.plot_3d(plot_dual=False, plot_vertices=False)

    if True:
        field = np.random.rand(complex.topology.n_elements[0])
    else:
        print(complex.box)
        field = complex.topology.chain(0, dtype=np.float)
        idx = 0
        idx = np.argmin(np.linalg.norm(complex.vertices - [0, 0], axis=1))
        field[idx] = 1

    diffusor = Diffusor(complex)
    field = diffusor.integrate_explicit_sigma(field, 1.5)
    print(field.min(), field.max())

    if kind == 'sphere':
        complex = complex.as_euclidian()
        complex.plot_primal_0_form(field)
    if kind == 'regular':
        tris = complex.to_simplicial()
        field = tris.topology.transfer_operators[0] * field
        tris.as_2().plot_primal_0_form(field)
    if kind == 'letter':
        complex.plot_primal_0_form(field, plot_contour=False)
