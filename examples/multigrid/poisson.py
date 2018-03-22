"""So far not easy to beat minres+amg

So far, twice as slow for same error on a regular grid
However, pure AMG is beat by a factor 4
need to try geometric MG as minres preconditioner
So far minres+mg gives about the same result as pure mg;
interesting how mg and amg are different that way


This may be different for vector-type problems
also interesting to see how mg performs on anisotropic problems
and interesting to see how mg does on eigenproblems; especially wrt numerical stability
no clear advantage thus far
"""

import numpy as np
import scipy.sparse
from cached_property import cached_property

from examples.multigrid.equation import Equation
from pycomplex.sparse import normalize_l1, inv_diag


class Poisson(Equation):
    """Wrap complex with poisson equation methods and state

    Poisson logic for primal 0-forms, or 0-laplace-beltrami

    A * x = B * y, or laplacian(x) = mass(y)

    Notes
    -----
    need to think of clear interface to extend this to multicomplex as well.
    what do we truly expect as interface from the complex that we bind to here?
    """

    def __init__(self, complex, k=0):
        """

        Parameters
        ----------
        complex : BaseComplex
        k : int
            laplace order

        """
        self.complex = complex
        self.k = k

    @cached_property
    def operators(self):
        """Construct laplace-beltrami on primal k-forms

        Returns
        -------
        A : Linear operator
            symmetric;
            maps primal k-form to dual k-form
        B : Linear operator
            symmetric diagonal;
            maps primal k-form to dual k-form
        BI :
            inverse of B
        """
        k = self.k
        DP = [scipy.sparse.diags(h) for h in self.complex.hodge_DP]
        PD = [scipy.sparse.diags(h) for h in self.complex.hodge_PD]

        DPl, DPm, DPr = ([0] + DP + [0])[k:][:3]
        PDl, PDm, PDr = ([0] + PD + [0])[k:][:3]
        Dl, Dr = ([0] + self.complex.topology.matrices + [0])[k:][:2]
        Pl, Pr = np.transpose(Dl), np.transpose(Dr)

        A = DPm * Pl * PDl * Dl * PDm + Dr * DPr * Pr

        # FIXME: is mass just hodge for non-scalar forms? seems like
        B = DPm
        BI = inv_diag(B)

        return A.tocsr(), B.tocsc(), BI.tocsc()

    @cached_property
    def null_space(self):
        """Return null space of the equations"""
        A, B, BI = self.operators

        return np.ones((B.shape[1], 1))

    def solve(self, y):
        """Solve poisson linear system in eigenbasis

        Parameters
        ----------
        y : ndarray

        Returns
        -------
        x : ndarray
        """
        A, B, BI = self.operators

        def poisson(y, v):
            x = np.zeros_like(y)
            # poisson linear solve is simple division in eigenspace. skip nullspace
            null_rank = self.null_space.shape[1]
            # swivel dimensions to start binding broadcasting dimensions from the left
            x[null_rank:] = (y[null_rank:].T / v[null_rank:].T).T
            return x

        return self._solve_eigen(B * y, poisson)

    # FIXME: mg operators belong to the complexes themselves
    @cached_property
    def transfer(self):
        """Transfer operator of state variables between fine and coarse representation

        Returns
        -------
        sparse matrix
            entry [c, f] is the overlap between fine and coarse dual n-cells

        Notes
        -----
        only available for SphericalTriangularComplex; implementation on regular grids should be easy tho
        """
        try:
            # triangular 0-form special case
            fine = self.complex
            coarse = fine.parent
            return self.complex.multigrid_transfer_dual(coarse, fine).T
        except:
            return self.complex.multigrid_transfers[self.k]

    @cached_property
    def restrictor(self):
        """Maps primal 0-form from fine to coarse"""
        A, B, BI = self.operators
        # FIXME: need a link to parent equation; this is hacky. should be coarse.BI
        # convert to dual; then transfer, then back to primal. why again?
        fine = scipy.sparse.diags(self.complex.hodge_DP[0])
        coarse = scipy.sparse.diags(self.complex.parent.hodge_PD[0])
        return coarse * normalize_l1(self.transfer.T, axis=0) * fine
    def restrict(self, fine):
        """Restrict solution from fine to coarse"""
        return self.restrictor * fine
    @cached_property
    def interpolator(self):
        """Maps primal 0-form from coarse to fine"""
        A, B, BI = self.operators
        # convert to dual; then transfer, then back to primal. why again?
        # getting correct operator in regular grids relies on not doing this
        fine = scipy.sparse.diags(self.complex.hodge_PD[0])
        coarse = scipy.sparse.diags(self.complex.parent.hodge_DP[0])
        return normalize_l1(self.transfer, axis=1)
    def interpolate(self, coarse):
        """Interpolate solution from coarse to fine"""
        return self.interpolator * coarse


# class PoissonDual(Equation):
#     """Wrap complex with poisson equation methods and state
#
#     Poisson logic for dual n-forms, or n-laplace-beltrami
#
#     A * x = B * y, or laplacian(x) = mass(y)
#
#     """
#
#     def __init__(self, complex, k, dual=True):
#         """
#
#         Parameters
#         ----------
#         complex : BaseComplex
#         k : int
#             degree of the laplacian
#         dual : bool
#             if true, dual boundary terms are included in the unknowns
#             if false, these are implicitly zero and only primal topology is used
#         """
#         self.complex = complex
#         self.k = k
#         self.dual = dual
#
#     @cached_property
#     def operators(self):
#         """Construct laplace-beltrami on dual k-forms
#
#         Returns
#         -------
#         A : Linear operator
#             symmetric;
#             maps dual k-form to primal k-form
#         B : Linear operator
#             symmetric diagonal;
#             maps dual k-form to primal k-form
#         BI :
#             inverse of B
#         """
#         k = self.k
#         DPl, DPm, DPr = ([0] + self.complex.hodge_DP + [0])[k:][:3]
#         PDl, PDm, PDr = ([0] + self.complex.hodge_PD + [0])[k:][:3]
#         if self.dual:
#             Dl, Dr = ([0] + self.complex.topology.dual.matrices_2 + [0])[k:][:2]
#             Pl, Pr = np.transpose(Dl), np.transpose(Dr)
#         else:
#             Pl, Pr = ([0] + self.complex.topology.matrices + [0])[k:][:2]
#             Dl, Dr = np.transpose(Pl), np.transpose(Pr)
#
#         A = Pl * PDl * Dl + \
#             PDm * Dr * DPr * Pr * PDm
#
#         # FIXME: is mass just hodge for non-scalar forms?
#         B = scipy.sparse.diags(PDm)
#         BI = inv_diag(B)
#
#         return A.tocsr(), B.tocsc(), BI.tocsc()
#
#     def solve(self, y):
#         """Solve poisson linear system in eigenbasis
#
#         Parameters
#         ----------
#         y : ndarray
#
#         Returns
#         -------
#         x : ndarray
#         """
#         A, B, BI = self.operators
#
#         def poisson(y, v):
#             x = np.zeros_like(y)
#             # poisson linear solve is simple division in eigenspace. skip nullspace
#             x[1:] = y[1:] / v[1:]
#             return x
#
#         return self._solve_eigen(B * y, poisson)
#
#     # FIXME: mg operators belong to the complexes themselves
#     @cached_property
#     def transfer(self):
#         """Transfer operator of state variables between fine and coarse representation
#
#         Returns
#         -------
#         sparse matrix
#             entry [c, f] is the overlap between fine and coarse dual n-cells
#
#         Notes
#         -----
#         might switch to special case for triangular complex and 0-forms
#         """
#         if False:
#             # triangular 0-form special case
#             fine = self.complex
#             coarse = fine.parent
#             return self.complex.multigrid_transfer_dual(coarse, fine).T
#         else:
#             return self.complex.multigrid_transfers[self.k]
#
#     @cached_property
#     def restrictor(self):
#         A, B, BI = self.operators
#         # FIXME: need a link to parent equation; this is hacky. should be coarse.BI
#         # note: is this the same for vector equations?
#         coarse = scipy.sparse.diags(self.complex.parent.hodge_PD[0])
#         return coarse * normalize_l1(self.transfer.T, axis=0) * B
#     def restrict(self, fine):
#         """Restrict solution from fine to coarse; dual to dual"""
#         return self.restrictor * fine
#     @cached_property
#     def interpolator(self):
#         A, B, BI = self.operators
#         coarse = scipy.sparse.diags(self.complex.parent.hodge_DP[0])
#         return BI * normalize_l1(self.transfer, axis=0) * coarse
#     def interpolate(self, coarse):
#         """Interpolate solution from coarse to fine; dual to dual"""
#         return self.interpolator * coarse


class GeometricMultiGrid(object):
    """Perhaps use this to cache the MG-specific data?"""
    def __init__(self, complex, equation):
        self.complex = complex
        self.equation = equation


class Hierarchy(object):
    def __init__(self, hierarchy, equation):
        self.hierarchy = hierarchy
        self.equations = [equation(l) for l in hierarchy]

    def solve(self, y):
        pass

    def as_preconditioner(self):
        pass


if __name__ == '__main__':
    from examples.multigrid import multigrid
    from time import clock
    from examples.diffusion.perlin_noise import perlin_noise

    from pycomplex import synthetic
    import matplotlib.pyplot as plt

    complex_type = 'sphere'
    print(complex_type)

    if complex_type == 'sphere':
        sphere = synthetic.icosahedron().copy(radius=30).subdivide_fundamental()
        hierarchy = [sphere]
        for i in range(4):
            hierarchy.append(hierarchy[-1].subdivide_loop())

    if complex_type == 'regular':
        # test if multigrid operators on primal-0-forms work correctly on regular grids
        root = synthetic.n_cube(n_dim=2).as_22().as_regular()
        hierarchy = [root]
        for i in range(5):
            hierarchy.append(hierarchy[-1].subdivide_cubical())

    # set up hierarchy of equations
    equations = [Poisson(l, k=0) for l in hierarchy]

    # test transfer operators
    if False:
        R = equations[-1].restrictor
        I = equations[-1].interpolator
        T = R * I
        ones = np.ones((T.shape[1], 1))
        q = T * ones
        print(q)
        T = I * R
        ones = np.ones((T.shape[1], 1))
        q = T * ones
        print(q)


    if False:
        # monkey patch this dumb assert
        from scipy.sparse.linalg.eigen.lobpcg import lobpcg
        lobpcg._assert_symmetric = lambda x: None

        # test eigen solve; seems to work just fine
        # V, v = equations[-1].eigen_basis(K=100, preconditioner='amg')
        V, v = equations[-1].eigen_basis(K=201, preconditioner=multigrid.as_preconditioner(equations[2:]), tol=1e-6)

        print(equations[-1].largest_eigenvalue)
        hierarchy[-1].plot_primal_0_form(V[:, -1])
        plt.show()

    # now test multigrid; what is a good testcase?
    # visually not that obvious; but we can focus on numbers first
    # if we can solve poisson with perlin input using mg,
    # we should be good, since it contains all frequency components

    p0 = perlin_noise(hierarchy[-1])
    p0 -= p0.mean()
    # p0 = np.random.normal(size=p0.size)




    x0 = np.zeros_like(p0)
    print('initial res')
    print(np.linalg.norm(equations[-1].residual(x0, p0)))

    # warm up cache
    x = multigrid.solve_full_cycle(equations, p0, iterations=1)
    t = clock()
    x = multigrid.solve_full_cycle(equations, p0, iterations=4)
    print('mg full time: ', clock() - t)
    print('mg full resnorm', np.linalg.norm(equations[-1].residual(x, p0)))

    # warm up amg
    x_minres = equations[-1].solve_minres(p0, preconditioner='amg')
    t = clock()
    x_minres = equations[-1].solve_minres(p0, preconditioner='amg')
    print('minres+amg time: ', clock() - t)
    print('minres+amg resnorm', np.linalg.norm(equations[-1].residual(x_minres, p0)))

    # warm up mg
    x_minres = equations[-1].solve_minres(p0, preconditioner=multigrid.as_preconditioner(equations[1:]))
    t = clock()
    # x_minres = equations[-1].solve_minres(p0, amg=True)
    x_minres = equations[-1].solve_minres(p0, preconditioner=multigrid.as_preconditioner(equations[1:]))
    print('minres+mg time: ', clock() - t)
    print('minres+mg resnorm', np.linalg.norm(equations[-1].residual(x_minres, p0)))


    t = clock()
    x_amg = equations[-1].solve_amg(p0)
    print('amg time: ', clock() - t)
    print('amg resnorm', np.linalg.norm(equations[-1].residual(x_amg, p0)))

    # x_eigen = equations[-1].solve(p0)
    # print(np.linalg.norm(equations[-1].residual(x_eigen, p0)))

    if True:
        v = x
        vmin, vmax = v.min(), v.max()
        if complex_type == 'sphere':

            hierarchy[-1].as_euclidian().plot_primal_0_form(v, vmin=vmin, vmax=vmax)
        if complex_type == 'regular':
            hierarchy[-1].plot_primal_0_form(v, vmin=vmin, vmax=vmax)

        # hierarchy[-2].as_euclidian().plot_primal_0_form(equations[-1].restrict(p0), vmin=vmin, vmax=vmax)
        # hierarchy[-1].as_euclidian().plot_primal_0_form(equations[-1].interpolate(equations[-1].restrict(p0)), vmin=vmin, vmax=vmax)

        plt.show()
