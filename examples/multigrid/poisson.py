import numpy as np
import scipy.sparse
from cached_property import cached_property

from pycomplex.sparse import normalize_l1


class Equation(object):
    """Encode operators and other defining information about a system of equations"""

    def solve(self, x):
        raise NotImplementedError

    def residual(self, x, rhs):
        raise NotImplementedError

    def _solve_eigen(self, x, func):
        """map a dual n-form eigenspace, and solve func in eigenspace

        usefull for for poisson and diffusion solver
        """
        V, v = self.eigen_all
        y = np.einsum('vji,...ji->v', V, x)
        y = func(y, v)
        y = np.einsum('vji,v...->ji', V, y)
        return y


class Poisson(Equation):
    """Wrap complex with poisson equation methods and state

    Poisson logic for primal 0-forms, or 0-laplace-beltrami

    Can look at equation in two ways;
    M-1 * L * x = y, or L * x = M * y
    L and M both map primal-0-forms to dual-n-forms

    Notes
    -----
    currently sphericalTriangular complexes only;
    need to think of clear interface to extend this to multicomplex

    """

    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def laplacian(self):
        """construct laplacian on primal 0-forms"""
        S = self.complex.topology.selector[0]   # this term sets boundary to zero
        T01 = self.complex.topology.matrix(0, 1).T
        grad = T01
        div = T01.T
        mass = S * self.complex.hodge_DP[0]
        D1P1 = scipy.sparse.diags(self.complex.hodge_DP[1])

        # construct our laplacian
        laplacian = S * div * D1P1 * grad * S.T
        return laplacian.tocsr(), mass

    @cached_property
    def largest_eigenvalue(self):
        L, M = self.laplacian
        return scipy.sparse.linalg.eigsh(
            L, M=scipy.sparse.diags(M).tocsc(),
            k=1, which='LM', tol=1e-6, return_eigenvectors=False)

    @cached_property
    def complete_eigen_basis(self):
        """Get full 0-laplacian eigenbasis of a small complex

        Returns
        -------
        V : ndarray, [complex.n_vertices, complex.n_vertices]
            n-th column is n-th eigenvector
        v : ndarray, [complex.n_vertices]
            eigenvalues, sorted low to high
        """
        L, M = self.laplacian
        M = scipy.sparse.diags(M)
        v, V = scipy.sparse.linalg.lobpcg(A=L, B=M, X=np.random.normal(size=L.shape))
        return V, v

    def solve(self, rhs):
        """solve poisson linear system in eigenbasis

        Parameters
        ----------
        rhs : dual n-form

        Returns
        -------
        primal 0-form
        """

        def poisson(x, v):
            y = np.zeros_like(x)
            y[1:] = x[1:] / v[1:]  # poisson linear solve is simple division in eigenspace. skip nullspace
            return y

        return self._solve_eigen(rhs, poisson)

    def residual(self, x, rhs):
        L, M = self.laplacian
        return rhs - L * x

    def descent(self, x, rhs):
        """alternative to jacobi iteration"""
        r = self.residual(x, rhs)
        return x + r / (self.largest_eigenvalue * 0.9)  # no harm in a little overrelaxation

    def overrelax(self, x, rhs, knots):
        """overrelax, forcing the eigencomponent to zero at the specified overrelaxation knots"""
        for s in knots / self.largest_eigenvalue:
            x = x + self.residual(x, rhs) * s
        return x

    def jacobi(self, x, rhs):
        """jacobi iteration on a d2-form"""
        raise NotImplementedError
        # return self.inverse_diagonal * (rhs - self.boundify(self.off_diagonal * self.deboundify( x)))

    def minres(self, rhs, x0):
        """Solve equation using minres"""


    def eigen_basis(self, K, amg=False, tol=1e-10):
        """Compute partial eigen decomposition"""
        L, M = self.laplacian

        B = scipy.sparse.diags(M).tocsc()
        if not amg:
            W, V = scipy.sparse.linalg.eigsh(
                L., M=B, which='SA', k=K, tol=tol)
        else:
            # create the AMG hierarchy
            from pyamg import smoothed_aggregation_solver
            ml = smoothed_aggregation_solver(L)
            # initial approximation to the K eigenvectors
            X = scipy.rand(L.shape[0], K)
            # preconditioner based on ml
            M = ml.aspreconditioner()
            # compute eigenvalues and eigenvectors with LOBPCG
            W, V = scipy.sparse.linalg.lobpcg(L, X, B=B, tol=tol, M=M, largest=False)
        return V, W

    @cached_property
    def transfer(self):
        """

        Returns
        -------
        sparse matrix
            entry [c, f] is the overlap between fine and coarse dual n-cells

        Notes
        -----
        only available for SphericalTriangularComplex; implementation on regular grids should be easy tho
        """
        fine = self.complex
        coarse = fine.parent
        return self.complex.multigrid_transfer_dual(coarse, fine).T

    @cached_property
    def restrict(self):
        f2c = normalize_l1(self.transfer.T, axis=0)
        return f2c
    @cached_property
    def interpolate(self):
        c2f = normalize_l1(self.transfer, axis=0)
        return c2f


class GeometricMultiGrid(object):
    """Perhaps use this to cache the MG-specific data?"""
    def __init__(self, complex, equation):
        self.complex = complex
        self.equation = equation




if __name__ == '__main__':
    from pycomplex import synthetic

    sphere = synthetic.icosahedron()# .subdivide_fundamental()
    hierarchy = [sphere]
    for i in range(3):
        hierarchy.append(hierarchy[-1].subdivide_loop())
    equations = [Poisson(l) for l in hierarchy]

    from examples.multigrid.multigrid import solve_full_cycle