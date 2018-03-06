import numpy as np
import scipy.sparse
from cached_property import cached_property

from pycomplex.sparse import normalize_l1


class Equation(object):
    """Encode operators and other defining information about a system of equations

    We consider here generalized linear problems of the form Ax=By,
    where A is symmetric, and B diagonal

    Notes
    -----
    Seems like there is overlap in intention with BlockSystem;
    not sure yet what the cleanest conceptual breakup is
    BlockSystem manages blocks;
    this defines the core operation (self.poisson),
    and provides methods aimed at interfacing with mg-solver
    defining the operation is prob better left to the Block class

    First lets focus on getting the poisson problem to work.
    Generalizing to block systems can be done guided by stokes problem
    """

    @cached_property
    def operators(self):
        """Return A, B and B.I"""
        raise NotImplementedError

    def solve(self, y):
        """Solve system exactly. most efficient way may be problem-dependent"""
        raise NotImplementedError

    def solve_dense(self, y):
        """Solve system using dense linear algebra; for debugging purposes, mostly"""
        raise NotImplementedError

    def residual(self, x, rhs):
        raise NotImplementedError

    @cached_property
    def largest_eigenvalue(self):
        """Get largest eigenvalue for `A x = B x`

        Returns
        -------
        float
        """
        A, B, BI = self.operators
        return scipy.sparse.linalg.eigsh(A, M=B, k=1, which='LM', tol=1e-6, return_eigenvectors=False)

    @cached_property
    def complete_eigen_basis(self):
        """Get full eigenbasis for `A x = B x`

        Returns
        -------
        V : ndarray, [A.shape]
            n-th column is n-th eigenvector
        v : ndarray, [A.shape[0]]
            eigenvalues, sorted low to high
        """
        return self.eigen_basis(K=)
        A, B, BI = self.operators
        v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=np.random.normal(size=A.shape))
        return V, v

    def eigen_basis(self, K, amg=False, tol=1e-10):
        """Compute partial eigen decomposition for `A x = B x`

        Returns
        -------
        V : ndarray, [A.shape, K]
            n-th column is n-th eigenvector
        v : ndarray, [K]
            eigenvalues, sorted low to high
        """
        A, B, BI = self.operators

        if not amg:
            v, V = scipy.sparse.linalg.eigsh(A, M=B, which='SA', k=K, tol=tol)
        else:
            # create the AMG hierarchy
            from pyamg import smoothed_aggregation_solver
            ml = smoothed_aggregation_solver(A)
            # initial approximation to the K eigenvectors
            X = scipy.rand(A.shape[0], K)
            # preconditioner based on ml
            M = ml.aspreconditioner()
            # compute eigenvalues and eigenvectors with LOBPCG
            v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, M=M, largest=False)
        return V, v

    def _solve_eigen(self, x, func):
        """solve func in eigenspace

        usefull for for poisson and diffusion solver, and probably others too
        """
        V, v = self.complete_eigen_basis
        y = np.einsum('vji,...ji->v', V, x)
        y = func(y, v)
        y = np.einsum('vji,v...->ji', V, y)
        return y

    def solve_minres(self, y, x0, tol=1e-6):
        """Solve equation using minres"""
        A, B, BI = self.operators
        return scipy.sparse.linalg.minres(A=A, b=B * y, x0=x0, tol=tol)


class Poisson(Equation):
    """Wrap complex with poisson equation methods and state

    Poisson logic for primal 0-forms, or 0-laplace-beltrami

    Can look at equation in two ways;
    B.I * A * x = y, or A * x = B * y
    A and B both map primal-0-forms to dual-n-forms
    We should think of Equation as encoding a 'generalized' linear equation
    makes eigensolve a natural extension
    B need not even be diagonal; but for all currently planned operations it will be trivially invertable


    Notes
    -----
    do we seek to use the 'midpoint' between primal and dual variable encoding as used in eschersque?
    preferably we find a more orthodox solution. generalized linear equation may be just that

    currently sphericalTriangular complexes only; need to add MG operators for regular complexes, at least

    need to think of clear interface to extend this to multicomplex as well.
    what do we truly expect as interface from the complex that we bind to here?
    """

    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def operators(self):
        """Construct laplacian on primal 0-forms

        Such that `A * x = B * y` encodes the poisson problem

        x is taken to be a primal 0-form, and the residual-space r = By-Ax is a dual n-form
        Note that it often feels more natural to associate dual-forms with unknowns,
        and the primal with equations, seeing as how the primal set needs to be completed
        in accordance with the dual boundary, to complete the system

        Returns
        -------
        A : Linear operator
        B : Linear operator
        BI : inverse of B
        """
        S = self.complex.topology.selector[0]   # this term sets boundary to zero
        T01 = self.complex.topology.matrix(0, 1).T
        grad = T01
        div = T01.T
        mass = S * self.complex.hodge_DP[0]
        D1P1 = scipy.sparse.diags(self.complex.hodge_DP[1])

        # construct our laplacian
        A = S * div * D1P1 * grad * S.T
        B = scipy.sparse.diags(mass)
        BI = scipy.sparse.diags(1. / mass)

        return A.tocsr(), B.tocsc(), BI.tocsc()

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
        """Maps 0-form unkonwns to 0-form residual"""
        A, B, BI = self.operators
        return BI * (rhs - A * x)

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
        """jacobi iteration"""
        raise NotImplementedError


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


class Stokes(Equation):
    pass
class Elasticity(Equation):
    pass


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