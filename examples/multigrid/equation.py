
from cached_property import cached_property
from abc import abstractmethod

import numpy as np
import scipy
import scipy.sparse

import pycomplex


class GeneralizedEquation(object):
    """Generalized linear equation of the form A x = B y"""
    def __init__(self, complex):
        self.complex = complex

    @cached_property
    def operators(self):
        """Return A and B"""
        raise NotImplementedError

    @cached_property
    def null_space(self):
        """Return null space, if known a-priori"""
        return None

    @cached_property
    def A(self):
        return self.operators[0]
    @cached_property
    def B(self):
        return self.operators[1]
    @cached_property
    def BI(self):
        return pycomplex.sparse.inv_dia(self.B)

    def solve(self, y):
        """Solve system exactly. most efficient way may be problem-dependent"""
        raise NotImplementedError

    def solve_dense(self, y):
        """Solve system using dense linear algebra; for debugging purposes, mostly"""
        raise NotImplementedError


class SymmetricEquation(GeneralizedEquation):
    """Encode operators and other defining information about a system of equations

    We consider here generalized linear problems of the form Ax=By,
    where A is symmetric, and B diagonal

    Notes
    -----
    Seems like there is overlap in intention with BlockSystem;
    not sure yet what the cleanest conceptual breakup is
    BlockSystem manages blocks;
    this defines the core operation (self.operators),
    and provides methods aimed at interfacing with mg-solver
    defining the operation is prob better left to the Block class

    First lets focus on getting the poisson problem to work.
    Generalizing to block systems can be done guided by stokes problem

    Would be nice to work in petrov-galerkin MG as well
    that is, construct coarse linear system from transfer operators
    alternative is rediscretization of anisotropies and bcs;
    feels kinda messy
    """


    @cached_property
    def largest_eigenvalue(self):
        """Get largest eigenvalue for `A x = B x`

        This is useful to compute the maximum effective stepsize for iterative procedures

        Returns
        -------
        float
            largest eigenvalue of the linear equation
        """
        A, B, BI = self.operators
        # X = scipy.rand(A.shape[0], 1)
        # tol=1e-9
        # v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, largest=True)
        # return v[0]
        return scipy.sparse.linalg.eigsh(A, M=B, k=1, which='LM', tol=1e-6, return_eigenvectors=False)[0]

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
        A, B, BI = self.operators
        return self.eigen_basis(K=A.shape[0])

    @cached_property
    def amg_solver(self):
        """Get AMG preconditioner for the action of A in isolation"""
        from pyamg import smoothed_aggregation_solver
        A, B, BI = self.operators
        return smoothed_aggregation_solver(A, B=self.null_space)

    def eigen_basis(self, K, preconditioner=None, tol=1e-10, nullspace=None):
        """Compute partial eigen decomposition for `A V = v B V`

        Parameters
        ----------
        K : int
            number of eigenvectors
        preconditioner : optional
            if `amg`, amg preconditioning is used
        tol : float

        Returns
        -------
        V : ndarray, [A.shape, K]
            n-th column is n-th eigenvector
        v : ndarray, [K]
            eigenvalues, sorted low to high
        """
        A, B, BI = self.operators
        if preconditioner == 'amg':
            M = self.amg_solver.aspreconditioner()
        else:
            M = preconditioner

        X = np.random.rand(A.shape[0], K)

        # monkey patch this dumb assert
        from scipy.sparse.linalg.eigen.lobpcg import lobpcg
        lobpcg._assert_symmetric = lambda x: None

        try:
            if nullspace is None:
                Y = self.null_space     # giving lobpcg the null space helps with numerical stability
            if nullspace is False:
                Y = None
            v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, M=M, Y=Y, largest=False)
        except:
            v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, M=M, largest=False)

        return V, v

    def _solve_eigen(self, y, func):
        """Solve equation in eigenspace

        Parameters
        ----------
        y : ndarray, [n, ...], float
            right hand side
        func : callable
            maps y to x

        Returns
        -------
        x : ndarray, [n, ...], float
            solution

        Notes
        -----
        Usefull for for poisson and diffusion solver, and probably others too
        """
        V, v = self.complete_eigen_basis
        y = np.einsum('vi,i...->v...', V, y)
        x = func(y, v)
        x = np.einsum('vi,v...->i...', V, x)
        return x

    def solve_minres(self, y, x0=None, preconditioner=None, tol=1e-6):
        """Solve equation using minres"""
        A, B, BI = self.operators
        if preconditioner == 'amg':
            M = self.amg_solver.aspreconditioner()
        else:
            M = preconditioner
        return scipy.sparse.linalg.minres(A=A, b=B * y, M=M, x0=x0, tol=tol)[0]

    def restrict(self, fine):
        raise NotImplementedError
    def interpolate(self, coarse):
        raise NotImplementedError

    def solve_mg(self, y):
        raise NotImplementedError
        from examples.multigrid import multigrid
        multigrid.solve_full_cycle()

    def solve_amg(self, y):
        A, B, BI = self.operators
        return self.amg_solver.solve(b=B * y)

    def residual(self, x, y):
        """Computes residual; should BI be here? it should for richardson iteration"""
        A, B, BI = self.operators
        return BI * (A * x - B * y)
        # return q - q.mean()

    def richardson(self, x, y, relaxation=1):
        """Simple residual descent. Not a great overall solver, but a pretty decent smoother

        Parameters
        ----------
        x : ndarray
        y : ndarray
        relaxation : float
            if 1, stepsize is such that largest eigenvalue of the solution goes to zero in one step
            values > 1 denote overrelaxation
        """
        return x - self.residual(x, y) * (relaxation / self.largest_eigenvalue / 2)

    @cached_property
    def inverse_diagonal(self):
        A, B, BI = self.operators
        D = A.diagonal()
        assert np.all(D > 0)
        return scipy.sparse.diags(1. / D)

    def jacobi(self, x, y, relaxation=1):
        """Jacobi iteration.

        Notes
        -----
        More efficient than Richardson?
        Scaling with an equation-specific factor might indeed adapt better to anisotropy
        Requires presence of nonzero diagonal on A, which richardson does not
        but unlike richardson, zero mass diagonal terms are fine
        """
        A, B, BI = self.operators
        R = (A * x - B * y)
        return x - self.inverse_diagonal * R * (relaxation / 2)

    def overrelax(self, x, y, knots):
        """overrelax, forcing the eigencomponent to zero at the specified overrelaxation knots

        Parameters
        ----------
        x : ndarray
        y : ndarray
        knots : List[float]
            for interpretation, see self.descent
        """
        for s in knots:
            x = self.jacobi(x, y, s)
        return x

    def smooth(self, x, y):
        """Basic smoother; inspired by time integration of heat diffusion equation"""
        knots = np.linspace(1, 4, 3, endpoint=True)
        return self.overrelax(x, y, knots)


class NormalEquation(GeneralizedEquation):
    """Set of equations intended to be solved in a least-square sense,
    or through formulation of their normal equations

    A x = B y

    A.T A x = A.T B y

    Where A can be any matrix, and B must be diagonal

    What does it mean to take an eigenvector of the normal equations?
    and are the eigenvectors we obtain identical to those of a 'true' laplacian?
    """

    @cached_property
    def normal(self):
        """Return explicit normal equations"""
        A, B = self.operators
        An = A.T * A
        Bn = A.T * B
        e = SymmetricEquation(self.complex)
        e.operators = An, Bn
        e.parent = self
        return e

    def jacobi(self, x, y, relaxation=1):
        """Jacobi iteration on the normal equations."""
        R = self.residual(x, y)
        return x - self.normal.inverse_diagonal * (self.A.T * R) * (relaxation / 2)

    def residual(self, x, y):
        """Computes residual"""
        return self.A * x - self.B * y

    @cached_property
    def normal_operator(self):
        """Linear operator representing A.T * A"""
        def inner(x):
            return self.A.T * (self.A * x)
        return scipy.sparse.linalg.LinearOperator(
            shape=self.B.shape,
            matvec=inner
        )

    def solve_minres(self, y, x0=None, preconditioner=None, tol=1e-6):
        """Solve normal equation using minres


        """
        # if preconditioner == 'amg':
        #     M = self.amg_solver.aspreconditioner()
        # else:
        M = preconditioner
        return scipy.sparse.linalg.minres(
            A=self.normal_operator,
            b=self.A.T * (self.B * y),
            M=M,
            x0=x0, tol=tol
        )[0]

    def solve_lsqr(self, y, x0=None, tol=1e-6):
        """Solve iteratively in least square sense. """
        return scipy.sparse.linalg.lsqr(
            A=self.A,
            b=self.B * y,
            x0=x0, tol=tol
        )[0]

    def eigen_basis(self, K):
        """Obtain solutions of the form

        A V = v B V

        does this even make sense? not really for nonsquare...
        yet would still expect eigs to be usefull in solving
        """