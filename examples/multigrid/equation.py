import numpy as np
import scipy.sparse
from cached_property import cached_property


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

    def __init__(self, complex):
        self.complex = complex

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
        """Get AMG preconditioner for the ation of A in isolation"""
        from pyamg import smoothed_aggregation_solver
        A, B, BI = self.operators
        return smoothed_aggregation_solver(A)

    def eigen_basis(self, K, amg=False, tol=1e-10):
        """Compute partial eigen decomposition for `A x = B x`

        Parameters
        ----------
        K : int
            number of eigenvectors
        amg : bool
            if true, amg preconditioning is used
        tol : float

        Returns
        -------
        V : ndarray, [A.shape, K]
            n-th column is n-th eigenvector
        v : ndarray, [K]
            eigenvalues, sorted low to high
        """
        A, B, BI = self.operators
        X = scipy.rand(A.shape[0], K)
        M = self.amg_solver.aspreconditioner() if amg else None
        v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, M=M, largest=False)
        return V, v

    def _solve_eigen(self, y, func):
        """solve func in eigenspace

        usefull for for poisson and diffusion solver, and probably others too
        """
        V, v = self.complete_eigen_basis
        y = np.einsum('vi,i...->v...', V, y)
        x = func(y, v)
        x = np.einsum('vi,v...->i...', V, x)
        return x

    def solve_minres(self, y, x0=None, amg=False, tol=1e-6):
        """Solve equation using minres"""
        A, B, BI = self.operators
        M = self.amg_solver.aspreconditioner() if amg else None
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
        """Computes residual; should BI be here?"""
        A, B, BI = self.operators
        return BI * (A * x - B * y)

    def descent(self, x, y, relaxation=1):
        """Simple residual descent. Not a great overall solver, but a pretty decent smoother

        This is also known as Richardson iteration

        Parameters
        ----------
        x : ndarray
        y : ndarray
        relaxation : float
            if 1, stepsize is such that largest eigenvalue of the solution goes to zero in one step
            values > 1 denote overrelaxation
        """
        return x - self.residual(x, y) * (relaxation / self.largest_eigenvalue / 2)

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
            x = self.descent(x, y, s)
        return x

    def smooth(self, x, y):
        """Basic smoother; inspired by time integration of heat diffusion equation"""
        knots = np.linspace(1, 4, 3, endpoint=True)
        return self.overrelax(x, y, knots)

    def jacobi(self, x, rhs):
        """Jacobi iteration. Potentially more efficient than our current iteration scheme"""
        raise NotImplementedError