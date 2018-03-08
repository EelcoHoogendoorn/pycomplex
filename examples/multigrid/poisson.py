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


class Poisson(Equation):
    """Wrap complex with poisson equation methods and state

    Poisson logic for primal 0-forms, or 0-laplace-beltrami

    A * x = B * y, or laplacian(x) = mass(y)

    Notes
    -----
    currently sphericalTriangular complexes only; need to add MG operators for regular complexes, at least

    need to think of clear interface to extend this to multicomplex as well.
    what do we truly expect as interface from the complex that we bind to here?
    """

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
            x[1:] = y[1:] / v[1:]
            return x

        return self._solve_eigen(B * y, poisson)

    # FIXME: mg operators belong to the complexes themselves
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
    def restrictor(self):
        A, B, BI = self.operators
        # FIXME: need a link to parent equation; this is hacky. should be coarse.BI
        coarse = scipy.sparse.diags(self.complex.parent.hodge_PD[0])
        return coarse * normalize_l1(self.transfer.T, axis=0) * B
    def restrict(self, fine):
        """Restrict solution from fine to coarse"""
        return self.restrictor * fine
    @cached_property
    def interpolator(self):
        A, B, BI = self.operators
        coarse = scipy.sparse.diags(self.complex.parent.hodge_DP[0])
        return BI * normalize_l1(self.transfer, axis=0) * coarse
    def interpolate(self, coarse):
        """Interpolate solution from coarse to fine"""
        return self.interpolator * coarse


class Stokes(Equation):
    pass
class Elasticity(Equation):
    pass


class GeometricMultiGrid(object):
    """Perhaps use this to cache the MG-specific data?"""
    def __init__(self, complex, equation):
        self.complex = complex
        self.equation = equation


class Hierarchy(object):
    def __index__(self, hierarchy, equation):
        self.hierarchy = hierarchy
        self.equations = [equation(l) for l in hierarchy]

    def solve(self, y):
        pass

    def as_preconditioner(self):
        pass


if __name__ == '__main__':
    from pycomplex import synthetic
    import matplotlib.pyplot as plt

    sphere = synthetic.icosahedron().copy(radius=30)# .subdivide_fundamental()
    hierarchy = [sphere]
    for i in range(6):
        hierarchy.append(hierarchy[-1].subdivide_loop())
    equations = [Poisson(l) for l in hierarchy]

    if False:
        # test eigen solve; seems to work just fine
        V, v = equations[-1].eigen_basis(K=10, amg=True)
        print(equations[-1].largest_eigenvalue)
        hierarchy[-1].as_euclidian().plot_primal_0_form(V[:, -1])
        plt.show()

    # now test multigrid; what is a good testcase?
    # visually not that obvious; but we can focus on numbers first
    # if we can solve poisson with perlin input using mg,
    # we should be good, since it contains all frequency components

    from examples.diffusion.perlin_noise import perlin_noise
    p0 = perlin_noise(hierarchy[-1])
    p0 -= p0.mean()
    # p0 = np.random.normal(size=p0.size)
    if False:
        vmin, vmax = p0.min(),p0.max()
        hierarchy[-1].as_euclidian().plot_primal_0_form(p0, vmin=vmin, vmax=vmax)

        hierarchy[-2].as_euclidian().plot_primal_0_form(equations[-1].restrict(p0), vmin=vmin, vmax=vmax)
        hierarchy[-1].as_euclidian().plot_primal_0_form(equations[-1].interpolate(equations[-1].restrict(p0)), vmin=vmin, vmax=vmax)

        plt.show()


    from examples.multigrid import multigrid

    from time import clock

    x0 = np.zeros_like(p0)
    print('initial res')
    print(np.linalg.norm(equations[-1].residual(x0, p0)))

    # warm up cache
    x = multigrid.solve_full_cycle(equations, p0)
    t = clock()
    x = multigrid.solve_full_cycle(equations, p0, iterations=2)
    print('mg full time: ', clock() - t)
    print('mg full resnorm', np.linalg.norm(equations[-1].residual(x, p0)))


    x_minres = equations[-1].solve_minres(p0, amg=True)
    t = clock()
    x_minres = equations[-1].solve_minres(p0, amg=True)
    print('minres time: ', clock() - t)
    print('minres resnorm', np.linalg.norm(equations[-1].residual(x_minres, p0)))


    t = clock()
    x_amg = equations[-1].solve_amg(p0)
    print('amg time: ', clock() - t)
    print('amg resnorm', np.linalg.norm(equations[-1].residual(x_amg, p0)))

    # x_eigen = equations[-1].solve(p0)
    # print(np.linalg.norm(equations[-1].residual(x_eigen, p0)))
