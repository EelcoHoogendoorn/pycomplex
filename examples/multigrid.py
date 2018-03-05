"""Attempt at geometric multigrid solver
Start with primal 0-form laplacian; but aim for more general framework
Start by cleaning up escheresque solver
"""

import numpy as np
import scipy.sparse

from pycomplex import synthetic


class Equation(object):
    """Encode operators, but also solver in eigen-space, and how to perform relaxation"""


class GeometricMultiGrid(object):
    def __init__(self, complex, equation):
        self.complex = complex
        self.equation = equation


def laplacian(complex):
    """construct laplacian on primal 0-forms"""
    S = complex.topology.selector[0]
    T01 = complex.topology.matrix(0, 1).T
    grad = T01
    div = T01.T
    mass = S * complex.hodge_DP[0]
    D1P1 = scipy.sparse.diags(complex.hodge_DP[1])

    # construct our laplacian
    laplacian = S * div * D1P1 * grad * S.T
    return laplacian, mass


def eigen_all(complex):
    """Get full 0-laplacian eigenbasis of a small complex

    Returns
    -------
    V : ndarray, [complex.n_vertices, complex.n_vertices]
        n-th column is n-th eigenvector
    v : ndarray, [complex.n_vertices]
        eigenvalues, sorted low to high
    """
    L, M = laplacian(complex)
    M = scipy.sparse.diags(M)
    v, V = scipy.sparse.linalg.lobpcg(A=L, B=M, X=np.random.normal(size=L.shape))
    return V, v


def solve_eigen(eigen, x, func):
    """map a dual n-form eigenspace, and solve func in eigenspace

    used for poisson and diffusion solver
    """
    V, v = eigen
    y = np.einsum('vji,...ji->v', V, x)
    y = func(y, v)
    y = np.einsum('vji,v...->ji', V, y)
    return y


def solve_laplacian_eigen(eigen, rhs):
    """solve poisson linear system in eigenbasis

    Parameters
    ----------
    eigen : tuple with V and v
    rhs : dual n-form

    Returns
    -------
    primal 0-form
    """
    def poisson(x, v):
        y = np.zeros_like(x)
        y[1:] = x[1:] / v[1:]   # poisson linear solve is simple division in eigenspace. skip nullspace
        return y
    return solve_eigen(eigen, rhs, poisson)


def solve_poisson_rec(hierarchy, rhs, x):
    """recursive poisson solver step.

    Parameters
    ----------
    hierarchy: list of complex
    rhs : dual n-form
        right hand side of poisson problem
    x : dual n-form
        current best guess at a solution
    """

    complex = hierarchy[-1]

    #coarse level break
    if len(hierarchy) == 1:
        eigen = eigen_all(complex)
        return solve_laplacian_eigen(eigen, rhs)


    def profile(func):
        """debug output for intermediate steps"""
        def inner(ix):
            ox = func(ix)
            err = np.linalg.norm(complex.poisson_residual(ix, rhs).ravel()) - \
                  np.linalg.norm(complex.poisson_residual(ox, rhs).ravel())
            print('improvement', func.__name__)
            print(err)
            return ox
        return inner

    knots = np.linspace(1, 4, 8, True)  #we need to zero out eigenvalues from largest to factor 4 smaller
##    knots = np.sqrt( (knots-1)) + 1
    def solve_poisson_overrelax(x):
        return complex.poisson_overrelax(x, rhs, knots)

    def solve_poisson_iterate(x, iterations):
        for i in range(iterations):
##            x = complex.jacobi_d2(x, rhs)
            x = complex.poisson_descent(x, rhs)
        return x

    def coarsesmooth(x):
        coarse_complex = hierarchy[-2]
        fine_res = complex.poisson_residual(x, rhs)
        coarse_res = coarse_complex.restrict_d2(fine_res)
        coarse_error = solve_poisson_rec(
            hierarchy[:-1],
            coarse_res,
            np.zeros_like(coarse_res),
            )
        fine_error = coarse_complex.interpolate_d2(coarse_error)
        return x - fine_error      #residual correction scheme

    presmooth    = (solve_poisson_iterate)
    postsmooth   = (solve_poisson_iterate)
    coarsesmooth = (coarsesmooth)

##    x = presmooth(x, 5)
    x = solve_poisson_overrelax(x)
    x = coarsesmooth(x)
##    x = postsmooth(x, 5)
    x = solve_poisson_overrelax(x)

    return x


sphere = synthetic.icosahedron()# .subdivide_fundamental()
hierarchy = [sphere]
V, v = eigen_all(hierarchy[0])

for i in range(3):
    hierarchy.append(hierarchy[-1].subdivide_loop())




    # can we construct a preconditioner such that