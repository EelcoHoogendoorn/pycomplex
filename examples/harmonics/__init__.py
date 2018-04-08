"""Generate harmonic functions"""

# FIXME: generalize this to full nd
# FIXME: merge with other code bases; like Equation object

import numpy as np
import scipy.sparse

# monkey patch this dumb assert
from scipy.sparse.linalg.eigen.lobpcg import lobpcg
lobpcg._assert_symmetric = lambda x: None


def get_harmonics_0(complex2, zero_boundary=False, amg=False, K=20, tol=1e-6):
    """primal 0-form harmonics with zero boundary"""
    # grab all the operators we will be needing
    S = complex2.topology.selector_interior[0]
    if zero_boundary is False:
        S = scipy.sparse.eye(complex2.topology.n_elements[0])
    T01 = complex2.topology.matrix(0, 1).T
    grad = T01
    div = T01.T
    mass = S * complex2.hodge_DP[0]

    # construct our laplacian
    laplacian = S * div * scipy.sparse.diags(complex2.hodge_DP[1]) * grad * S.T

    B = scipy.sparse.diags(mass).tocsc()
    if not amg:
        W, V = scipy.sparse.linalg.eigsh(
            laplacian.tocsc(), M=B, which='SA', k=K, tol=tol)
    else:
        # create the AMG hierarchy
        from pyamg import smoothed_aggregation_solver
        ml = smoothed_aggregation_solver(laplacian)
        # initial approximation to the K eigenvectors
        X = scipy.rand(laplacian.shape[0], K)
        # preconditioner based on ml
        M = ml.aspreconditioner()
        # compute eigenvalues and eigenvectors with LOBPCG
        Y = np.ones((laplacian.shape[0], 1))
        print(np.abs(laplacian * Y).max())
        # Y = None
        W, V = scipy.sparse.linalg.lobpcg(laplacian.tocsr(), X, B=B, tol=tol, M=M, Y=Y, largest=False)

    print(W)
    return S.T * V


def get_harmonics_1(complex2):
    # grab all the operators we will be needing
    T01, T12 = complex2.topology.matrices
    P1P0 = T01.T
    D2D1 = T01
    P2P1 = T12.T
    D1D0 = T12

    P0D2, P1D1, P2D0 = [scipy.sparse.diags(h) for h in complex2.hodge_PD]
    D2P0, D1P1, D0P2 = [scipy.sparse.diags(h) for h in complex2.hodge_DP]

    # construct our laplacian-beltrami
    L = D1D0 * D0P2 * P2P1 - D1P1 * P1P0 * P0D2 * D2D1 * D1P1
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(L, M=D1P1, which='SA', k=20)
    return v


def get_harmonics_2(complex2, amg=False, K=20, tol=1e-6):
    """Solve for dual-0-form harmonics. boundary implicitly zero for now"""
    # grab all the operators we will be needing
    T12 = complex2.topology.matrix(1, 2).T
    grad = T12.T
    div = T12
    n = complex2.topology.n_dim
    mass = complex2.hodge_PD[n]

    # construct our laplacian
    laplacian = (div * scipy.sparse.diags(complex2.hodge_PD[n-1]) * grad).tocsr()
    scipy.sparse.diags(mass).tocsc()

    # solve for some eigenvectors
    B = scipy.sparse.diags(mass).tocsc()
    if not amg:
        W, V = scipy.sparse.linalg.eigsh(laplacian, M=B, which='SA', k=20)
    else:
        # create the AMG hierarchy
        from pyamg import smoothed_aggregation_solver
        ml = smoothed_aggregation_solver(laplacian)
        # initial approximation to the K eigenvectors
        X = scipy.rand(laplacian.shape[0], K)
        # preconditioner based on ml
        M = ml.aspreconditioner()
        Y = np.ones((laplacian.shape[0], 1))
        # compute eigenvalues and eigenvectors with LOBPCG
        W, V = scipy.sparse.linalg.lobpcg(laplacian.tocsr(), X, B=B, tol=tol, M=M, Y=Y, largest=False)

    # print(W)
    return V

