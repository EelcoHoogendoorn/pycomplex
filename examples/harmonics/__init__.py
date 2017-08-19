"""Generate harmonic functions"""

# FIXME: generate this to full nd

import numpy as np
import scipy.sparse


def get_harmonics_0(complex2):
    # grab all the operators we will be needing
    T01 = complex2.topology.matrix(0, 1).T
    grad = T01
    div = T01.T
    mass = complex2.hodge_DP[0]

    # construct our laplacian
    laplacian = div * scipy.sparse.diags(complex2.hodge_DP[1]) * grad
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(laplacian, M=scipy.sparse.diags(mass), which='SA', k=20)
    # print(w)
    return v


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


def get_harmonics_2(complex2):
    # grab all the operators we will be needing
    T12 = complex2.topology.matrix(1, 2).T
    grad = T12.T
    div = T12
    n = complex2.topology.n_dim
    mass = complex2.hodge_DP[n]

    # construct our laplacian
    laplacian = div * scipy.sparse.diags(complex2.hodge_PD[n-1]) * grad
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(laplacian, M=scipy.sparse.diags(mass), which='SA', k=20)
    # print(w)
    return v

