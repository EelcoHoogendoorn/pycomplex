import numpy as np
import scipy.sparse


def sparse_diag(diag):
    s = len(diag)
    i = np.arange(s)
    return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))


# FIXME: need to make common laplacians centrally available somewhere
def get_harmonics_0(complex2):
    # grab all the operators we will be needing
    T01 = complex2.topology.matrix(0, 1).T
    grad = T01
    div = T01.T
    mass = complex2.hodge_PD[0]

    # construct our laplacian
    laplacian = div * sparse_diag(complex2.hodge_DP[1]) * grad
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(laplacian, M=sparse_diag(mass), which='SA', k=20)
    print(w)
    return v


def get_harmonics_1(complex2):
    # grab all the operators we will be needing
    T01, T12 = complex2.topology.matrices
    P1P0 = T01.T
    D2D1 = T01
    P2P1 = T12.T
    D1D0 = T12

    n = complex2.topology.n_dim

    P0D2, P1D1, P2D0 = [sparse_diag(h) for h in complex2.hodge_PD]
    D2P0, D1P1, D0P2 = [sparse_diag(h) for h in complex2.hodge_DP]

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
    laplacian = div * sparse_diag(complex2.hodge_PD[n-1]) * grad
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(laplacian, M=sparse_diag(mass), which='SA', k=20)
    print(w)
    return v