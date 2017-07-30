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
    mass = complex2.P0D2

    # construct our laplacian
    laplacian = div * sparse_diag(complex2.D1P1) * grad
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(laplacian, M=sparse_diag(mass), which='SA', k=20)
    print(w)
    return v


def get_harmonics_2(complex2):
    # grab all the operators we will be needing
    T12 = complex2.topology.matrix(1, 2).T
    grad = T12.T
    div = T12
    mass = complex2.D0P2

    # construct our laplacian
    laplacian = div * sparse_diag(complex2.P1D1) * grad
    # solve for some eigenvectors
    w, v = scipy.sparse.linalg.eigsh(laplacian, M=sparse_diag(mass), which='SA', k=20)
    print(w)
    return v