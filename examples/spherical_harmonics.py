"""Example script to compute and visualize spherical harmonics"""

import numpy as np
import scipy.sparse.linalg

import pycomplex.synthetic


def sparse_diag(diag):
    s = len(diag)
    i = np.arange(s)
    return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))

# construct a spherical complex
sphere = pycomplex.synthetic.icosphere(refinement=4)
sphere.metric()

# grab all the operators we will be needing
T01 = sphere.topology.matrix(0, 1).T
grad = T01
div = T01.T
mass = sphere.P0D2

# construct our laplacian
laplacian = div * sparse_diag(sphere.D1P1) * grad
# solve for some eigenvectors
w, v = scipy.sparse.linalg.eigsh(laplacian, M=sparse_diag(mass), which='SA', k=20)

print(w)
# plot a spherical harmonic
sphere.as_euclidian().plot_primal_0_form(v[:, -1])
