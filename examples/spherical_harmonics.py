"""Example script to compute spherical harmonics"""

import numpy as np
import scipy.sparse.linalg

import pycomplex.synthetic

# construct a spherical complex
sphere = pycomplex.synthetic.icosphere(refinement=3)

# grab all the operators we will be needing
T01 = sphere.topology.matrix(0, 1)
grad = T01
div = T01.T
mass = sphere.P0D2
laplacian = div * sphere.D1P1 * grad

v, w = scipy.sparse.linalg.eigsh(laplacian, M=mass, which='SM')
print(v)