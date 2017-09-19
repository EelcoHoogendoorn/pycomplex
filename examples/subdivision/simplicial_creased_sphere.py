"""Demonstrate 2d triangle surface embedded in 3-space with creased edges"""

import numpy as np
np.random.seed(21)

from pycomplex import synthetic
from pycomplex.math import linalg
import matplotlib.pyplot as plt


sphere = synthetic.icosahedron().as_euclidian()
# map sphere complex to triangle complex
sphere = sphere.transform(linalg.orthonormalize(np.random.randn(3, 3)))

# set every third edge to be crease
crease1 = sphere.topology.range(1) % 3 == 0
crease0 = np.abs(sphere.topology.matrix(0, 1) * crease1) == 1

for i in range(3):
    sphere = sphere.subdivide_loop(smooth=True, creases={0: crease0, 1: crease1})
    crease1 = sphere.topology.transfer_matrices[1] * crease1
    crease0 = sphere.topology.transfer_matrices[0] * crease0

sphere.plot_3d(backface_culling=True, plot_dual=False)
plt.show()
