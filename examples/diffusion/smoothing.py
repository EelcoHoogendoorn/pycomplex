"""Showcase smoothing of a mesh by means of diffusion of the vertex positions over the mesh itself

Simple explicitly integrated diffusion is used here

TODO
----
- Add implicit integrators

"""

import numpy as np
import matplotlib.pyplot as plt

from pycomplex.math import linalg

from examples.diffusion.explicit import Diffusor
from examples.subdivision import letter_a
from examples.util import save_animation


surface = letter_a.create_letter(3).subdivide_simplicial().as_3()
surface = surface.transform(np.identity(3) * 30)
surface = surface.transform(linalg.power(linalg.orthonormalize(np.random.randn(3, 3)), 0.2))
# surface = surface.optimize_weights()

# visualize positivity of metric
plt.hist(surface.dual_metric[1], bins=50)
plt.show()
if False:
    surface.plot(plot_dual=False, plot_vertices=False)


path = r'../output/smoothing_1'

diffusor = Diffusor(surface)

for i in save_animation(path, frames=10, overwrite=True):
    print(i)
    surface.plot_3d(plot_dual=False, plot_vertices=False, backface_culling=True)

    surface = surface.copy(vertices=diffusor.integrate_explicit_sigma(surface.vertices, sigma=1))
