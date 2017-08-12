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


surface = letter_a.create_letter(3).to_simplicial().as_3()
surface = surface.copy(vertices=surface.vertices * 30)
surface = surface.copy(vertices=np.dot(surface.vertices, linalg.power(linalg.orthonormalize(np.random.randn(3, 3)), 0.2)))

if False:
    surface.plot(plot_dual=False, plot_vertices=False)


path = r'c:\development\examples\smoothing_1'

for i in save_animation(path, frames=10, overwrite=True):

    surface.plot_3d(plot_dual=False, plot_vertices=False, backface_culling=True)

    diffusor = Diffusor(surface)
    surface = surface.copy(vertices=diffusor.integrate_explicit_sigma(surface.vertices, sigma=1))
