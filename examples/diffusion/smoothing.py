"""Showcase smoothing of a mesh by means of diffusion of the vertex positions over the mesh itself

Simple explicitly integrated diffusion is used here

TODO
----
- Add implicit integrators
- Add animation
- Use the two above to showcase letter morphing into torus

"""

import numpy as np

from pycomplex.math import linalg
from examples.diffusion.explicit import Diffusor
from examples.subdivision import letter_a

surface = letter_a.create_letter(3).to_simplicial().as_3()
surface.vertices *= 30
surface.metric()


assert surface.topology.is_oriented
print(surface.topology.n_elements)
if False:
    surface.plot(plot_dual=False, plot_vertices=False)

diffusor = Diffusor(surface)
surface.vertices = diffusor.integrate_explicit_sigma(surface.vertices, sigma=3)


for i in range(100):
    surface.vertices = np.dot(surface.vertices, linalg.power(linalg.orthonormalize(np.random.randn(3, 3)), 0.2))
    surface.plot_3d(plot_dual=False, plot_vertices=False, backface_culling=True)
