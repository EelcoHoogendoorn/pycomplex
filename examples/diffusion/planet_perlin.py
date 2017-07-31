""""Generate perlin noise on a sphere"""

import numpy as np
from pycomplex import synthetic
from pycomplex.math import linalg


surface = synthetic.icosphere(refinement=5).copy(radius=30)

from examples.diffusion.explicit import Diffusor
diffusor = Diffusor(surface)

def perlin_noise(octaves):
    def normalize(x):
        x -= x.min()
        return x / x.max()
    def level(s, a):
        return normalize(diffusor.integrate_explicit_sigma(np.random.rand(surface.topology.n_elements[0]), s)) ** 1.5 * a
    return normalize(sum(level(*o) for o in octaves))

field = perlin_noise([
    (.5, .5),
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 8),
])

for i in range(100):
    surface = surface.copy(vertices=np.dot(surface.vertices, linalg.power(linalg.orthonormalize(np.random.randn(3, 3)), 0.2)))
    surface.as_euclidian().plot_primal_0_form(field, cmap='terrain', vmin=.42, plot_contour=False, shading='gouraud')
