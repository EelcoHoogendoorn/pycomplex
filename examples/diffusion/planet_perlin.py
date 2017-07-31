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

water_level = 0.42
field = np.clip(field, water_level, 1)
# add some bump mapping
surface = surface.copy(vertices=surface.vertices * (1 + field[:, None] / 5))

R = [
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
]
R = linalg.power(R, 0.1)
for i in range(100):
    surface = surface.copy(vertices=np.dot(surface.vertices, R))
    surface.as_euclidian().plot_primal_0_form(field, cmap='terrain', plot_contour=False, shading='gouraud')
