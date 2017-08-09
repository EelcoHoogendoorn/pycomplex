""""Generate Perlin noise on a sphere

This serves as an illustration of the utility of the ability to diffuse things over a general manifold.
Having such functionality makes it trivial to generalize Perlin noise to arbitrary manifolds,
without any distortion or warping that might otherwise result from texture-mapping type operations
"""

import numpy as np

from pycomplex import synthetic
from pycomplex.math import linalg

from examples.diffusion.explicit import Diffusor
import matplotlib.pyplot as plt


def perlin_noise(complex, octaves):
    """Generate Perlin noise over the given complex

    Parameters
    ----------
    complex : Complex
    octaves : iterable of (sigma, amplitude) tuples

    Returns
    -------
    ndarray
        primal 0-form

    """
    diffusor = Diffusor(complex)

    def normalize(x):
        x -= x.min()
        return x / x.max()
    def level(s, a):
        return normalize(diffusor.integrate_explicit_sigma(np.random.rand(complex.topology.n_elements[0]), s)) ** 1.5 * a
    return normalize(sum(level(*o) for o in octaves))


if __name__ == '__main__':
    sphere = synthetic.icosphere(refinement=5).copy(radius=30)

    field = perlin_noise(
        sphere,
        [
            (.5, .5),
            (1, 1),
            (2, 2),
            (4, 4),
            (8, 8),
        ]
    )

    water_level = 0.42
    field = np.clip(field, water_level, 1)
    # add some bump mapping
    sphere = sphere.copy(vertices=sphere.vertices * (1 + field[:, None] / 5))

    R = [
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ]
    R = linalg.power(R, 0.1)
    for i in range(100):
        sphere = sphere.copy(vertices=np.dot(sphere.vertices, R))
        sphere.as_euclidian().plot_primal_0_form(field, cmap='terrain', plot_contour=False, shading='gouraud')
        plt.show()
