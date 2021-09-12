"""Test spherical geometry functions"""

import numpy as np
import numpy.testing as npt

from pycomplex.geometry import spherical
from pycomplex import synthetic
from pycomplex.math import linalg


def test_circumcenter():
    c = spherical.circumcenter([[1, 0, 0], [0, 0, 1]])      # edge
    print(c)

    c = spherical.circumcenter([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # triangle
    print(c)


def test_unsigned_volume():
    v = spherical.unsigned_volume([[1, 0, 0], [0, 0, 1]])       # edge
    print(v)

    v = spherical.unsigned_volume([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # triangle
    print(v)

    v = spherical.unsigned_volume([[[1, 0, 0]], [[0, 0, 1]]])  # two points
    print(v)


def test_simplex_circumcenter():
    n_dim = 3
    simplex = synthetic.n_simplex(n_dim).boundary.select_subset(np.eye(n_dim+1)[0])

    from pycomplex.geometry.euclidian import circumcenter
    circ_e = linalg.normalized(circumcenter(simplex.vertices))

    circ = spherical.triangle_circumcenter(simplex.vertices)

    npt.assert_allclose(circ, circ_e)
