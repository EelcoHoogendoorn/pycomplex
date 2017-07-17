"""Test spherical geometry functions"""

import numpy as np

from pycomplex.geometry import spherical


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
