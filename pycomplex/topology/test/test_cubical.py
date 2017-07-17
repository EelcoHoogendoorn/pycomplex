import numpy as np

from pycomplex.synthetic import n_cube
from pycomplex.topology.cubical import TopologyCubical, generate_boundary
from pycomplex.topology.test.test_base import test_basic


def test_generate_boundary():
    n_dim = 3
    cube = n_cube(n_dim)

    edges = generate_boundary(cube.topology.elements[-1], degree=1)
    print(edges.shape)
    print(edges)


def test_cube():
    n_dim = 3

    n_cubes = np.arange(2**n_dim).reshape((1,)+(2,)*n_dim)

    topology = TopologyCubical.from_cubes(n_cubes)

    test_basic(topology)


def test_quads():
    """
    0 1 2
    3 4 5
    6 7
    """
    n_dim = 2
    quads = [
        [[0, 1], [3, 4]],
        [[1, 2], [4, 5]],
        [[3, 4], [6, 7]],
    ]

    topology = TopologyCubical.from_cubes(quads)

    test_basic(topology)
    b = topology.boundary()
    dual = topology.dual()


def test_product():
    quad = n_cube(2)
    c = quad.topology.extrude()
    print(c.elements)