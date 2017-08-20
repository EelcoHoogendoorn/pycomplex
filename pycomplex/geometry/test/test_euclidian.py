
import numpy as np

from pycomplex import synthetic
from pycomplex.geometry import euclidian


def test_circumcenter_construction():

    for n in [2, 3, 4]:
        simplex = synthetic.n_simplex(n)
        print(euclidian.circumcenter_construction(simplex.vertices))

    p = [[0, 0], [3, 0], [0, 4]]
    c = euclidian.circumcenter_construction(p, weights=[0**2, 3**2, 4**2])  # triangle in 2D
    print(c)

    p = [[0, 0], [3, 0], [0, 4]]
    c = euclidian.circumcenter_construction(p)  # triangle in 2D
    print(c)

    p = [[0], [4]]
    c = euclidian.circumcenter_construction(p, [16, 16])  # triangle in 2D
    print(c)

    p = [[0, 0], [0, 4]]
    c = euclidian.circumcenter_construction(p, [4, 0])  # triangle in 2D
    print(c)


def test_circumcenter_barycentric_weighted():
    # for d in np.linspace(0, 16, num=11, endpoint=True):
    #     p = [[0], [4]]
    #     # euclidian.circumcenter_construction(p, [d, 0])
    #     c = euclidian.circumcenter_barycentric_weighted(p, weights=[d, 0])
    #     print(c)
    #     print(np.einsum('ji,j->i', p, c))

    # line in 2d; exact same situation
    print()
    # diag = np.sqrt(2*4**2)
    p = [[0, 0], [0, 4]]
    c = euclidian.circumcenter_barycentric_weighted(p, weights=[0, 16])
    print(c)
    print(np.einsum('ji,j->i', p, c))

    print()
    p = [[0, 0], [3, 0], [0, 4]]
    # euclidian.circumcenter_construction(p, [0, 3, 4])
    c = euclidian.circumcenter_barycentric_weighted(p, weights=None)  # triangle in 2D

    print(c)
    print(np.einsum('ji,j->i', p, c))
    c = euclidian.circumcenter_barycentric_weighted(p, weights=[0, 3**2, 4**2])  # triangle in 2D
    print(c)
    print(np.einsum('ji,j->i', p, c))
    c = euclidian.circumcenter_barycentric(p, weights=[0, 3**2, 4**2])  # triangle in 2D
    print(c)
    print(np.einsum('ji,j->i', p, c))

    c = euclidian.circumcenter_barycentric_weighted(p, weights=[-(1**2), 2**2, 3**2])  # triangle in 2D
    print(c)
    print(np.einsum('ji,j->i', p, c))
    c = euclidian.circumcenter_barycentric_weighted(p, weights=[0**2+1, 3**2+1, 4**2+1])  # triangle in 2D
    print(c)
    print(np.einsum('ji,j->i', p, c))


def test_circumcenter_barycentric():
    c = euclidian.circumcenter_barycentric([[0], [4]])
    print(c)

    c = euclidian.circumcenter_barycentric([[0, 0], [4, 0], [0, 4]])  # triangle in 2D
    print(c)

    c = euclidian.circumcenter_barycentric(
        [
            [[0, 0], [4, 0], [0, 4]],
            [[0, 0], [2, 1], [0, 4]]
        ])  # triangles in 2D
    print(c)

    c = euclidian.circumcenter_barycentric([[0, 0], [4, 0], [0, 4], [4, 4]])  # quad in 2D
    print(c)

    # (part of) circle
    angles = np.linspace(0, np.pi*2, 100, endpoint=False)
    p = np.array([np.cos(angles), np.sin(angles)]).T
    c = euclidian.circumcenter_barycentric(p)  # quad in 2D
    print(c)


def test_circumcenter():
    c = euclidian.circumcenter([[0], [4]])
    print()
    print(c)

    c = euclidian.circumcenter(
        [
            [[0, 0], [4, 0], [0, 4]],
            [[0, 0], [2, 1], [0, 4]]
        ])  # triangles in 2D
    print()
    print(c)

    c = euclidian.circumcenter(
        [
            [[0, 0, 10], [4, 0, 10], [0, 4, 10]],
            [[0, 0, 10], [2, 1, 10], [0, 4, 10]]
        ])  # triangles in 3D; get in-plane circumcenter
    print()
    print(c)


def test_volume():
    v = euclidian.unsigned_volume([[0], [4]])
    print()
    print(v)

    v = euclidian.unsigned_volume(
        [
            [[0, 0], [4, 0], [0, 4]],
            [[0, 0], [2, 1], [0, 4]]
        ])  # triangles in 2D
    print(v)


def test_triangle_angles():
    tris = [
        [
            [4, 0],
            [0, 5],
            [0, 0],
        ],
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    ]
    angles = euclidian.triangle_angles(tris)
    print(angles)

    # check that it works in 3d just the same
    tri = np.random.randn(3, 3)
    angles = euclidian.triangle_angles(tri)
    print(angles)


def test_normals():
    corners = [[0], [1]]
    g = euclidian.simplex_normals(corners)
    print(g)

    corners = [[0, 0], [0, 3], [4, 0]]
    g = euclidian.simplex_normals(corners)

    print(g)


def test_gradients():
    corners = [[0], [1]]
    g = euclidian.simplex_gradients(corners)
    print(g)

    corners = [[0, 0], [0, 3], [4, 0]]
    g = euclidian.simplex_gradients(corners)

    print(g)


test_gradients()