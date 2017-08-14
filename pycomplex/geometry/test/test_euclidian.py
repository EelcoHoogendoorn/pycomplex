
import numpy as np

from pycomplex.geometry import euclidian


def test_circumcenter_barycentric_weighted():
    c = euclidian.circumcenter_barycentric_weighted([[0], [4]], weights=[2, 0])
    print(c)

    c = euclidian.circumcenter_barycentric_weighted([[0, 0], [3, 0], [0, 4]], weights=[0, 2, 2])  # triangle in 2D
    print(c)
test_circumcenter_barycentric_weighted()


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
