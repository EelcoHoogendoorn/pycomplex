
import matplotlib.pyplot as plt

from pycomplex.complex.cubical import *
from pycomplex.math import linalg
from pycomplex.synthetic import n_cube, n_cube_grid


def test_regular_2():
    complex = n_cube(3)
    dp = complex.dual_position

    for p in dp:
        print()
        print(p.shape)
        print(p)


def test_quad_3():
    """test of three connected quads"""
    vertices = np.indices((3, 3))
    vertices = vertices.reshape(2, -1).T[:-1]
    quads = [
        [[0, 1],
         [3, 4]],
        [[1, 2],
         [4, 5]],
        [[3, 4],
         [6, 7]],
    ]

    complex = ComplexCubical2Euclidian2(vertices=vertices, cubes=quads)

    for i in range(1):
        complex = complex.subdivide()
    complex.plot(plot_dual=True)


def test_2cube_3space():
    """Test the surface of a cube embedded in 3-space"""
    quads = n_cube(3, centering=True).boundary.as_23()
    for i in range(3):
        quads = quads.subdivide(smooth=True)

    quads = quads.copy(vertices=np.dot(quads.vertices, linalg.orthonormalize(np.random.randn(3, 3))))
    quads.plot()

    quads.subdivide_simplicial().as_3().plot_3d(plot_vertices=False, backface_culling=True, plot_dual=False)


def test_cube():
    """Test 3d cube embedded in 3-space"""
    cube = n_cube(3)
    # random rotation
    cube = cube.copy(vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(3, 3))))

    cube.plot()
    for i in range(2):
        cube = cube.subdivide()
    cube.boundary().plot(plot_dual=True)


def test_triangulated_cube():
    cube = n_cube(3)
    # cube = cube.copy(vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(3, 3))))

    surface = cube.boundary.as_23()
    # surface = surface.subdivide(smooth=True)
    surface = surface.subdivide_simplicial().smooth()
    surface = surface.subdivide(smooth=True)
    # surface.plot_3d(backface_culling=True)

    # map back to quads again
    surface = surface.subdivide_cubical().smooth()

    surface = surface.subdivide().smooth()
    # surface = surface.subdivide()

    assert surface.topology.is_oriented
    surface.plot(plot_dual=False)
    plt.show()


def test_cube_grid_2():
    grid = n_cube_grid((2, 3))
    grid.plot()


def test_cube_grid_3():
    grid = n_cube_grid((1, 2, 3))
    grid = grid.copy(vertices = np.dot(grid.vertices, linalg.orthonormalize(np.random.randn(3, 3))))
    grid.plot(plot_dual=True)
    assert grid.topology.is_oriented


def test_product_2():
    d1 = n_cube_grid((3,))
    d2 = n_cube_grid((5,))

    grid = d1.product(d2)
    grid.plot()


def test_product_2_1():
    d1 = n_cube_grid((3,))
    d2 = n_cube_grid((2, 4))

    grid = d1.product(d2)

    assert grid.topology.is_oriented
    assert grid.boundary.topology.is_oriented

    grid = grid.copy(vertices = np.dot(grid.vertices, linalg.orthonormalize(np.random.randn(3, 3))))
    grid.plot()


def test_n_cube():
    for n_dim in [2, 3, 4, 5, 6]:
        cube = n_cube(n_dim).subdivide()

        assert cube.topology.is_oriented
        assert cube.topology.is_connected
        assert not cube.topology.is_closed
        assert cube.topology.boundary.is_connected
        assert cube.topology.boundary.is_oriented
        assert cube.topology.boundary.is_closed

        np.random.seed(1)
        cube = cube.copy(vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim))))
        cube.plot(plot_dual=True)

test_n_cube()