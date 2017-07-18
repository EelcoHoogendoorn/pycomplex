from pycomplex.complex.cubical import *
from pycomplex.math import linalg
from pycomplex.synthetic import n_cube, n_cube_grid


def test_regular_2():
    complex = n_cube(3)
    dp = complex.dual_position()

    for p in dp:
        print()
        print(p.shape)
        print(p)


def test_quad_3():
    """test of three connected quads"""
    n_dim = 2
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
    # we grab the boundary of the cube
    quads = n_cube(3).boundary().as_23()
    for i in range(2):
        quads = quads.subdivide(smooth=True)
        if i == 0:
            quads.vertices = linalg.normalized(quads.vertices)

    quads.vertices = np.dot(quads.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    # triangular = cube.to_simplicial()
    # triangular.plot_3d()
    quads.plot()


def test_cube():
    """Test 3d cube embedded in 3-space"""
    cube = n_cube(3).as_33()
    # random rotation
    cube.vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    cube.plot()
    for i in range(2):
        cube = cube.subdivide()
    cube.plot(plot_dual=True)


def test_triangulated_cube():
    cube = n_cube(3)
    # cube.vertices = np.dot(cube.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    surface = cube.boundary().as_23()
    # surface = surface.subdivide(smooth=True)
    surface = surface.to_simplicial().smooth()
    surface = surface.subdivide(smooth=True)
    # surface.plot_3d(backface_culling=True)

    # map back to quads again
    surface = surface.to_cubical().smooth()

    # surface = surface.subdivide()
    # surface = surface.subdivide()

    surface.as_23().plot(plot_dual=False)
    assert surface.topology.is_oriented


def test_cube_grid_2():
    grid = n_cube_grid((2, 3))
    grid.as_22().plot()


def test_cube_grid_3():
    grid = n_cube_grid((1, 2, 3))
    grid.vertices = np.dot(grid.vertices, linalg.orthonormalize(np.random.randn(3, 3)))
    grid.as_33().plot(plot_dual=True)
    assert grid.topology.is_oriented


def test_product_2():
    d1 = n_cube_grid((3,))
    d2 = n_cube_grid((5,))

    grid = d1.product(d2)
    grid.as_22().plot()


def test_product_2_1():
    d1 = n_cube_grid((3,))
    d2 = n_cube_grid((5, 4))

    grid = d1.product(d2)
    grid.vertices = np.dot(grid.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    grid.as_33().plot()


test_triangulated_cube()