
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


def test_quad_3(show_plot):
    """Test the construction and subdivision of a domain with three connected quads"""
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

    from pycomplex.complex.cubical import ComplexCubical2Euclidian2
    complex = ComplexCubical2Euclidian2(vertices=vertices, cubes=quads)

    for i in range(1):
        complex = complex.subdivide_cubical()
    complex.plot(plot_dual=True)
    show_plot()


def test_2cube_3space(show_plot):
    """Test the surface of a cube embedded in 3-space"""
    quads = n_cube(3, centering=True).boundary.as_23()
    for i in range(3):
        quads = quads.subdivide_cubical(smooth=True)

    quads = quads.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    quads.plot()

    quads.subdivide_simplicial().as_3().plot_3d(plot_vertices=False, backface_culling=True, plot_dual=False)
    show_plot()


def test_cube(show_plot):
    """Test 3d cube embedded in 3-space"""
    cube = n_cube(3)
    cube = cube.transform(linalg.orthonormalize(np.random.randn(3, 3)))

    cube.plot()
    for i in range(2):
        cube = cube.subdivide_cubical()
    cube.boundary.plot(plot_dual=True)

    show_plot()


def test_triangulated_cube(show_plot):
    """This produces some funky patterns"""
    cube = n_cube(3)
    # cube = cube.transform(linalg.orthonormalize(np.random.randn(3, 3)))

    surface = cube.boundary.as_23()
    # surface = surface.subdivide(smooth=True)
    surface = surface.subdivide_simplicial().smooth()
    surface = surface.subdivide_cubical()
    # surface.plot_3d(backface_culling=True)

    # map back to quads again
    surface = surface.subdivide_cubical().smooth()

    surface = surface.subdivide_cubical().smooth()
    # surface = surface.subdivide()

    assert surface.topology.is_oriented
    surface.plot(plot_dual=False)
    show_plot()


def test_cube_grid_2(show_plot):
    """Simple test of a regular 2d grid"""
    grid = n_cube_grid((2, 3))
    grid.plot()
    show_plot()


def test_cube_grid_3(show_plot):
    """Simple test of a regular 3d grid"""
    grid = n_cube_grid((1, 2, 3))
    grid = grid.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    assert grid.topology.is_oriented
    grid.plot(plot_dual=True)
    show_plot()


def test_product_2(show_plot):
    """Test creating a 2d grid as product of 2 1d grids"""
    d1 = n_cube_grid((3,))
    d2 = n_cube_grid((5,))

    grid = d1.product(d2)
    grid.plot()
    show_plot()


def test_product_2_1(show_plot):
    """Test creating a 3d grid as product of 1d and 2d grids"""
    d1 = n_cube_grid((3,))
    d2 = n_cube_grid((2, 4))

    grid = d1.product(d2)

    assert grid.topology.is_oriented
    assert grid.boundary.topology.is_oriented

    grid = grid.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    grid.plot()
    show_plot()


def test_n_cube(show_plot):
    """Test cube grids and their subdivisions in a range of dimensions"""
    for n_dim in [2, 3, 4, 5, 6]:
        cube = n_cube(n_dim).subdivide_cubical()

        assert cube.topology.is_oriented
        # assert cube.topology.is_connected
        assert not cube.topology.is_closed
        # assert cube.topology.boundary.is_connected
        assert cube.topology.boundary.is_oriented
        assert cube.topology.boundary.is_closed

        np.random.seed(1)
        cube = cube.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))
        cube.plot(plot_dual=True)
        show_plot()


def test_transfer(show_plot):
    """Test the direct transfer matrices """
    n_dim = 2
    cube = n_cube(n_dim)
    cube = cube.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    hierarchy = [cube]
    for i in range(2):
        hierarchy.append(hierarchy[-1].subdivide_cubical())

    if True:
        # test that directionality is inherited as expected
        hierarchy[-2].plot(plot_dual=True, plot_arrow=True)
        hierarchy[-1].plot(plot_dual=True, plot_arrow=True)
        show_plot()

    DT = hierarchy[-1].topology.dual.transfer_matrices

    # little check
    for res, c, f in zip(DT, hierarchy[-2].topology.dual.n_elements, hierarchy[-1].topology.dual.n_elements):
        assert res.shape == (f, c)


def test_multigrid():
    """Test the full multigrid transfer operators"""
    # FIXME: test that null space vectors are preserved in a roundtrip
    # FIXME: not sure this makes sense outside of a regular grid context
    n_dim = 2
    cube = n_cube(n_dim)
    cube = cube.transform(linalg.orthonormalize(np.random.randn(n_dim, n_dim)))

    hierarchy = [cube]
    for i in range(2):
        hierarchy.append(hierarchy[-1].subdivide_cubical())

    T = hierarchy[-1].multigrid_transfers
    q = T[0].todense()
    print(q)
    z = T[-1].todense()
    print()
