
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.complex.simplicial.spherical import ComplexSpherical2
from pycomplex.math import linalg


def test_single():
    """Test a single spherical triangle"""
    sphere = ComplexSpherical2(vertices=np.eye(3), simplices=[[0, 1, 2]])
    sphere = sphere.subdivide_loop()
    fig, ax = plt.subplots(1, 1)
    sphere.plot(ax=ax)
    for i in range(1):
        sphere = sphere.subdivide_loop()
        sphere.plot(ax=ax, primal_color='c', dual_color='m')
    plt.show()


def test_icosahedron():
    """Test a full icosahedron"""
    sphere = synthetic.icosahedron()
    sphere = sphere.copy(vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3))))
    fig, ax = plt.subplots(1, 1)
    sphere.plot(ax=ax, backface_culling=True)
    for i in range(1):
        sphere = sphere.subdivide_loop()
        sphere.plot(ax=ax, primal_color='c', dual_color='m', backface_culling=True)
    plt.show()


def test_icosahedron_subset():
    """Test that a concave boundary works just the same on a sphere"""
    sphere = synthetic.icosahedron()
    sphere = sphere.copy(vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3))))
    triangle_position = sphere.primal_position[2]
    selection = triangle_position[:, 2] != triangle_position[:,2].max()
    sphere = sphere.select_subset(selection)
    sphere = sphere.subdivide_loop()
    sphere = sphere.subdivide_loop()

    sphere.plot(plot_dual=True, backface_culling=True)


def test_subdivide():
    """Test if subdivision works well for big triangles up to 90deg angle too"""
    sphere = ComplexSpherical2(vertices=linalg.normalized(np.eye(3)), simplices=[[0, 1, 2]])
    sphere = sphere.subdivide_loop()
    sphere = sphere.subdivide_loop()
    sphere = sphere.subdivide_loop()
    sphere.plot(plot_dual=True)


def test_tetrahedron():
    n_dim = 3
    tet = synthetic.n_simplex(n_dim).boundary.as_spherical().as_2()
    tet = tet.fix_orientation()
    tet = tet.copy(vertices = np.dot(tet.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim))))
    for i in range(0):      # subdivision on a tet gives rather ugly tris
        tet = tet.subdivide_loop()
    tet.plot(backface_culling=True, plot_dual=True)


def test_circle():
    n_dim = 2
    circle = synthetic.n_simplex(n_dim).boundary.as_spherical()
    # circle.topology = circle.topology.fix_orientation()
    circle.plot(backface_culling=False, plot_dual=True)


def test_hexacosichoron():

    complex = synthetic.hexacosichoron()

    deg = complex.topology.degree[0]
    npt.assert_equal(deg, 20)
    assert complex.topology.is_oriented
    assert complex.topology.n_elements == [120, 720, 1200, 600]

    edges = complex.topology.elements[1]
    edges = complex.vertices[edges]
    length = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1)
    npt.assert_allclose(length, length[0])

    n_dim = complex.n_dim
    complex = complex.copy(vertices = np.dot(complex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim))))
    complex.plot(plot_dual=True, backface_culling=False)


def test_n_cube_dual():
    for n_dim in [2, 3, 4, 5]:
        complex = synthetic.n_cube_dual(n_dim)

        complex = complex.copy(vertices = np.dot(complex.vertices, linalg.orthonormalize(np.random.randn(n_dim, n_dim))))
        complex.plot(plot_dual=True, backface_culling=n_dim==3)


def test_picking():
    for n_dim in [2, 3, 4, 5]:
        sphere = synthetic.n_cube_dual(n_dim)
        points = linalg.normalized(np.random.randn(10, n_dim))
        simplex_idx, bary = sphere.pick_primal(points)
        assert np.alltrue(bary >= 0)    # this could fail for points on a boundary
        assert np.allclose(bary.sum(axis=1), 1)
        simplex_idx, bary = sphere.pick_primal(points, simplex_idx)
        assert np.alltrue(bary >= 0)    # this could fail for points on a boundary
        assert np.allclose(bary.sum(axis=1), 1)


def test_picking_alt():
    for n_dim in [2, 3, 4, 5]:
        sphere = synthetic.n_cube_dual(n_dim)

        points = linalg.normalized(np.random.randn(10, n_dim))

        sphere.pick_fundamental(points)

        simplex_idx, bary = sphere.pick_primal(points)
        simplex_idx_alt, bary_alt = sphere.pick_primal_alt(points)

        npt.assert_equal(simplex_idx, simplex_idx_alt)
        npt.assert_allclose(bary, bary_alt)


def test_picking_alt_visual():
    for n_dim in [3]:
        sphere = synthetic.optimal_delaunay_sphere(100, n_dim, iterations=20, push_iterations=20, condition=None)
        assert sphere.topology.is_oriented
        # sphere = sphere.copy(weights = np.random.uniform(0, 0.05, 400))
        # sphere = sphere.optimize_weights()
        sphere = sphere.copy(weights=None).optimize_weights_metric()
        print(sphere.is_well_centered)
        print(sphere.is_pairwise_delaunay)

        p = np.linspace(-1, +1, 1024, endpoint=True)
        x, y = np.meshgrid(p, p)
        r2 = x ** 2 + y ** 2
        mask = r2 < 1
        z = np.sqrt(1 - np.clip(r2, 0, 1))

        points = np.array([x, y, z]).T
        if True:
            domain, bary = sphere.pick_primal(points.reshape(-1, 3))

            print(bary.min(), bary.max())
            # bary = np.clip(bary, 0, 1)
            # color = domain.reshape(len(p), len(p))
            # p = np.random.permutation(sphere.topology.n_elements[-1])
            # color = p[color]
            color = bary.reshape(len(p), len(p), 3)

        import matplotlib.pyplot as plt
        plt.imshow(np.swapaxes(color, 0, 1)[::-1], cmap='jet')
        sphere.plot(backface_culling=True)
        plt.autoscale(tight=True)
        plt.show()


def test_picking_fundamental_visual():
    sphere = synthetic.optimal_delaunay_sphere(300, 3, iterations=5, weights=False, condition=None)

    print(sphere.is_well_centered)
    # sphere = synthetic.n_cube_dual(3).as_2().subdivide().subdivide().subdivide()

    sphere.plot(backface_culling=True)
    plt.autoscale(tight=True)

    sphere = sphere.optimize_weights()

    sphere.plot(backface_culling=True)
    plt.autoscale(tight=True)
    plt.show()

    p = np.linspace(-1, +1, 512, endpoint=True)
    x, y = np.meshgrid(p, p)
    r2 = x**2 + y**2
    mask = r2 < 1

    z = np.sqrt(1 - np.clip(r2, 0, 1))

    points = np.array([x, y, z]).T
    idx, bary, domain = sphere.pick_fundamental(points.reshape(-1, 3))
    color = bary.reshape(len(p), len(p), 3)
    color[np.logical_not(mask)] = 0

    plt.figure()
    plt.imshow(np.swapaxes(color, 0, 1)[::-1], cmap='jet')
    plt.show()


def test_fundamental_subdivide():
    sphere = synthetic.icosphere(1)
    sphere = sphere.copy(vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3))))
    sphere = sphere.subdivide_fundamental()
    sphere.plot(backface_culling=True, plot_vertices=False)
    # FIXME: does not yet work for n > 3
    n = 3
    sphere = synthetic.n_cube_dual(n)
    sphere = sphere.copy(vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(n, n))))
    sphere = sphere.subdivide_fundamental().optimize_weights()
    sphere.plot(backface_culling=True, plot_vertices=False)
    plt.show()


def test_overlap():
    # drawing to intuit multigrid transfer operators
    sphere = synthetic.icosahedron()#.subdivide_fundamental()
    sphere = sphere.select_subset(np.eye(20)[0])
    for i in range(1):
        sphere = sphere.subdivide_loop_direct()

    fig, ax = plt.subplots(1, 1)
    # sphere = sphere.optimize_weights()
    # sphere = sphere.optimize_weights_metric()
    # FIXME: invocation of direct loop fails when not on simple triangle
    subsphere = sphere.subdivide_loop()
    sphere.plot(ax=ax)
    # subsphere = subsphere.optimize_weights()
    subsphere.plot(ax=ax, primal_color='c', dual_color='m')
    plt.show()


def test_multigrid_tri():
    sphere = synthetic.icosahedron().subdivide_fundamental()
    sphere_0 = sphere.select_subset(np.eye(20*6)[1])
    sphere_1 = sphere_0.subdivide_loop()
    sphere_2 = sphere_1.subdivide_loop()
    sphere_3 = sphere_2.subdivide_loop()
    sphere_4 = sphere_3.subdivide_loop()


    t = sphere.multigrid_transfer_dual(sphere_0, sphere_1).T
    from pycomplex.sparse import normalize_l1

    # t = normalize_l1(t)

    t = t.tocoo()
    plt.scatter(t.row, t.col, c=t.data)
    plt.gca().invert_yaxis()

    plt.axis('equal')

    plt.colorbar()
    plt.show()

    print(t)


def test_multigrid_form():
    sphere_0 = synthetic.icosahedron().subdivide_fundamental()
    sphere_1 = sphere_0.subdivide_loop()
    sphere_2 = sphere_1.subdivide_loop()
    sphere_3 = sphere_2.subdivide_loop()
    sphere_4 = sphere_3.subdivide_loop()
    fine, coarse = sphere_2, sphere_1

    t = sphere_0.multigrid_transfer_dual(coarse, fine).T

    npt.assert_allclose(t.sum(axis=0), coarse.dual_metric[-1][None, :])
    npt.assert_allclose(t.sum(axis=1), fine.dual_metric[-1][:, None])

    from pycomplex.sparse import normalize_l1
    c2f = normalize_l1(t, axis=0)
    f2c = normalize_l1(t.T, axis=0)

    P0 = np.random.rand(coarse.topology.n_elements[0])
    D2 = coarse.hodge_DP[0] * P0
    d2 = c2f * D2
    p0 = fine.hodge_PD[0] * d2
    print(d2.sum(), D2.sum())

    fig, ax = plt.subplots(1, 1)
    coarse.as_euclidian().plot_primal_0_form(P0, ax=ax, vmin=0, vmax=1)
    fig, ax = plt.subplots(1, 1)
    fine.as_euclidian().plot_primal_0_form(p0, ax=ax, vmin=0, vmax=1)
    plt.show()


    p0 = np.random.rand(fine.topology.n_elements[0])
    d2 = fine.hodge_DP[0] * p0
    D2 = f2c * d2
    P0 = coarse.hodge_PD[0] * D2
    print(d2.sum(), D2.sum())


    fig, ax = plt.subplots(1, 1)
    coarse.as_euclidian().plot_primal_0_form(P0, ax=ax, vmin=0, vmax=1)
    fig, ax = plt.subplots(1, 1)
    fine.as_euclidian().plot_primal_0_form(p0, ax=ax, vmin=0, vmax=1)
    plt.show()
    plt.show()


    if False:
        t = t.tocoo()
        plt.scatter(t.row, t.col, c=t.data)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.colorbar()
    plt.show()


def test_flux_to_vector():
    """Test if a constant gradient potential produces constant vectors"""
    sphere = synthetic.optimal_delaunay_sphere(n_dim=3, n_points=200, condition='centered')
    # potential that is a linear gradient
    phi_p0 = sphere.primal_position[0][:, 0]

    P01, P12 = sphere.topology.matrices
    flux_p1 = P01.T * phi_p0
    # check what happens to non-solenoidal components
    # flux_p1 += np.random.randn(*flux_p1.shape) * 1e-2
    flux_d1 = sphere.hodge_DP[1] * flux_p1
    # add zero boundary terms
    flux_d1 = sphere.topology.dual.selector[1].T * flux_d1
    velocity_d0 = sphere.dual_flux_to_dual_velocity(flux_d1)

    # npt.assert_allclose(velocity_d0, [[0, 1]]*len(velocity_d0), atol=1e-9, rtol=1)

    sphere.plot(plot_dual=True, backface_culling=True)
    p = sphere.primal_position[2]
    mask = p[:, 2] > 0
    plt.quiver(*p[mask, :2].T, *velocity_d0[mask, :2].T)
    plt.show()
