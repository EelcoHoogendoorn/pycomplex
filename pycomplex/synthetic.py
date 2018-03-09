"""Generation of some simple complexes"""

import itertools
import numpy as np
import scipy.spatial

from pycomplex.complex.simplicial.euclidian import ComplexSimplicialEuclidian
from pycomplex.complex.cubical import ComplexCubical
from pycomplex.complex.simplicial.spherical import ComplexSpherical, ComplexSpherical2, ComplexSpherical3
from pycomplex.topology import index_dtype
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial
from pycomplex.topology.cubical import TopologyCubical
from pycomplex.math import linalg


def n_simplex(n_dim, equilateral=True):
    """Generate a single n-simplex

    Parameters
    ----------
    n_dim : int
    equilateral : bool
        If False, canonical vertices are generated instead

    Returns
    -------
    ComplexSimplicialEuclidian

    """
    def simplex_vertices(n):
        """Recursively generate equidistant vertices on the n-sphere"""
        if n == 1:  # terminating special case; a line-segment
            return np.array([[-1], [+1]])
        depth = 1. / n
        shrink = np.sqrt(1 - depth ** 2)
        base = simplex_vertices(n - 1) * shrink
        top = np.eye(n)[:1]
        bottom = np.ones((n, 1)) * -depth
        return np.block([[top], [bottom, base]])

    if equilateral:
        vertices = simplex_vertices(n_dim)
    else:
        vertices = np.eye(n_dim + 1)[:, 1:] # canonical simplex

    corners = np.arange(n_dim + 1, dtype=index_dtype)
    return ComplexSimplicialEuclidian(vertices=vertices, simplices=corners[None, :])


def n_cube(n_dim, centering=False, mirror=True):
    """Generate a single n-cube in euclidian n-space

    Parameters
    ----------
    n_dim : int
        dimension of the cube to be generated
    centering : bool
        if True, cube has coords in range [-0.5, +0.5]
        if False, cube has coords in range [0.0, 1.0]
    mirror : bool
        if False, a non-oriented cubical complex is generated

    Returns
    -------
    ComplexCubical
    """
    return n_cube_grid((1,) * n_dim, centering=centering, mirror=mirror)


def n_cube_grid(shape, centering=True, mirror=True):
    """Generate a regular grid of n-cubes in euclidian n-space

    Parameters
    ----------
    shape : tuple of int
        shape of the grid to be generated, as the number of cubes in each dimension
    centering : bool
        if True, centroid of the complex is at the origin
        if False, minimum of the complex is at the origin
    mirror : bool
        if False, a non-oriented cubical complex is generated

    Returns
    -------
    CubicalComplex
    """
    n_dim = len(shape)
    cube_shape = (2,) * n_dim
    vshape = tuple(np.array(shape) + 1)
    vertices = np.indices(vshape, dtype=np.float)
    vertices = vertices.reshape(n_dim, -1).T
    if centering:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)
    # clever bit of striding logic to construct our grid
    idx = np.arange(np.prod(vshape), dtype=index_dtype).reshape(vshape)
    cubes = np.ndarray(
        buffer=idx,
        strides=idx.strides + idx.strides,  # step along the grid is the same as a step to the other side of the cube
        shape=tuple(shape) + cube_shape,
        dtype=idx.dtype
    )
    return ComplexCubical(
        vertices=vertices,
        topology=TopologyCubical.from_cubes(
            cubes.reshape((-1,) + cube_shape),
            mirror=mirror
        ),
    )


def n_cube_dual(n_dim):
    """Dual of n-cube boundary

    Parameters
    ----------
    n_dim : int
        dimensionality of the embedding space

    Returns
    -------
    ComplexSpherical
        consists of 2**n_dim regular simplices on the surface of an n_dim-1-sphere

    Notes
    -----
    quad, octahedron, hexadecachoron for n_dim = 2,3,4 resp.
    """
    cube = n_cube(n_dim, centering=True).boundary

    # grab simplices from cube corners
    from pycomplex.topology import sparse_to_elements
    cubes = cube.topology.matrix(n_dim - 1, 0)
    simplices = sparse_to_elements(cubes)

    return ComplexSpherical(
        vertices=linalg.normalized(cube.dual_position[0]),
        topology=TopologySimplicial.from_simplices(simplices).fix_orientation()
    )


def icosahedron():
    """Generate an icosahedron. Biggest symmetry group on the 2-sphere

    Returns
    -------
    ComplexSpherical2
    """
    phi = (1 + np.sqrt(5)) / 2
    Q = np.array(list(itertools.product([0], [-1, +1], [-phi, +phi])))
    vertices = np.vstack([np.roll(Q, r, axis=1) for r in range(3)])     # even permutations

    triangles = np.array([t for t in itertools.combinations(range(12), 3)])
    corners = vertices[triangles]
    centers = corners.mean(axis=1, keepdims=True)
    d = np.linalg.norm(corners - centers, axis=-1).sum(axis=1)
    triangles = triangles[d < 4].astype(index_dtype)

    return ComplexSpherical2(
        vertices=linalg.normalized(vertices),
        topology=TopologyTriangular.from_simplices(triangles).fix_orientation()
    )


def icosphere(refinement=0):
    """Generate a sphere by recursive subdivision of an icosahedron.

    Parameters
    ----------
    refinement: int, optional
        number of levels of subdivision

    Returns
    -------
    ComplexSpherical2
    """
    sphere = icosahedron()
    for _ in range(refinement):
        sphere = sphere.subdivide_loop()
    return sphere


def hexacosichoron():
    """Biggest symmetry group on the 4-sphere, analogous to the icosahedron on the 3-sphere

    Its dual is a hyperdodecahedron, consisting of 120 dodecahedra

    Returns
    -------
    ComplexSpherical3
    """
    def snub_24():
        """Generate the vertices of a snub-24 complex"""
        phi = (1 + np.sqrt(5)) / 2

        b = [phi, 1, 1/phi, 0]
        from pycomplex.math.combinatorial import permutations
        par, perm = zip(*permutations(list(range(4))))
        par, perm = np.array(par), np.array(perm)
        perm = perm[par == 0]     # get only even permutations

        flips = np.indices((2, 2, 2, 1)) - 0.5
        flips = flips.T.reshape(-1, 4)

        return np.asarray([(flip * b)[p] for flip in flips for p in perm])

    vertices = np.concatenate([
        n_cube(4, centering=True).vertices,
        n_cube_dual(4).vertices,
        snub_24()
    ], axis=0)

    tets = scipy.spatial.ConvexHull(vertices).simplices
    topology = TopologySimplicial.from_simplices(tets).fix_orientation()
    return ComplexSpherical3(vertices=vertices, topology=topology)


def optimal_delaunay_sphere(n_points, n_dim, iterations=50, weights=True, push_iterations=10, condition='delaunay'):
    """Try and construct an optimal delaunay mesh on the sphere, in the sense described in [1],
    by repeated averaging over dual centroid positions

    Parameters
    ----------
    n_points : int
        number of vertices of the primal mesh
    n_dim : int
        dimension of the embedding space

    Returns
    -------
    ComplexSpherical

    References
    ----------
    [1] https://www.math.uci.edu/~chenlong/mesh.html

    Notes
    -----
    This tends to rapidly produce well-centered triangles; but tets are another matter already
    """
    # FIXME: convergence here is extremely sensitive to parameter tuning, and only really seems to converge in the n=3 case
    points = np.random.randn(n_points, n_dim)
    import numpy_indexed as npi

    def complex_from_points(points):
        points = linalg.normalized(points)
        delaunay = scipy.spatial.ConvexHull(points)
        topology = TopologySimplicial.from_simplices(delaunay.simplices).fix_orientation()
        complex = ComplexSpherical(vertices=points, topology=topology)
        if weights:
            complex = complex.optimize_weights()
        return complex

    def push_points(points, r=5, magnitude=.1):
        # FIXME: use connectivity for this instead of tree?
        # FIXME: ohrased differently; use laplacian smoothing step instead. can do implicit solve to make larger steps?
        # FIXME: or use it as descent direction, and do line search with some criterium; same applies to loyds iteration
        r = r / np.power(n_points, 1 / (n_dim - 1))
        tree = scipy.spatial.cKDTree(points)
        s, e = tree.query_pairs(r=r, output_type='ndarray').T
        n, d = linalg.normalized(points[s] - points[e], return_norm=True)
        f = n * (r - d)[:, None] * magnitude
        a, b = npi.group_by(s).sum(f)
        points[a] += b
        a, b = npi.group_by(e).sum(f)
        points[a] -= b
        return linalg.normalized(points)

    for i in range(iterations):
        print(i)
        for i in range(push_iterations):
            points = push_points(points)
        complex = complex_from_points(points)
        # FIXME: this is essentially Lioyds algorithm; that is, insofar area weighted method produces true centroid; which i am not sure it does
        cc = complex.dual_position[0]
        from pycomplex.geometry import euclidian
        W = euclidian.unsigned_volume(complex.vertices[complex.topology.corners[-1]])[:, None]
        A = complex.topology.averaging_operators_N[0]
        # take the average at each dual vertex, of all incident n-simplices
        points = (A * (cc * W)) / (A * W)
        complex = complex_from_points(points)

        if condition == 'delaunay':
            if complex.is_pairwise_delaunay:
                break
        if condition == 'centered':
            if complex.is_well_centered:
                break

    return complex


def delaunay_cube(density=30, n_dim=2, iterations=30):
    """Generate a delaunay simplex mesh on a cube

    Parameters
    ----------
    density : int
        number of vertices along one axis of the cube
    n_dim : int
        dimensionality of the complex
    iterations : int
        number of aspect-ratio-optimization iterations

    Returns
    -------
    ComplexSimplicialEuclidian
    """
    import scipy.spatial

    idx = np.indices((density + 1,) * n_dim)
    outer = np.where(np.any(np.logical_or(idx == 0, idx == density), axis=0))
    outer = np.array(outer).T / density

    def complex_from_points(points):
        e = 0.5 / density
        r = np.any(np.logical_or(points < (0 + e), points > (1 - e)), axis=1)
        points = np.delete(points, np.flatnonzero(r), axis=0)

        points = np.concatenate([outer, points], axis=0)
        simplices = scipy.spatial.Delaunay(points).simplices
        a, b = np.unique(simplices, return_inverse=True)
        simplices = b.reshape(simplices.shape).astype(index_dtype)
        points = points[a]
        topology = TopologySimplicial.from_simplices(simplices.astype(index_dtype)).fix_orientation()
        complex = ComplexSimplicialEuclidian(points, topology=topology)
        return complex

    points = np.random.uniform(0, 1, (density ** n_dim, n_dim))
    complex = complex_from_points(points)

    for i in range(iterations):
        cc = complex.primal_position[-1]
        from pycomplex.geometry import euclidian
        W = euclidian.unsigned_volume(complex.vertices[complex.topology.corners[-1]])[:, None]
        A = complex.topology.averaging_operators_N[0]
        # take the average at each dual vertex, of all incident n-simplices
        points = (A * (cc * W)) / (A * W)
        complex = complex_from_points(points)

    return complex
