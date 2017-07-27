"""Generation of some simple complexes"""

import itertools
import numpy as np


from pycomplex.complex.simplicial import ComplexSimplicial
from pycomplex.complex.cubical import ComplexCubical
from pycomplex.complex.spherical import ComplexSpherical
from pycomplex.topology import index_dtype
from pycomplex.math import linalg


def icosahedron():
    """Generate an icosahedron.

    Returns
    -------
    ComplexSpherical
    """
    phi = (1 + np.sqrt(5)) / 2
    Q = np.array(list(itertools.product([0], [-1, +1], [-phi, +phi])))
    vertices = np.vstack([np.roll(Q, r, axis=1) for r in range(3)])

    triangles = np.array([t for t in itertools.combinations(range(12), 3)])
    corners = vertices[triangles]
    centers = corners.mean(axis=1, keepdims=True)
    d = np.linalg.norm(corners - centers, axis=-1).sum(axis=1)
    triangles = triangles[d < 4]

    return ComplexSpherical(linalg.normalized(vertices), triangles)


def icosphere(refinement=0):
    """Generate a sphere by recursive subdivision of an icosahedron.

    Parameters
    ----------
    refinement: int, optional
        number of levels of subdivision

    Returns
    -------
    ComplexSpherical
    """
    sphere = icosahedron()
    for _ in range(refinement):
        sphere = sphere.subdivide()
    return sphere


def n_simplex(n_dim, symmetric=True):
    """Generate a single n-simplex

    Parameters
    ----------
    n_dim : int
    symmetric : bool

    Returns
    -------
    ComplexSimplicial

    """
    def simplex_vertices(n):
        """Recursively generate equidistant vertices on the n-sphere"""
        if n == 1:
            return np.array([[-1], [+1]])
        depth = 1. / n
        shrink = np.sqrt(1 - depth ** 2)
        base = simplex_vertices(n - 1) * shrink
        top = np.eye(n)[:1]
        bottom = np.ones((n, 1)) * -depth
        return np.block([[top], [bottom, base]])

    if symmetric:
        vertices = simplex_vertices(n_dim)
    else:
        vertices = np.eye(n_dim + 1)[:, 1:] # canonical simplex

    corners = np.arange(n_dim + 1, dtype=index_dtype)
    return ComplexSimplicial(vertices=vertices, simplices=corners[None, :])


def n_cube(n_dim, centering=False):
    """Generate a single n-cube in euclidian n-space

    Parameters
    ----------
    n_dim : int
        dimension of the cube to be generated

    Returns
    -------
    ComplexCubical
    """
    return n_cube_grid((1,) * n_dim, centering=centering)


def n_cube_grid(shape, centering=True):
    """Generate a regular grid of n-cubes in euclidian n-space

    Parameters
    ----------
    shape : tuple of int
        shape of the grid to be generated, as the number of cubes in each dimension

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
        strides=idx.strides + idx.strides,  # step along the grid is the same as a step to a new cube
        shape=tuple(shape) + cube_shape,
        dtype=idx.dtype
    )
    return ComplexCubical(
        vertices=vertices,
        cubes=cubes.reshape((-1,) + cube_shape)
    )
