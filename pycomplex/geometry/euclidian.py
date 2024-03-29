"""Some routines for geometric calculations in euclidian space

Note: these are vectorized variants of the functions found in pyDEC
"""

import numpy as np
import scipy.special

from pycomplex.math import linalg


def is_wellcentered(pts, tol=1e-8):
    """Determines whether a set of points defines a well-centered simplex.
    """
    barycentric_coordinates = circumcenter_barycentric(pts)
    return min(barycentric_coordinates) > tol


def circumcenter_construction(pts, weights=None):
    """

    Parameters
    ----------
    pts : ndarray. [..., n_pts, n_dim], float
        set of points euclidian space.
    weights : ndarray, [..., n_pts], float
        weight specifying a squared distance offset

    Returns
    -------
    circumcenter : ndarray. [..., n_dim], float
        coordinates of the (weighted) circumcenter of the simplex defined by pts.

    Notes
    -----
    Implemented after [1]; but 'without the Agonizing Pain', if I may say so myself

    References
    ----------
    [1] http://www.geometry.caltech.edu/pubs/MMdGD11.pdf
    """
    pts = np.asarray(pts)
    if weights is not None:
        weights = np.asarray(weights)

    first = pts[..., 0, :]
    moved = pts[..., 1:, :] - first
    e2 = linalg.dot(moved, moved)
    if weights is not None:
        e2 -= (weights[..., 1:] - weights[..., 0])
    pinv = linalg.pinv(moved)

    centroid = linalg.dot(pinv, e2) / 2 + first
    return centroid


def circumcenter_barycentric_weighted(pts, weights=None, ratio=1e6):
    """Barycentric coordinates of the circumcenter of a set of points in euclidian space.

    Parameters
    ----------
    pts : ndarray. [..., n_pts, n_dim], float
        set of points euclidian space.
    weights : ndarray, [..., n_pts], float
        weight specifying a squared distance offset

    Returns
    -------
    coords : ndarray. [..., n_pts], float
        Barycentric coordinates of the circumcenter of the simplex defined by pts.

    References
    ----------
    https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm
    """
    pts = np.asarray(pts)
    if weights is not None:
        weights = np.asarray(weights)

    n_pts, n_dim = pts.shape[-2:]
    gu = pts.shape[:-2]
    N = n_pts + 1

    A = np.ones(gu + (N, N))
    delta = pts[..., :, None, :] - pts[..., None, :, :]
    A[..., -1, -1] = 0
    a = linalg.dot(delta, delta)
    if weights is not None:
        a = a - (weights[..., :, None] + weights[..., None, :])
    A[..., :-1, :-1] = a

    v, w = np.linalg.eigh(A)
    vr = 1 / v
    pinv = np.einsum('...ij,...j,...kj->...ik', w, vr, w)

    b = np.ones(gu + (N,))
    b[..., :-1] = 0
    x = np.einsum('...ij,...j->...i', pinv, b)
    bary_coords = x[..., :-1]

    return bary_coords


def circumcenter_barycentric(pts, weights=None, ratio=1e6):
    """Barycentric coordinates of the circumcenter of a set of points in euclidian space.

    Parameters
    ----------
    pts : ndarray. [..., n_pts, n_dim], float
        set of points euclidian space.
    weights : ndarray, [..., n_pts], float
        vertex weights in a power-diagram sense
    ratio : float
        eigenvalue ratio in pseudoinverse;
        determines if barycentric coords should snap to a subspace of n_dim

    Returns
    -------
    coords : ndarray. [..., n_pts], float
        Barycentric coordinates of the (weighted) circumcenter of the simplex defined by pts.

    Examples
    --------
    >>> circumcenter_barycentric([[0],[4]])           # edge in 1D
    array([ 0.5,  0.5])
    >>> circumcenter_barycentric([[0,0],[4,0]])       # edge in 2D
    array([ 0.5,  0.5])
    >>> circumcenter_barycentric([[0,0],[4,0],[0,4]]) # triangle in 2D
    array([ 0. ,  0.5,  0.5])

    See Also
    --------
    circumcenter_barycentric

    Notes
    -----
    Note that the linear dependence on weights is trivially visible in this formulation

    References
    ----------
    https://github.com/hirani/pydec
    """

    pts = np.asarray(pts)
    if weights is not None:
        weights = np.asarray(weights)

    n_pts, n_dim = pts.shape[-2:]
    gu = pts.shape[:-2]
    N = n_pts + 1

    A = np.ones(gu + (N, N))
    A[..., -1, -1] = 0
    A[..., :-1, :-1] = np.einsum('...ij,...kj->...ik', pts, pts) * 2

    v, w = np.linalg.eigh(A)
    vr = 1 / v
    vr[np.abs(vr) > np.abs(vr).min() * ratio] = 0
    pinv = np.einsum('...ij,...j,...kj', w, vr, w)

    b = np.ones(gu + (N,))
    b[..., :-1] = np.einsum('...ij,...ij->...i', pts, pts)
    if weights is not None:
        b[..., :-1] -= weights
    x = np.einsum('...ij,...j->...i', pinv, b)
    bary_coords = x[..., :-1]

    return bary_coords


def circumcenter(pts):
    """Circumcenter of a set of points in Euclidian space.

    Parameters
    ----------
    pts : ndarray. [..., n_pts, n_dim], float
        set of points Euclidian space.

    Returns
    -------
    center : ndarray, [..., n_dim], float
        Circumcenters of the simplices defined by pts.

    Examples
    --------
    >>> circumcenter([[0],[1]])             # edge in 1D
    (array([ 0.5]), 0.5)
    >>> circumcenter([[0,0],[1,0]])         # edge in 2D
    (array([ 0.5,  0. ]), 0.5)
    >>> circumcenter([[0,0],[1,0],[0,1]])   # triangle in 2D
    (array([ 0.5,  0.5]), 0.70710678118654757)

    See Also
    --------
    circumcenter_barycentric

    References
    ----------
    Uses an extension of the method described here:
    http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
    """
    # FIXME: seems unweighted codepath here is no longer used
    pts = np.asarray(pts)
    mean = pts.mean(axis=-2, keepdims=True)  # centering
    pts = pts - mean

    bary_coords = circumcenter_barycentric(pts)
    center = np.einsum('...ji,...j->...i', pts, bary_coords) + mean[..., 0, :]
    return center


def unsigned_volume(pts):
    """Unsigned volume of a simplex in euclidian space

    Computes the unsigned volume of an M-simplex embedded in N-dimensional
    space. The points are stored row-wise in an array with shape (M+1,N).

    Parameters
    ----------
    pts : ndarray, [..., n_pts, n_dim], float
        coordinates of the vertices of the simplex.

    Returns
    -------
    volume : ndarray, [...], float
        Unsigned volume of the simplex

    Notes
    -----
    Zero-dimensional simplices (points) are assigned unit volumes.

    Examples
    --------
    >>> # 0-simplex point
    >>> unsigned_volume( [[0,0]] )
    1.0
    >>> # 1-simplex line segment
    >>> unsigned_volume( [[0,0],[1,0]] )
    1.0
    >>> # 2-simplex triangle
    >>> unsigned_volume( [[0,0,0],[0,1,0],[1,0,0]] )
    0.5

    References
    ----------
    [1] http://www.math.niu.edu/~rusin/known-math/97/volumes.polyh
    """

    pts = np.asarray(pts)

    M, N = pts.shape[-2:]
    M -= 1

    if M < 0 or M > N:
        raise ValueError('array has invalid shape')

    if M == 0:
        return np.ones_like(pts[..., 0, 0])

    head, tail = np.split(pts, [1], axis=-2)
    A = tail - head
    B = np.einsum('...ji, ...ki->...jk', A, A)
    return np.sqrt(np.abs(np.linalg.det(B))) / scipy.special.factorial(M)


def signed_volume(pts):
    """Signed volume of a simplex in euclidian space

    Computes the signed volume of an M-simplex embedded in M-dimensional
    space. The points are stored row-wise in an array with shape (M+1,M).

    Parameters
    ----------
    pts : ndarray, [..., n_pts, n_dim], float
        coordinates of the vertices of the simplex.

    Returns
    -------
    volume : ndarray, [...], float
        Signed volume of the simplex

    Examples
    --------
    >>> # 1-simplex line segment
    >>> signed_volume( [[0],[1]] )
    1.0
    >>> # 2-simplex triangle
    >>> signed_volume( [[0,0],[1,0],[0,1]] )
    0.5
    >>> # 3-simplex tetrahedron
    >>> signed_volume( [[0,0,0],[3,0,0],[0,1,0],[0,0,1]] )
    0.5
    References
    ----------
    [1] http://www.math.niu.edu/~rusin/known-math/97/volumes.polyh
    """

    pts = np.asarray(pts)

    M, N = pts.shape[-2:]
    M -= 1

    if M != N:
        raise ValueError('array has invalid shape')

    head, tail = np.split(pts, [1], axis=-2)
    A = tail - head
    return np.linalg.det(A) / scipy.special.factorial(M)


def triangle_angles(vertices):
    """Compute interior angles for each triangle-vertex

    Parameters
    ----------
    vertices : ndarray, [..., 3, n_dim], float
        set of triangles described by the coordinates of its vertices in euclidian space

    Returns
    -------
    ndarray, [..., 3], float, radians
        interior angle of each vertex of each triangle
        the i-th angle is the angle corresponding to the i-th vertex of the triangle

    Notes
    -----
    Can this be generalized to tets as well? Is there a cotangent-laplacian equivalent for 3d?
    """
    edges = np.roll(vertices, -1, axis=-2) - np.roll(vertices, +1, axis=-2)
    edges = linalg.normalized(edges, ignore_zeros=True)
    return np.arccos(-linalg.dot(np.roll(edges, -1, axis=-2), np.roll(edges, +1, axis=-2)))


def segment_normals(corners):
    """Calculate the normals of a segment (an n-1-simplex in n-space)

    Parameters
    ----------
    corners : ndarray, [..., n_dim, n_dim], float
        set of simplices described by the coordinates of its vertices in euclidian space

    Returns
    -------
    normals : ndarray, [..., n_dim], float
        normals of the segments (not actually normalized)
    """
    corners = np.asarray(corners)
    n_corners, n_dim = corners.shape[-2:]
    assert n_corners == n_dim
    normals = np.empty_like(corners[..., 0])

    corners = corners[..., 1:, :] - corners[..., 0:1, :]

    sign = 1
    for c in range(n_corners):
        normals[..., c] = np.linalg.det(np.delete(corners, c, axis=-1)) * sign
        sign = -sign
    return normals


def simplex_normals(corners):
    """Compute simplex normals for the opposing face of all corner points

    Parameters
    ----------
    corners : ndarray, [..., n_pts, n_dim], float

    Returns
    -------
    normals : ndarray, [..., n_pts, n_dim], float
    """
    corners = np.asarray(corners)
    barycenter = corners.mean(axis=-2, keepdims=True)
    gradients = linalg.pinv(corners - barycenter)
    gradients = np.swapaxes(gradients, -2, -1)
    return linalg.normalized(gradients)


def simplex_gradients(corners):
    """Compute simplex volume gradients with respect to all corner points

    Parameters
    ----------
    corners : ndarray, [..., n_pts, n_dim], float

    Returns
    -------
    gradients : ndarray, [..., n_pts, n_dim], float
    """
    corners = np.asarray(corners)
    gradients = simplex_normals(corners)
    # multiply each normal with the area of the face opposing each corner
    for i in range(corners.shape[-2]):
        gradients[..., i, :] *= unsigned_volume(np.roll(corners, -i, axis=-2)[..., 1:, :])[..., None]
    return gradients
