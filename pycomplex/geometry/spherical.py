"""Some routines for geometrical computations on a sphere surface"""

import numpy as np

from pycomplex.math import linalg


def edge_length(*edge):
    """compute spherical edge length.

    Parameters
    ----------
    edge : 2 x ndarray, [n, 3], float
        arc segments described by their start and end position

    Returns
    -------
    lengths: ndarray, [n], float
        length along the unit sphere of each segment
    """
    return np.arccos(linalg.dot(*edge))


def triangle_area_from_normals(*edge_planes):
    """compute spherical area from triplet of great circles

    Parameters
    ----------
    edge_planes : 3 x ndarray, [n, 3], float
        edge normal vectors of great circles

    Returns
    -------
    areas : ndarray, [n], float
        spherical area enclosed by the input planes
    """
    edge_planes = [linalg.normalized(ep, ignore_zeros=True) for ep in edge_planes]
    angles      = [linalg.dot(edge_planes[p - 2], edge_planes[p - 1]) for p in range(3)] #a3 x [faces, c3]
    areas       = sum(np.arccos(-np.clip(a, -1, 1)) for a in angles) - np.pi                        #faces
    return areas


def triangle_area_from_corners(*tri):
    """compute spherical area from triplet of triangle corners

    Parameters
    ----------
    tri : 3 x ndarray, [n, 3], float
        corners of each triangle

    Returns
    -------
    areas : ndarray, [n], float
        spherical area enclosed by the input corners
    """
    return triangle_area_from_normals(*[np.cross(tri[v-2], tri[v-1]) for v in range(3)])


def triangle_areas_around_center(center, corners):
    """given a triangle formed by corners, and its dual point center,
    compute spherical area of the voronoi faces

    Parameters
    ----------
    center : ndarray, [..., 3], float
    corners : ndarray, [..., 3, 3], float

    Returns
    -------
    areas : ndarray, [..., 3], float
        spherical area opposite to each corner
    """
    areas = np.empty(corners.shape[:-1])
    for i in range(3):
        areas[:,:,i] = triangle_area_from_corners(center, corners[:,:,i-2], corners[:,:,i-1])
    #swivel equilaterals to vonoroi parts
    return (areas.sum(axis=2)[:,:,None]-areas) / 2


def unsigned_volume(pts):
    """

    Parameters
    ----------
    pts : ndarray, [..., n_pts, n_dim], float
        corners of

    Returns
    -------
    unsigned_volume : ndarray, [...], float
        volume of each simplex

    """
    pts = np.asarray(pts)
    n_pts, n_dim = pts.shape[-2:]

    if n_pts == 3:
        if n_dim != 3:
            # FIXME: crap; nd-case seems really hard: https://arxiv.org/pdf/1011.2584.pdf
            # are we any better off working with fundamental domains with right angles?
            # https://www.math.cornell.edu/~hatcher/Other/hopf-samelson.pdf
            # this seems inspiring; seems like generalizing spherical excess formula should be possible
            # 3 dihedral wedges cover the triangle three times, and the rest of the sphere once.
            # 6 dihedral wedges on a tet should cover the tet 6 times, and the rest of the sphere once, no? do they indeed cocer the sphere?
            # but what about areas of tet faces?
            raise ValueError('Only embedding dimensions of 3 is currently supported')
        a, b, c = [pts.take(i, axis=-2) for i in range(3)]
        return np.abs(triangle_area_from_corners(a, b, c))
    if n_pts == 2:
        return edge_length(*np.rollaxis(pts, axis=-2))
    if n_pts == 1:
        return np.ones_like(pts[..., 0, 0])


def edge_circumcenter(pts):
    n_pts, n_dim = pts.shape[-2:]
    if n_pts != 2:
        raise ValueError('Edge needs to be defined in terms of 2 points')
    return linalg.normalized(pts.sum(axis=-2))


def triangle_circumcenter(pts):
    """Circumcenter of a triangle on a sphere"""

    n_pts, n_dim = pts.shape[-2:]
    if n_dim != 3:
        raise ValueError('Only embedding dimensions of 3 is currently supported')
    if n_pts != 3:
        raise ValueError('Triangle needs to be defined in terms of 3 points')

    l = np.take(pts, [0, 1], axis=-2)
    r = np.take(pts, [1, 2], axis=-2)
    d = l - r
    return linalg.normalized(np.cross(d.take(1, axis=-2), d.take(0, axis=-2)))


def circumcenter(pts):
    """Compute circumcenter of spherical n_simplex

    Parameters
    ----------
    pts : ndarray, [..., n_pts, n_dim], float

    Returns
    -------
    circumcenters : ndarray, [..., n_dim], float

    Notes
    -----
    This can be computed as a renormalzied version of the euclidian case
    """
    pts = np.asarray(pts)
    n_pts, n_dim = pts.shape[-2:]
    # if n_pts == 3 and n_dim == 3:
    #     return triangle_circumcenter(pts)
    if n_pts == 2:
        return edge_circumcenter(pts)
    if n_pts == 1:
        return pts[:, 0, :]
    from pycomplex.geometry import euclidian
    return linalg.normalized(euclidian.circumcenter(pts))


def intersect_edges(s0, e0, s1, e1):
    """intersect spherical edges defined by their endpoints

    Parameters
    ----------
    s0 : ndarray, [..., 3], float
    e0 : ndarray, [..., 3], float
    s1 : ndarray, [..., 3], float
    e1 : ndarray, [..., 3], float

    Returns
    -------
    ndarray, [..., 3], float

    Notes
    -----
    It is assumed the edges are not colinear
    """
    n0 = np.cross(s0, e0)
    n1 = np.cross(s1, e1)
    return linalg.normalized(np.cross(n0, n1))
