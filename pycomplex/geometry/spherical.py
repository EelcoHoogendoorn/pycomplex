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
    edge_planes = [linalg.normalized(ep) for ep in edge_planes]
    angles      = [linalg.dot(edge_planes[p - 2], edge_planes[p - 1]) for p in range(3)] #a3 x [faces, c3]
    areas       = sum(np.arccos(-a) for a in angles) - np.pi                        #faces
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
            raise ValueError('Only embedding dimensions of 3 is currently supported')
        a, b, c = [pts.take(i, axis=-2) for i in range(3)]
        return triangle_area_from_corners(a, b, c)
    if n_pts == 2:
        return edge_length(*pts.take([0, 1], axis=-2))
    if n_pts == 1:
        return np.ones_like(pts[..., 0, 0])


def edge_circumcenter(pts):
    n_pts, n_dim = pts.shape[-2:]
    if n_pts != 2:
        raise ValueError('Edge needs to be defined in terms of 2 points')
    return linalg.normalized(pts.mean(axis=-2))


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

    """
    pts = np.asarray(pts)
    n_pts, n_dim = pts.shape[-2:]
    if n_pts == 3:
        return triangle_circumcenter(pts)
    if n_pts == 2:
        return edge_circumcenter(pts)
    if n_pts == 1:
        return pts[:, 0, :]
    raise ValueError()
