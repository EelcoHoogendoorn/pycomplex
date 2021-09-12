"""Metric calculations on regular grids"""

import numpy as np


def edge_length(s, e):
    return np.max(np.abs(s - e), axis=-1)


def hypervolume(a, b, n=None):
    """Compute unsigned hypervolume

    Parameters
    ----------
    a : ndarray, [n_cubes, n_dim], float
        one corner of the cubes
    b : ndarray, [n_cubes, n_dim], float
        opposing corner of the cubes
    n : int, optional
        to compute the restricted hypervolume of dimension n for cubes embedded in higher dimensional spaces

    Returns
    -------
    ndarray, [n_cubes], float
    """
    d = a - b
    if n is not None:
        d = np.sort(np.abs(d), axis=1)
        d = d[:, -n:]
    return np.abs(np.prod(d, axis=-1))


# def vertex_gradients(cubes):
#     """Gradient of hypervolume with respect to displacement along the vertices of each cube
#
#     Parameters
#     ----------
#     cubes, [n_cubes, (2,)**n_dim, n_dim], float
#
#     Returns
#     -------
#     gradients, [n_cubes, (2,)**n_dim, n_dim], float
#     """
#
#     # cubes = cubes.reshape(len(cubes), -1)
#     gradients = cubes - cubes.mean(axis=np.arange(cubes.ndim)[1:-1], keepdims=True)
#     # FIXME: this only is true for square cubes
#     return gradients
#
#
# def boundary_gradients(cubes):
#     """Gradient of hypervolume with respect to displacement along the bounding faces of each cube
#
#     Parameters
#     ----------
#
#     """
