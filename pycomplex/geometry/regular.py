"""Metric calculations on regular grids"""

import numpy as np


def edge_length(s, e):
    return np.max(np.abs(s - e), axis=-1)


def hypervolume(a, b, n=None):
    """Compute unsigned hypervolume"""
    d = a - b
    if n is not None:
        d = np.sort(np.abs(d), axis=1)
        d = d[:, -n:]
    return np.abs(np.prod(d, axis=-1))