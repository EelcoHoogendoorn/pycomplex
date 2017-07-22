"""Metric calculations on regular grids"""
import numpy as np

def edge_length(s, e):
    return np.max(np.abs(s - e), axis=-1)

def hypervolume(a, b):
    return np.abs(np.prod(a - b, axis=-1))