from functools import reduce
import numpy_indexed as npi
import numpy as np


x = np.array([[3, 4, 5], [5, 12, 13], [6, 8, 10], [7, 24, 25]])
y = np.array([[3, 4], [4, 5], [3, 5], [5, 12]])

def contains_union(x, y):
    """Returns an ndarray with a bool for each element in y"""
    idx = [[0, 1], [1, 2], [0, 2]]
    y = npi.as_index(y)   # not required, but a performance optimization
    return reduce(np.logical_and, (npi.in_(x[:, i], y) for i in idx))

def contains_indices(x, y):
    """Returns an ndarray with a bool for each element in y"""
    idx = [[0, 1], [1, 2], [0, 2]]
    r = np.empty((len(y), 3))
    y = npi.as_index(y)   # not required, but a performance optimization
    i = [npi.indices(x[i], y, missing=-1) for i in idx]



print (contains_union(x, y))