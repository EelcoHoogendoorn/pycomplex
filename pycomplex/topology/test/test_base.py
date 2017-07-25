from pycomplex.topology.simplicial import generate_boundary
import numpy as np


def test_combinations():

    tris = [np.arange(3), np.arange(3)+1]
    comb = generate_boundary(tris, 2, axis=1)
    print(comb)


def basic_test(rt):
    """Some basic sanity checks on a topology matrix"""
    for i in range(rt.n_dim):
        t = rt.matrix(i)
        print(t.shape)
        print(t.todense())

    for i in range(rt.n_dim - 1):
        a, b = rt.matrix(i), rt.matrix(i+1)
        print((a * b).todense())

    for r in rt.corners:
        print(r.shape)