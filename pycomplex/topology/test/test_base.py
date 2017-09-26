
import numpy as np


def basic_test(topology):
    """Some basic sanity checks on a topology matrix"""
    print('check structure of boundary')
    for i in range(topology.n_dim):
        t = topology.matrix(i)
        print(t.shape)
        print(t.todense())

    print('check closing of chains')
    for i in range(topology.n_dim - 1):
        a, b = topology.matrix(i), topology.matrix(i + 1)
        print((a * b).todense())

    print('check elements')
    for e in topology.elements:
        print(e.shape)
        print(e)