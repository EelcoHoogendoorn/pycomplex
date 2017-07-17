
import numpy as np
from pycomplex.sparse import Sparse


def test():
    a = Sparse([[1, 2, 3, 1], [0, 1, 2, 2]], np.random.rand(4))
    b = Sparse([[1, 2, 3], [0, 1, 1]], np.random.rand(3))
    c = (a+b)
    (i,j),d = c.todense()
    print(d)
    m = c*c.T


def test_2():
    s = Sparse(
        ([0, 0, 0, 1, 2],
         [0, 1, 2, 2, 0]),
        data=[1, 1, 1, 1, 1]
    )
    r = Sparse(
        [[0, 0, 0, 1],
         [0, 1, 2, 2]],
        data = [1, 1, 1, 1]
    )
    q = s * s.T
    (i,j),d = q.todense()
    print(d)


def test_dot():
    a = np.arange(10)
    d = Sparse([a,a], data=np.ones_like(a))
    a = a
    od = Sparse([a+1, a], data=-np.ones_like(a))
    diag = Sparse([a, a], data=a+1)
    r = d + od
    q = r.dot(diag.dot(r.T))

    (i,j),d = q.todense()
    print(np.nan_to_num(d))
    return
    # test matrix-vector dot; needs work still
    s = Sparse([[5]], [1])
    r = q.dot(s)
    (i,j),d = r.todense()
    print(np.nan_to_num(d))


def test_dense():
    a = np.random.normal(size=(4, 5))
    b = np.random.normal(size=(5, 6))

    print(np.dot(a, b))
    print(Sparse.fromdense(a).dot(Sparse.fromdense(b)).todense()[1])


test_dot()
