
import numpy as np
import numpy.testing as npt
from pycomplex.stencil.complex import StencilComplex


def dense(op, shape):
    """Transform a matrix-free operator into a dense operator by brute force evaluation"""
    def canonical(i):
        r = np.zeros(shape)
        r[tuple(i)] = 1
        return r

    idx = np.indices(shape)
    r = [op(canonical(c)) for c in idx.reshape(len(shape), -1).T]
    return np.array(r).reshape(len(r), -1)


def test_transpose():
    """Test numerically that matching primal and dual operators obey a tranpose relationship"""
    for i in range(2, 5):
        complex = StencilComplex(tuple(np.arange(i, dtype=np.int) * 2 + 2))

        for n in range(complex.ndim):
            p = dense(complex.primal[n].right, complex.primal[n].shape[1])
            d = dense(complex.dual[n].right, complex.dual[n].shape[1])

            npt.assert_allclose(p, d.T)


def test_chain():
    """Test that primal and dual chain complexes are closed"""
    for i in range(2, 5):
        complex = StencilComplex(tuple(np.arange(i, dtype=np.int) * 2 + 2))
        print(complex.shape)
        for n in range(complex.ndim - 1):
            f0 = complex.form(n)
            f0[...] = np.random.normal(size=f0.shape)
            f1 = complex.primal[n](f0)
            f2 = complex.primal[n+1](f1)
            npt.assert_allclose(f2, 0, atol=1e-5)

        for n in range(complex.ndim, 1, -1):
            f0 = complex.form(n)
            f0[...] = np.random.normal(size=f0.shape)
            f1 = complex.dual[n-1](f0)
            f2 = complex.dual[n-2](f1)
            npt.assert_allclose(f2, 0, atol=1e-5)


def test_transfers():
    """Test that constant functions are preserved by a roundtrip across scales"""
    for i in range(2, 5):
        complex = StencilComplex(tuple(np.arange(i, dtype=np.int) * 2 + 2))
        print(complex.shape)

        for n in range(complex.ndim + 1):
            f0 = complex.form(n, init='ones')
            c0 = complex.coarsen[n](f0)
            f0_ = complex.refine[n](c0)

            npt.assert_allclose(f0, f0_)
