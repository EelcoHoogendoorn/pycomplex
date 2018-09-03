
import pytest
import numpy as np
import numpy.testing as npt

from pycomplex.stencil.complex import StencilComplex


@pytest.mark.parametrize('i', [2, 3, 4])
def test_transpose(i):
    """Test numerically that matching primal and dual operators obey a transpose relationship"""
    complex = StencilComplex(tuple(np.arange(i, dtype=np.int) * 2 + 2))

    for n in range(complex.ndim):
        p = complex.primal[n].to_dense()
        d = complex.dual[n].to_dense()

        npt.assert_allclose(p, d.T)


@pytest.mark.parametrize('i', [2, 3, 4])
def test_chain(i):
    """Test that primal and dual chain complexes are closed"""
    complex = StencilComplex(tuple(np.arange(i, dtype=np.int) * 2 + 2))
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


@pytest.mark.parametrize('i', [2, 3, 4])
def test_transfers(i):
    """Test that constant functions are preserved by a roundtrip across scales"""
    complex = StencilComplex(tuple(np.arange(i, dtype=np.int) * 2 + 2))

    for n in range(complex.ndim + 1):
        f0 = complex.form(n, init='ones')
        c0 = complex.coarsen[n](f0)
        f0_ = complex.refine[n](c0)

        npt.assert_allclose(f0, f0_)
