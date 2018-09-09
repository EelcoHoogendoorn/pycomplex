
import numpy as np
import numpy.testing as npt
import pytest

from pycomplex.stencil.complex import StencilComplex


@pytest.mark.parametrize('i', [2, 3, 4])
def test_transfers(i):
    """Test that constant functions are preserved by a roundtrip across scales"""
    complex = StencilComplex.from_shape(tuple(np.arange(i, dtype=np.int) * 2 + 2))

    for n in range(complex.n_dim + 1):
        f0 = complex.topology.form(n, init='ones')
        c0 = complex.coarsen[n](f0)
        f0_ = complex.refine[n](c0)

        npt.assert_allclose(f0, f0_)


def test_sample():
    """Test wrapping sampling logic"""
    complex = StencilComplex.from_shape((16, 16))

    source = complex.topology.form(0)

    source[0, 0, :] = 10
    sample = complex.sample_0(source, [[15, 0], [15.5, 0], [16, 0], [16.5, 0], [17, 0]])
    npt.assert_allclose(sample, [ 0.,  5., 10.,  5.,  0.])
    sample = complex.sample_0(source, [[-1, 0], [-0.5, 0], [0, 0], [0.5, 0], [1, 0]])
    npt.assert_allclose(sample, [ 0.,  5., 10.,  5.,  0.])

