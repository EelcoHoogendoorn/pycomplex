
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
