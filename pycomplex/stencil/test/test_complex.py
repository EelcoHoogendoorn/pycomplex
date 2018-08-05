
import numpy as np
import numpy.testing as npt
from pycomplex.stencil.complex import StencilComplex


def test_2d():
    complex = StencilComplex((3, 4))
    f0 = complex.form(0)
    f0 = np.arange(12).reshape(1, 3, 4)
    f1 = complex.dual[0].transpose(f0)
    f2 = complex.primal[1](f1)
    npt.assert_allclose(f2, 0)


def test_3d():
    complex = StencilComplex((2, 3, 4))
    f0 = complex.form(0)
    f0 = np.arange(24).reshape(1, 2, 3, 4)
    f1 = complex.dual[0].transpose(f0)
    f2 = complex.primal[1](f1)
    npt.assert_allclose(f2, 0)


def test_transfers():
    complex = StencilComplex((2, 4))
    f0 = complex.form(0)
    f0[...] = 1
    c0 = complex.coarsen[0](f0)

    f0 = complex.refine[0](c0)

    print(c0)
    print(f0)
