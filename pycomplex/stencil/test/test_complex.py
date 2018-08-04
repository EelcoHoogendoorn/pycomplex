
import numpy as np
from pycomplex.stencil.complex import StencilComplex


def test_2d():
    complex = StencilComplex((3, 4))
    f0 = complex.form(0)
    f0 = np.arange(12).reshape(1, 3, 4)
    f1 = complex.dual[0].transpose(f0)
    f2 = complex.primal[1](f1)
    print(f2)


def test_3d():
    complex = StencilComplex((2, 3, 4))
    f0 = complex.form(0)
    f0 = np.arange(24).reshape(1, 2, 3, 4)
    f1 = complex.dual[0].transpose(f0)
    f2 = complex.primal[1](f1)
    print(f2)
