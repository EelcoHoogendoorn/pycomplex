

import numpy as np
import numpy.testing as npt

from pycomplex.stencil.complex import StencilComplex
from pycomplex.stencil.linear_system import System


def print_sys(system):
    for r in system.A.block:
        for e in r:
            print(e.shape)

def plot_sys(system):
    import matplotlib.pyplot as plt
    plt.imshow(system.A.to_dense())
    plt.show()


def test_basic():
    complex = StencilComplex((8, 16, 4))
    system = System.canonical(complex)

    print_sys(system[1:, 1:])


def test_dense():
    complex = StencilComplex((2, 4, 6))
    system = System.canonical(complex)
    plot_sys(system[:3, 1:2])


def test_normal():
    complex = StencilComplex((2, 4, 6))
    #system = System.canonical(complex)[[0, 2], 1:2]
    system = System.canonical(complex)#[:2, :2]
    # system.A[0, 0] = 0
    # plot_sys(system)
    normal = system.normal()
    diag = normal.A.diagonal()

    d = np.block([e.flatten() for e in diag.block])

    s = np.diag(normal.A.to_dense())
    # FIXME: see if it still passes when we add nontrivial scalings
    npt.assert_allclose(s, d)
    # plot_sys(normal)
