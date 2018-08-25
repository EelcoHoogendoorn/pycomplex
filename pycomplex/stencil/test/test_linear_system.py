
from pycomplex.stencil.complex import StencilComplex
from pycomplex.stencil.linear_system import System


def print_sys(system):
    for r in system.A.block:
        for e in r:
            print(e.shape)


def test_basic():
    complex = StencilComplex((8, 16, 4))
    system = System.canonical(complex)

    print_sys(system[1:, 1:])


def test_normal():
    complex = StencilComplex((8, 16, 4))
    system = System.canonical(complex)
    normal = system.normal()
    print_sys(normal)