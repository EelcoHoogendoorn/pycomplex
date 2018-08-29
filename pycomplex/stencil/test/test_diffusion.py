"""Dump things here to make self contained diffusion example

refactor later
need to rewrite mg code to accept blockarrays
"""

from pycomplex.stencil.complex import StencilComplex
from pycomplex.stencil.linear_system import System


def test_diffusion():
    """setup and solve 2d diffusion problem

    solve by means of dec-multigrid; without variable elimination
    this is likely to be less efficient but generelizes better to more complex problems
    """
    complex = StencilComplex((64, 64))
    system = System.canonical(complex)[:2, :2]
    system.A[0, 0] = 0  # nothing acts on temperature directly; just want to constrain divergence

    # setup simple source and sink
    source = complex.form(0)




