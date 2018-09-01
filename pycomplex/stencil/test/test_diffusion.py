"""Dump things here to make self contained diffusion example

refactor later
need to rewrite mg code to accept blockarrays
"""

from pycomplex.stencil.complex import StencilComplex2D
from pycomplex.stencil.linear_system import System


def plot_sys(system):
    import matplotlib.pyplot as plt
    plt.imshow(system.A.to_dense())
    plt.show()


def test_diffusion():
    """setup and solve 2d diffusion problem

    solve by means of dec-multigrid; without variable elimination
    this is likely to be less efficient but generelizes better to more complex problems
    """
    complex = StencilComplex2D((16, 16))
    system = System.canonical(complex)[:2, :2]
    system.A[0, 0] = 0  # nothing acts on temperature directly; just want to constrain divergence

    # setup simple source and sink
    source = complex.form(0)

    """    
    pin value at boundary to zero? infact not obvious how to set these constraints
    0 * T + d flux = Q
    just introduce some nonzeros in diagonal of T?
    or need we model auxiliary constraint equations?
    no not really have room for hard constraints though;
    either redundant or need to be satisfied in least square sense
    seems like we can add relation between divergence and temperature
    cannot relate flux and temp directly
    but divergence and temp relation over distance can have same effect
    the higher weight we set on temp, the more the equation becomes dominated to just being a temp constraint
    """

    """
    when does it make sense to eliminate a variable, versus solving it in first order normal form?
    sometimes there is no choice; can only eliminate if there is an invertable diagonal.
    examples of impossible elimination are stokes-pressure and EM-field

    elimination seems more efficient a priori. but in the limit to incompressibility probably a bad idea
    should we eliminate pressure in near-incompressible, or rubber-like material?
    or should we eliminate flux in reservoir simulation?
    of course with block diagonal it is still trivial to solve these variables with one jacobi step,
    but simultanious updates in other vars mean making residual zero still isnt trivial,
    meaning there is something to move to the coarse grid.
    
    which makes an interesting point; we can do per-variable gauss-seidel
    that way, the elimination candidate would have zero residual
    is this equivalent to variable elimination?

    """

    mid = 8
    ext = 2
    sep = 4
    source[0, mid-ext-sep:mid+ext-sep, mid-ext:mid+ext] = -1
    source[0, mid-ext+sep:mid+ext+sep, mid-ext:mid+ext] = +1

    system.rhs[0] = source
    normal = system.normal()
    # plot_sys(normal)

    diagonal = normal.A.diagonal()
    print(diagonal[0])
    print(diagonal[1])


    x = normal.allocate_x()

    from pycomplex.stencil.linear_system import Equation
    eq = Equation(normal)


    for i in range(300):
        x = eq.smooth(x, normal.rhs)


    complex.plot_0(x[0])
    # complex.plot_0(system.rhs[0])




