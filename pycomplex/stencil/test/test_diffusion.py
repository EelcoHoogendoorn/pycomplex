"""Dump things here to make self contained diffusion example

refactor later
need to rewrite mg code to accept blockarrays
"""

from cached_property import cached_property

import numpy as np

from pycomplex.stencil.complex import StencilComplex2D
from pycomplex.stencil.linear_system import System


def plot_sys(system):
    import matplotlib.pyplot as plt
    plt.imshow(system.A.to_dense())
    plt.show()


def build_hierarchy(equation, depth):
    """Build a hierarchy for a given equation object

    Parameters
    ----------
    equation

    Returns
    -------
    List[Equation]
    """
    equations = [equation]
    for d in range(depth):
        coarse = equations[-1].system.complex.coarse
        # FIXME: need logic here for coarsening all fields that occur in the system
        # need this be encapsulated by the equation object? does that mean specific equation should be a subclass
        # complex object contains logic for applying scale to hodges
        # but we need similar logic for other fields introduced at the system/equation level
        # we really need to reconsider the fields at each level, considering constucting the coarse operator
        # as some kind of petrov-galerkin method is unthinkable for a stencil based method
        # could maintain each field as a seperate symbolic expression;
        # coarse variant then just a matter of swapping out hodge and fields.
        # kindof an optimization tho and more complex code
        system = None
        eq = equations[-1].copy(system=system)
        equations.append(eq)


from pycomplex.stencil.multigrid import MGEquation
from pycomplex.stencil.block import BlockArray


class Diffusion(System, MGEquation):
    """0-form based diffusion, based on first-order operator logic

    constraint field has physical analogy; if domain is 2d, constraint is thermal connection in z-layer
    could have seperate setpoint and constraint fields,
    but for now all setpoints are zero so it drops out

    Add R and C mask fields as well?
    driving conductance to zero changes divergence operator interpretation
    driving resistance to zero changes gradient operator interpretation

    is there a difference between a domain mask and a regular parameter?
    we do not scale the field afterwards with regular parameter I suppose;
    or do we? can we just think of them as distinguishing flux and gradient?

    Note that having dual R and C fields need not be bad; it also gives us control over
    parallel vs series coarsening
    """

    @classmethod
    def formulate(cls, complex, fields, fine=None):
        self = cls.canonical(complex)[:2, :2]

        # overwrite diagonal; set constraint term
        self.A[0, 0] = fields['constraint']
        # field effect can be driven to zero in either equation
        self.A[0, 1] = self.A[0, 1] * fields['conductance']
        self.A[1, 1] = self.A[1, 1] * fields['resistance']
        # FIXME: source should not be part of equations anymore, no? will get coarsened as part of residual
        # self.rhs[0] = fields['source']
        self.fine = fine
        self.fields = fields
        return self

    @cached_property
    def coarse(self):
        """Coarse variant of self"""
        return type(self)(
            complex=self.complex.coarse,
            fine=self,
            fields={
                'constraint': self.complex.coarsen[0] * self.fields['constraint'],
                'conductance': self.complex.coarsen[1] * self.fields['conductance'],
                'resistance': self.complex.coarsen[1] * self.fields['resistance'],
            }
        )

    # NOTE: we restrict the residual, and interpolate error-corrections
    # FIXME: interpolate, refine, prolongate... need to make up our mind sometime
    # note that solution is in terms of primal forms
    # residual of first order equation tends to be dual forms
    # but residual of normal equations ought to be in primal space too?
    def restrict(self, y):
        return BlockArray([self.complex.coarsen[n] * b for n, b in zip(self.L, y.block)])
    def interpolate(self, x):
        return BlockArray([self.complex.refine[n] * b for n, b in zip(self.R, x.block)])


class StencilForm(object):
    """Lets see how this feels; form object... not sure yet"""
    def __init__(self, degree, shape, dual=False):
        self.degree = degree
        self.data = np.zeros(shape)
        self.dual = dual

    def __add__(self, other):
        """plain operators passed on to data array, given that degree and duality match"""


def test_diffusion(show_plot):
    """setup and solve 2d diffusion problem

    solve by means of dec-multigrid; without variable elimination
    this is likely to be less efficient but generalizes better to more complex problems;
    that is zero-resistance term precludes elimination
    and even when it is not precluding elimination,
    dividing by small values leads to poor conditioning.
    remains to be seen how this works out in first order system exactly,
    but the suspicion is that it will work much better

    """
    complex = StencilComplex2D.from_shape((16, 16))

    # setup simple source and sink
    source = complex.topology.form(0)

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
    is this equivalent to variable elimination, if done in the right order?
    
    not really. note that variable elimination fails for diffusion in areas where diagonal goes to zero;
    that is, where gradients in the 0-form are constrained to be 0. using normal equation method, 
    like other zero-diagonal terms, this is not an issue, and we can solve for this flat region,
    which is effectively a super-conductor.
    
    does it go both ways; does normal equation enable regions of zero-conductivity too?
    here, diagonal goes to infinity; or off-diag to zero. no matter the gradient, there is no flux
    this would have been possible with var-eliminated second order laplace already.
    
    but so even for a simple diffusion problem, first order form has objective benefits, at least in terms of generality 
    
    would there by any benefit in having both a conductivity and resistance field,
    and coarsening both independently?
    """

    mid = 8
    ext = 3
    sep = 4


    def rect(pos, size):
        r = complex.topology.form(0)
        r[0, pos[0]-size[0]:pos[0]+size[0], pos[1]-size[1]:pos[1]+size[1]] = 1
        return r

    def circle(pos, radius):
        p = complex.primal_position[0]
        d = np.linalg.norm(p - pos, axis=-1) - radius
        return 1 / (1 + np.exp(d * 8))


    source = circle([8, 8], 4)
    constraint = 1 - source

    complex.plot_0(source)
    complex.plot_2(complex.topology.averaging_operators_0[2] * source)
    show_plot()


    fields = {
        'constraint': constraint,
        'conduction': np.ones_like(source),
        'resistance': np.ones_like(source),
    }

    system = Diffusion(complex, fields)

    from pycomplex.stencil.equation import NormalSmoothEquation
    eq = NormalSmoothEquation(system)

    x = eq.solve(source)

    complex.plot_0(x[0])
    complex.coarse.plot_0(complex.coarsen[0](x[0]))
    complex.plot_0(complex.refine[0](complex.coarsen[0](x[0])))

    # complex.plot_0(system.rhs[0])
    show_plot()
