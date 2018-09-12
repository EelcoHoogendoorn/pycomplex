"""Dump things here to make self contained diffusion example



"""

from cached_property import cached_property

import numpy as np

from pycomplex.stencil.complex import StencilComplex2D
from pycomplex.stencil.operator import DiagonalOperator
from pycomplex.stencil.block import BlockArray
from pycomplex.stencil.linear_system import System
from pycomplex.stencil.equation import NormalSmoothEquation
from pycomplex.stencil.multigrid import MultiGridEquation


class Diffusion(NormalSmoothEquation, MultiGridEquation):
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

    # FIXME: complex.scale still completely unused

    def __init__(self, system, fine, fields):
        self.system = system
        self.fine = fine
        self.fields = fields

    @property
    def complex(self):
        return self.system.complex

    @classmethod
    def formulate(cls, complex, fields, fine=None):
        """

        Parameters
        ----------
        complex
        fields
        fine

        Returns
        -------

        Notes
        -----
        [* P, d * C] [T] = [S]    * P T    + * div(C * F)  = S
        [* d, * R  ] [F]   [0]    * div(T) + * R F         = 0
        """
        def DO(d):
            return DiagonalOperator(d, d.shape)
        system = System.canonical(complex)[:2, :2]
        # overwrite diagonal; set constraint term
        system.A[0, 0] = system.A[0, 0] * DO(-fields['constraint'])
        # field effect can be driven to zero in either equation
        system.A[0, 1] = system.A[0, 1] * DO(fields['conductance'])
        system.A[1, 1] = system.A[1, 1] * DO(fields['resistance'])
        return cls(system, fine, fields)

    @cached_property
    def coarse(self):
        """Coarse variant of self"""
        return type(self).formulate(
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
    # also, full mg-cycle coarsens rhs first, which is certainly in dual space
    # that is, as long as B operator equals identity
    def restrict(self, y):
        return BlockArray([
            self.complex.coarse.hodge[n] * (self.complex.coarsen[n] * (self.complex.hodge[n].I * b))
            for n, b in zip(self.system.L, y.block)
        ])
    def interpolate(self, x):
        return BlockArray([self.complex.refine[n] * b for n, b in zip(self.system.R, x.block)])


def rect(complex, pos, size):
    r = complex.topology.form(0)
    r[0, pos[0]-size[0]:pos[0]+size[0], pos[1]-size[1]:pos[1]+size[1]] = 1
    return r

def circle(complex, pos, radius):
    p = complex.primal_position[0]
    d = np.linalg.norm(p - pos, axis=-1) - radius
    return 1 / (1 + np.exp(d * 8))


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

    complex = StencilComplex2D.from_shape((128, 128))

    # setup simple source and sink
    source = complex.topology.form(0)

    mid = 8
    ext = 3
    sep = 4




    # source = circle(complex, [8, 8], 4)
    source = rect(complex, [64, 64], [32, 16])
    source -= rect(complex, [32, 64], [32, 8])
    source = np.clip(source, 0, 1)
    constraint = 1 - source

    # complex.plot_0(source)
    # complex.plot_2(complex.topology.averaging_operators_0[2] * source)
    # show_plot()

    conductance = complex.topology.form(1, init='ones')
    conductance[:, :, 64:] *= 1e-1
    resistance = complex.topology.form(1, init='ones')
    # resistance[:, :, 64:] *= 1e-1
    # FIXME: ah; the reason we cannot lower conductance in this manner is diagonal dominance of jacobi of course!
    # FIXME: what can be done about this? absorb scaling into unknown using (left?)right-preconditioner?
    # FIXME: also there is the question of overall balance of jacobi equations;
    # just because we tune a coefficient does not mean we intend to give the equation more weight
    # FIXME: would like to have sum of abolute coefficients to judge diagonal dominance; but not sure if easy to get from operator
    # NOTE: gauss-seidel does not require diagonal dominance, but SPD will do
    # seems like block-gauss-seidel may address both dominance and balance concerns?
    # seems like we have working block SG;
    # but it seems to have the same behavior as jacobi in this respect...
    # so is it not a problem with dominance, or do with still have dominance problems inside the diagonal block that we do jacobi on?
    # also; would the transpose algorithm have better results? should be promising for under-determined system;
    # which is what we are moving towards with the problematic changes


    fields = {
        'constraint': constraint * 2e-1,
        'conductance': conductance,
        'resistance': resistance,
    }

    system = Diffusion.formulate(complex, fields)

    D = system.inverse_sg_normal_diagonal
    print()
    rhs = BlockArray([-source, complex.topology.form(1)])

    hierarchy = system.hierarchy(levels=5)
    from pycomplex.stencil.multigrid import solve_full_cycle
    from time import time
    t = time()
    # x = system.solve_minres_normal(rhs, x0=rhs * 0)
    x = solve_full_cycle(hierarchy, rhs, iterations=10)
    print()
    print(time() - t)

    complex.plot_0(x[0])

    show_plot()
