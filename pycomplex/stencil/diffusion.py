from cached_property import cached_property

from pycomplex.stencil.block import BlockArray
from pycomplex.stencil.equation import NormalSmoothEquation
from pycomplex.stencil.linear_system import System
from pycomplex.stencil.multigrid import MultiGridEquation
from pycomplex.stencil.operator import DiagonalOperator


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
        """Diffusion formulation using potential formulation, in first order form

        Parameters
        ----------
        complex
        fields
        fine

        Returns
        -------
        cls

        Notes
        -----
        [* P, d * C] [T] = [S]    * P T     + * div(C * F)  = S
        [* d, * R  ] [F]   [0]    * grad(T) + * R F         = 0


        """
        def DO(d):
            return DiagonalOperator(d, d.shape)
        system = System.canonical(complex)[:2, :2]  # potential is the primal 0-form
        # overwrite diagonal; set constraint term, constraining the potential
        system.A[0, 0] = system.A[0, 0] * DO(-fields['constraint'])
        # field effect can be driven to zero in either equation
        system.A[0, 1] = system.A[0, 1] * DO(fields['conductance'])     # div
        system.A[1, 1] = system.A[1, 1] * DO(fields['resistance'])      # grad
        system.A[1, 0] = DO(fields['foo']) * system.A[1, 0]
        return cls(system, fine, fields)

    @classmethod
    def formulate_flux_only(cls, complex, fields, fine=None):
        """Diffusion formulated without reference to potential

        This precludes implementing dynamics specifically referencing said potential


        Parameters
        ----------
        complex
        fields
        fine

        Returns
        -------

        Notes
        -----
        [d * C] [F] = [S]    * div(F * C)  = S
        [* d R]       [0]    * curl(F * R) = 0
        """
        def DO(d):
            return DiagonalOperator(d, d.shape)
        system = System.canonical(complex)[[0, 2], 1:2]
        # field effect can be driven to zero in either equation
        system.A[0, 0] = system.A[0, 0] * DO(fields['conductance'])     # div
        system.A[1, 0] = system.A[1, 0] * DO(fields['resistance'])      # curl relation
        return cls(system, fine, fields)

    @classmethod
    def formulate_all(cls, complex, fields, fine=None):
        """does having both potential and curl help when resistance goes to zero?
        probably not; both equations are zeroed out; especially if a region is zeroed out,
        should give zero equations in normal equations regardless

        however, seems like overall smoother action would be different. in a good way?
        """
        def DO(d):
            return DiagonalOperator(d, d.shape)
        system = System.canonical(complex)[:3, :2]
        # overwrite diagonal; set constraint term
        # field effect can be driven to zero in either equation
        system.A[0, 0] = system.A[0, 0] * DO(-fields['constraint'])
        system.A[0, 1] = system.A[0, 1] * DO(fields['conductance']) # div
        system.A[1, 1] = system.A[1, 1] * DO(fields['resistance'])  # grad diag
        system.A[2, 1] = system.A[2, 1] * DO(fields['resistance'])  # curl relation
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
                'foo': self.complex.coarsen[1] * self.fields['foo'],
            }
        )

    # NOTE: we restrict the residual, and interpolate error-corrections
    # FIXME: interpolate, refine, prolongate... need to make up our mind sometime
    # note that solution is in terms of primal forms
    # residual of first order equation tends to be dual forms
    # but residual of normal equations ought to be in primal space too?
    # also, full mg-cycle coarsens rhs first, which is certainly in dual space
    # that is, as long as B operator equals identity

    def restrict_dual(self, y):
        return BlockArray([
            self.complex.coarse.hodge[n] * (self.complex.coarsen[n] * (self.complex.hodge[n].I * b))
            for n, b in zip(self.system.L, y.block)
        ])
    def restrict_primal(self, y):
        return BlockArray([self.complex.coarsen[n] * b for n, b in zip(self.system.L, y.block)])


    def restrict(self, y):
        return self.restrict_dual(y)
    def interpolate(self, x):
        return BlockArray([self.complex.refine[n] * b for n, b in zip(self.system.R, x.block)])