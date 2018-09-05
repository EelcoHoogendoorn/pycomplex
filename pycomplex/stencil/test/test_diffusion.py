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


def build_hierarchy(equation, depth):
    """Build a hierarchy for a given equation object

    The the complex of the equation, split that, and hook it up

    equation.system.complex

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
        eq = equations[-1].copy(system=system.coarse())
        equations.append(eq)


class Diffusion(System):
    """0-form based diffusion

    constraint field has physical analogy; if domain is 2d, constraint is thermal connection in z-layer
    could have seperate setpoint and constraint fields,
    but for now all setpoints are zero so no point

    conduction can be set on either 0 or 1 form
    """

    @classmethod
    def formulate(cls, complex, fields):
        self = cls.canonical(complex)[:2, :2]

        self.A[0, 0] = fields['constraint']
        self.A[1, 1] = self.A[1, 1] * fields['resistance']
        self.rhs[0] = fields['source']

        self.fields = fields
        return self

    def coarse(self):
        """Create coarse variant of self"""
        return type(self).formulate(self.coarse, self.fields)

    def coarsen(self, y):
        """coarsen a residual vector; one block for each equation"""
        l = self.L
        assert y.shape == (2,)
        from pycomplex.stencil.block import BlockArray
        C = self.complex.coarsen
        return BlockArray([C[n] * b for b, n in zip(y.blocks, l)])

    def refine(self, x):
        """refine solution vector"""
        r = self.R
        assert y.shape == (2,)
        from pycomplex.stencil.block import BlockArray
        R = self.complex.refine
        return BlockArray([R[n] * b for b, n in zip(y.blocks, r)])

    def downsample(self, fields):
        """Down sample all required fields to a coarser level"""
        return {
            'resistance': self.complex.coarsen[1] * fields['resistance'],   # FIXME: constant resistance would be fine for starters?
            'constraint': self.complex.coarsen[0] * fields['constraint'],
            'source': self.complex.coarsen[0] * fields['source'],   # FIXME: source should be treated as a dual form?
        }

    # FIXME: need to find a nice way to track dimension of fields used, like we do for equations and unknowns
    # def downsample(self, fields):
    #     """Down sample all required fields to a coarser level
    #
    #     Parameters
    #     ----------
    #     fields: Dict[str, Tuple[int, form]]
    #         primal form describing a field over the domain, with an int denoting its degree
    #     """
    #     # FIXME: move to baseclass?
    #     # FIXME: do we need control over coarsening field versus coarsening its inverse?
    #     # or should this be controlled by the place in which the effect is inserted?
    #     # FIXME: also... maybe it is time for a dedicated form class, that tracks its degree, duality
    #     # give our operators an input and output 'signature', instead of a shape?
    #     return {k: self.complex.coarsen[n] * f for k, (n, f) in fields.items()}




def test_diffusion(show_plot):
    """setup and solve 2d diffusion problem

    solve by means of dec-multigrid; without variable elimination
    this is likely to be less efficient but generelizes better to more complex problems
    """
    complex = StencilComplex2D((16, 16))
    system = System.canonical(complex)[:2, :2]
    print()
    print(system.A)
    print()
    system.A[0, 0] = 0  # nothing acts on temperature directly; just want to constrain divergence
    print(system.A)
    print()

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
    source[0, mid-ext-sep:mid+ext-sep, mid-ext:mid+ext] = -1
    source[0, mid-ext+sep:mid+ext+sep, mid-ext:mid+ext] = +1

    print(source[0])
    sample = complex.sample_0(source, [[4, 4], [4, 4.5], [4, 5]])

    # source[0, 0, :] = 10
    sample = complex.sample_0(source, [[0, 0], [-0.5, 0], [-1, 0]])

    sample = complex.sample_0(source, [[15, 0], [15.5, 0], [16, 0], [16.5, 0], [17, 0]])
    sample = complex.sample_0(source, [[-1, 0], [-0.5, 0], [0, 0], [0.5, 0], [1, 0]])

    print(sample)

    system.rhs[0] = source


    from pycomplex.stencil.linear_system import NormalSmoothEquation
    eq = NormalSmoothEquation(system)

    x = eq.solve(system.rhs)

    complex.plot_0(x[0])
    complex.coarse.plot_0(complex.coarsen[0](x[0]))
    complex.plot_0(complex.refine[0](complex.coarsen[0](x[0])))

    # complex.plot_0(system.rhs[0])
    show_plot()
