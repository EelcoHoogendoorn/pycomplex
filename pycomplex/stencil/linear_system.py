"""Linear system specialized to the stencil case

Remains to be seen how much of this can be unified into a more abstract base class shared with non-stencil code;
dont worry about it yet, can unify things when we know what they look like
"""

import numpy as np
from cached_property import cached_property

from pycomplex.stencil.complex import StencilComplex
from pycomplex.stencil.block import BlockOperator, BlockArray
from pycomplex.topology.util import index_dtype


class System(object):

    def __init__(self, complex, A, B=None, L=None, R=None, rhs=None):
        """

        Parameters
        ----------
        A : block stencil operator
            left matrix in A x = B y
        B : block stencil operator, optional
            right matrix in A x = B y
        L : List[int]
            primal form associated with rows of the system
            or space of left-multiplication
        R : List[int]
            primal form associated with columns of the system
            or space of right-multiplication
        """
        # FIXME: is there any point in having the right-hand operator based form for stencils? seems like it, for symbolic normal equations
        self.complex = complex
        self.A = A
        self.B = B
        self.L = np.array(L)
        self.R = np.array(R)
        self.rhs = self.allocate_y() if rhs is None else rhs

    def allocate_x(self):
        return BlockArray([self.complex.form(n) for n in self.R])

    def allocate_y(self):
        return BlockArray([self.complex.form(n) for n in self.L])

    @staticmethod
    def canonical(complex: StencilComplex):
        """Set up the full cochain complex

        Parameters
        ----------
        complex : StencilComplex

        Returns
        -------
        System
            system of full cochain complex
            maps from dual to primal
            as a result the first order system is symmetric

        Examples
        --------
        for a 3d complex, the full structure is thus, and scales according to this pattern with dimension

        [ * , *δ , 0  , 0  ]
        [ d*, *  , *δ , 0  ]
        [ 0 , d* , *  , *δ ]
        [ 0 , 0  , d* , *  ]

        note that the inclusion of hodges can make rows/cols kinda unbalanced
        primal hodges scales as l^k
        and corresponding dual as l^(n-k)
        with l a characteristic edge length
        so f.i. hodge mapping from dual to primal on 2d complex scales as
        0: -2
        1: 0
        2: +2

        on 3d:
        0: -3
        1: -1
        2: +1
        3: +3

        [-3, -3, +0, +0]
        [-3, -1, -1, +0]
        [+0, -1, +1, +1]
        [+0, +0, +1, +3]

        Is this an argument in favor of working in 'dimensionless midpoint space' by default?
        worked for eschereque
        if doing so, we should not expect refinement to influence the scaling of our equations at all?
        every extterior derivative would be wrapped left and right by metric.
        multiply with k-metric, exterior derivative, and then divide by k+1 metric;
        so every operator scales as 1/l
        may also be more elegant in an mg-transfer scenario?

        note that this problem resolves itself after variable elimination to form laplace
        otoh, absence of symmetry does not resolve itself in midpoint method, after elimination
        is there a standard scaling we can apply, that rebalances this type of system without affecting symmetry?
        left/right multiplication by l^-k factor seems like itd work

        [-3, -4, +0, +0]
        [-4, -3, -4, +0]
        [+0, -4, -3, -4]
        [+0, +0, -4, -3]

        """

        Tp = complex.primal
        Td = complex.dual[::-1]
        N = complex.ndim + 1

        NE = complex.n_elements
        A = BlockOperator.zeros(NE, NE)
        B = BlockOperator.identity(NE)

        from pycomplex.stencil.operator import DiagonalOperator

        PD = complex.hodge
        # PD = [h * DiagonalOperator(np.random.random(h.shape[0]) + 1, h.shape[0]) for h in PD] # randomize hodges
        for i, (tp, td) in enumerate(zip(Tp, Td)):
            A[i, i + 1] = PD[i] * td.T
            A[i + 1, i] = A[i, i + 1].T

        # put hodges on diag by default; easier to zero out than to fill in
        for i in range(N):
            A[i, i] = PD[i]

        LR = np.arange(N, dtype=index_dtype)
        return System(complex, A=A, B=B, L=LR, R=LR)

    def copy(self, **kwargs):
        """Copy self with some constructor args overridden. Part of general functional logic"""
        import funcsigs
        args = funcsigs.signature(type(self)).parameters.keys()
        nkwargs = {}
        for a in args:
            if hasattr(self, a):
                nkwargs[a] = getattr(self, a)

        nkwargs.update(kwargs)
        c = type(self)(**nkwargs)
        c.parent = self
        return c

    def __getitem__(self, item):
        """Slice a subsystem of full cochain complex"""
        return self.copy(
            A=self.A.__getslice__(item).copy(),
            B=self.B.__getslice__([item[0], item[0]]).copy(),
            L=self.L[item[0]],
            R=self.R[item[1]],
            rhs=self.rhs.__getslice__(item[0]).copy()
        )

    def normal(self):
        """Form normal equations

        Returns
        -------
        System
            normal equations belonging to self
        """
        AT = self.A.transpose()
        return self.copy(
            A=(AT * self.A).simplify(),
            # FIXME: add premultiplication of B; dont have a fixed rhs in mg scheme
            B=(AT * self.B).simplify(),
            rhs=(AT * self.rhs),
            R=self.R,
            L=self.R,   # NOTE: this is the crux of forming normal equations
        )

    def eliminate(self):
        raise NotImplementedError


class Equation(object):
    """Wraps a linear system with some solver-specific logic"""
    def __init__(self, system):
        self.system = system

    def residual(self, x, y):
        # if self.system.B:
        return (self.system.A * x - self.system.B * y)
        # else:
        #     return (self.system.A * x - y)

    def smooth(self):
        raise NotImplementedError

    def interpolate(self, x):
        raise NotImplementedError

    def coarsen(self, x):
        raise NotImplementedError

    def solve(self, y, x=None, iterations=100):
        if x is None:
            x = y * 0
        for i in range(iterations):
            x = self.smooth(x, y)
        return x

    @cached_property
    def normal(self):
        """Return normal equations corresponding to self"""
        return NormalEquation(self.system.normal())



class NormalSmoothEquation(Equation):
    """Equation object set up to be smoothed through its normal equations"""

    @cached_property
    def inverse_normal_diagonal(self) -> BlockArray:
        """Expensive; only compute this once"""
        return self.normal.system.A.diagonal().invert()

    def jacobi(self, x, y, relaxation: float=1) -> BlockArray:
        """Jacobi iteration.

        Notes
        -----
        More efficient than Richardson?
        Scaling with an equation-specific factor might indeed adapt better to anisotropy
        Requires presence of nonzero diagonal on A, which richardson does not
        but unlike richardson, zero mass diagonal terms are fine
        But given normal equations, we always have a nonzero on the diagonal
        """
        # residual = self.system.A.transpose() * self.residual(x, y)
        # FIXME: is the below more efficient? seems like keeping A.T factored out of the residual calc is more efficient
        # In factored calc we have twice the cost of the first order system; otherwise once first order and once second order
        residual = self.normal.residual(x, y)
        print(residual.abs().sum())
        return x - self.inverse_normal_diagonal * residual * (relaxation / 2)

    def block_jacobi(self):
        """Perform jacobi iteration in blockwise fashion"""
        raise NotImplementedError

    def overrelax(self, x: BlockArray, y: BlockArray, knots):
        """overrelax, forcing the eigencomponent to zero at the specified overrelaxation knots

        Parameters
        ----------
        x : ndarray
        y : ndarray
        knots : List[float]
            for interpretation, see self.descent
        """
        for s in knots:
            x = self.jacobi(x, y, s)
        return x

    def smooth(self, x: BlockArray, y: BlockArray, base=0.5):
        """Basic smoother; inspired by time integration of heat diffusion equation

        Notes
        -----
        base relaxation rate needs to be lower in case of a multi-block jacobi
        not yet fully understood

        """
        knots = np.linspace(1, 4, 3, endpoint=True) * base
        return self.overrelax(x, y, knots)


class NormalEquation(Equation):
    pass