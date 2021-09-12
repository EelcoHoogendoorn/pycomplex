"""Linear system specialized to the stencil case

Remains to be seen how much of this can be unified into a more abstract base class shared with non-stencil code;
dont worry about it yet, can unify things when we know what they look like
"""

import numpy as np

from pycomplex.stencil.block import BlockOperator, BlockArray
from pycomplex.stencil.complex import StencilComplex
from pycomplex.topology.util import index_dtype


class System(object):

    def __init__(self, complex, A, B=None, L=None, R=None):
        """

        Parameters
        ----------
        A : block stencil operator
            left matrix in A x = B y
        B : block stencil operator, optional
            right matrix in A x = B y
        L : List[int]
            primal form associated with rows of the system
            or space of left-multiplication of A
        R : List[int]
            primal form associated with columns of the system
            or space of right-multiplication of A
        """
        self.complex = complex
        self.A = A
        self.B = B
        self.L = np.array(L)
        self.R = np.array(R)

    def allocate_x(self):
        return BlockArray([self.complex.form(n) for n in self.R])

    def allocate_y(self):
        return BlockArray([self.complex.form(n) for n in self.L])

    @classmethod
    def canonical(cls, complex: StencilComplex):
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

        Tp = complex.topology.primal
        Td = complex.topology.dual
        N = complex.n_dim + 1

        NE = complex.n_elements
        A = BlockOperator.zeros(NE, NE)
        # FIXME: is diagonal of hodges a better default? does it make residual-vs-rhs mg logic more consistent?
        B = BlockOperator.identity(NE)

        PD = complex.hodge
        # from pycomplex.stencil.operator import DiagonalOperator
        # PD = [h * DiagonalOperator(np.random.random(h.shape[0]) + 1, h.shape[0]) for h in PD] # randomize hodges
        for i, (tp, td) in enumerate(zip(Tp, Td)):
            A[i + 1, i] = PD[i + 1] * tp
            A[i, i + 1] = td * PD[i + 1]

        # put hodges on diag by default; easier to zero out than to fill in
        for i in range(N):
            A[i, i] = PD[i]

        LR = np.arange(N, dtype=index_dtype)
        return cls(complex, A=A, B=B, L=LR, R=LR)

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
        )

    def normal(self):
        """Form normal equations by left-multiplication with A.T

        Returns
        -------
        System
            normal equations belonging to self
        """
        AT = self.A.transpose()
        return self.copy(
            A=(AT * self.A).simplify(),
            B=(AT * self.B).simplify(),
            R=self.R,
            L=self.R,   # NOTE: this is the crux of forming normal equations
        )

    def normal_ne(self):
        """Transform to symmetric problem by substituting x = A.T * x'

        After solving these equations, the solution to the original problem
        can be obtained by multiplying by A.T again

        Returns
        -------
        System

        Notes
        -----
        This may have an advantage over normal equations,
        if a system has more unknowns than equations; which is basically never?
        However, there might be some conditioning aspects too that id like to investigate.
        """
        AT = self.A.transpose()
        return self.copy(
            A=(self.A * AT).simplify(),
            B=self.B,
            R=self.L,
            L=self.L,
        )

    def eliminate(self):
        """Eliminate equations containing an invertable diagonal"""
        raise NotImplementedError

    @staticmethod
    def laplace_beltrami(n: int):
        raise NotImplementedError

    def restrict_dual(self, y):
        return BlockArray([
            self.complex.coarse.hodge[n] * (self.complex.coarsen[n] * (self.complex.hodge[n].I * b))
            for n, b in zip(self.L, y.block)
        ], ndim=1)

    def restrict_primal(self, y):
        return BlockArray([
            self.complex.coarsen[n] * b
            for n, b in zip(self.R, y.block)
        ], ndim=1)


