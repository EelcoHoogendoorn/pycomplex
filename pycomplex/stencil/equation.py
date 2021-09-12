
from cached_property import cached_property

import numpy as np
import scipy.sparse.linalg

from pycomplex.stencil.block import BlockArray


class Equation(object):
    """Wraps a linear system with some solver-specific logic"""
    def __init__(self, system):
        self.system = system

    def residual(self, x, y):
        return (self.system.A * x - self.system.B * y)

    def smooth(self):
        raise NotImplementedError

    def solve(self, y, x0=None, **kwargs):
        """Default solver; minres on the normal equation is a pretty good default
        but this can be overridden for specific problems
        """
        return self.solve_minres_normal(y=y, x0=x0, **kwargs)

    @cached_property
    def normal(self):
        """Return normal equations corresponding to self"""
        return NormalEquation(self.system.normal())

    @cached_property
    def normal_ne(self):
        """Return normal equations corresponding to self"""
        return NormalEquation(self.system.normal_ne())

    def solve_minres(self, y, x0=None, **kwargs):
        A = self.system.A.aslinearoperator()
        b = (self.system.B * y).to_dense()
        x, _ = scipy.sparse.linalg.minres(A=A, b=b, x0=x0.to_dense(), **kwargs)
        return x0.from_dense(x)

    def solve_minres_normal(self, y, x0=None, **kwargs):
        """Works both for zero-conduction and zero-resistance zero-order problems"""
        if x0 is None:
            x0 = y * 0
        return self.normal.solve_minres(y, x0, **kwargs)

    def solve_qmr(self, y, x0, **kwargs):
        """Does not seem to work at all?
        even for uniform coefficient problem returns with 0 solution immediately"""
        A = self.system.A.aslinearoperator()
        b = (self.system.B * y).to_dense()
        x, _ = scipy.sparse.linalg.qmr(A=A, b=b, x0=x0.to_dense(), **kwargs)
        return x0.from_dense(x)


class NormalSmoothEquation(Equation):
    """Equation object set up to be smoothed through its normal equations

    For performance and numerical stability reasons this makes it preferred that
    the parent system is itself a first-order system
    """

    @cached_property
    def inverse_normal_diagonal(self) -> BlockArray:
        """Inverse diagonal of the normal equations; needed for jacobi iteration

        Notes
        -----
        This is somewhat expensive to compute for a stencil operator,
        requiring multiple evaluations of the operator with a checkered pattern
        """
        return self.normal.system.A.diagonal().invert()

    @cached_property
    def inverse_normal_ne_diagonal(self) -> BlockArray:
        """Inverse diagonal of the normal equations; needed for jacobi iteration

        Notes
        -----
        This is somewhat expensive to compute for a stencil operator,
        requiring multiple evaluations of the operator with a checkered pattern
        """
        return (self.normal_ne.system.A.diagonal() + 1e-16).invert()


    @cached_property
    def projected_rows(self):
        q = []
        for ri in range(len(self.system.L)):
            row = self.system.A.__getslice__((slice(None), slice(ri, ri+1)))
            N = row.transpose() * row
            q.append(N)
        return q

    @cached_property
    def inverse_sg_normal_diagonal(self):
        """Inverse diagonal of normal equations formed on a per-block-row-equation basis"""
        return BlockArray([pr.diagonal()[0] for pr in self.projected_rows]).invert()

    @cached_property
    def AT(self):
        return self.system.A.transpose()

    def jacobi(self, x, y, relaxation: float=1) -> BlockArray:
        """Jacobi iteration."""
        residual = self.AT * self.residual(x, y)
        # FIXME: keeping A.T factored out of the residual calc is more efficient
        # In factored calc we have twice the cost of the first order system; otherwise once first order and once second order
        # residual = self.normal.residual(x, y)
        # print(residual.abs().sum())
        # FIXME: loop fusion on expressions like this would also be great from memory bandwidth perspective
        return x - self.inverse_normal_diagonal * residual * (relaxation / 2)

    def NR_BLOCK_GS(self, x, y):
        """Perform jacobi iteration in blockwise fashion on normal equations

        We update unknowns in x one at a time

        References
        ----------
        The following gives a decent exposition of this topic
        https://www-users.cs.umn.edu/~saad/PS/iter3.pdf
        """
        # FIXME: `by` needs to be computed only once; can lift it out of this function
        x = x * 1
        by = self.normal.system.B * y
        for ri, r in enumerate(self.system.R):
            # compute residual of normal equation for a given block row
            residual = sum(e * xc for e, xc in zip(self.normal.system.A[ri, :], x.block)) - by[ri]
            # residual = self.normal.system.A.__getslice__(slice(ri, ri + 1)) * x - by[ri]
            x[ri] = x[ri] - self.inverse_normal_diagonal[ri] * residual * (1/2)
        return x

    def NE_BLOCK_GS(self, x, y):
        """Relax on a row-by-row basis wrt the original system"""
        by = self.system.B * y
        DI = self.inverse_normal_ne_diagonal
        x = x * 1
        for ri, l in enumerate(self.system.L):
            residual = sum(a * xc for a, xc in zip(self.system.A[ri, :], x.block)) - by[ri]
            for ci, xc in enumerate(x.block):
                xc[:] += self.system.A[ri, ci].T * (DI[ri] * residual * (1 / 2))
        return x

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
            # x = self.NR_BLOCK_GS(x, y)
            x = self.jacobi(x, y, s)
        return x

    def smooth(self, x: BlockArray, y: BlockArray, base=0.5):
        """Basic smoother; inspired by time integration of heat diffusion equation

        Notes
        -----
        base relaxation rate needs to be lower in case of a multi-block jacobi
        not yet fully understood; 0.75 no longer stable; 0.7 is

        """
        knots = np.linspace(1, 4, 3, endpoint=True) * base
        return self.overrelax(x, y, knots)


class NormalEquation(Equation):
    def solve(self, y, x0=None, **kwargs):
        if x0 is None:
            x0 = y * 0
        return self.solve_minres(y=y, x0=x0, **kwargs)
