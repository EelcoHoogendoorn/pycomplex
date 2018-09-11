import numpy as np
from cached_property import cached_property

from pycomplex.stencil.block import BlockArray


class Equation(object):
    """Wraps a linear system with some solver-specific logic"""
    def __init__(self, system):
        self.system = system

    def residual(self, x, y):
        return (self.system.A * x - self.system.B * y)

    def smooth(self):
        raise NotImplementedError

    def solve(self, y, x=None, iterations=10):
        if x is None:
            x = y * 0
        for i in range(iterations):
            x = self.smooth(x, y)
        return x

    @cached_property
    def normal(self):
        """Return normal equations corresponding to self"""
        return NormalEquation(self.system.normal())

    def solve_minres_normal(self, y, x0):
        A = self.normal.A.aslinearoperator()
        b = (self.normal.system.B * y).to_dense()
        import scipy.sparse.linalg
        x = scipy.sparse.linalg.minres(A=A, b=b, x0=x0.to_dense())
        return x0.from_dense(x)

    def solve_qmr(self, y, x0):
        A = self.A.aslinearoperator()
        b = (self.system.B * y).to_dense()
        import scipy.sparse.linalg
        x = scipy.sparse.linalg.qmr(A=A, b=b, x0=x0.to_dense())
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

    def block_jacobi_sg(self, x, y):
        """Perform jacobi iteration in blockwise fashion

        We update unknowns in x one at a time
        """
        # FIXME: `by` needs to be computed only once; can lift it out of this function
        x = x * 1
        by = self.normal.system.B * y
        for ri, (r, pr, d) in enumerate(zip(self.system.R, self.projected_rows, self.inverse_sg_normal_diagonal)):
            # compute residual of normal equation for a given row
            residual = self.normal.system.A[ri, :] * x - by[ri]
            x[ri] -= self.inverse_sg_normal_diagonal[ri] * residual * (1/2)
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
            x = self.jacobi(x, y, s)
        return x

    def smooth(self, x: BlockArray, y: BlockArray, base=0.6):
        """Basic smoother; inspired by time integration of heat diffusion equation

        Notes
        -----
        base relaxation rate needs to be lower in case of a multi-block jacobi
        not yet fully understood; 0.75 no longer stable; 0.7 is

        """
        knots = np.linspace(1, 4, 3, endpoint=True) * base
        return self.overrelax(x, y, knots)


class NormalEquation(Equation):
    pass