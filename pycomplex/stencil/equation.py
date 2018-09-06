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

    def jacobi(self, x, y, relaxation: float=1) -> BlockArray:
        """Jacobi iteration."""
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