import numpy as np
from cached_property import cached_property
from scipy import ndimage

from pycomplex.stencil.complex import StencilComplex2D


class StencilComplexSimple2d(StencilComplex2D):

    """only implement a second order stencil on 0-forms"""

    @cached_property
    def stencil(self):
        return np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ])

    @cached_property
    def laplace_0(self):
        def inner(f0):
            return ndimage.convolve(f0[0], weights=-self.stencil, mode='wrap')[None, ...]
        # FIXME: use scipy.ndimage.laplace?
        return inner


class SimpleHeatEquation2d(object):


    def __init__(self, complex, constraint, source):
        assert isinstance(complex, StencilComplexSimple2d)
        self.complex = complex
        self.source = source
        self.constraint = constraint

    @cached_property
    def inverse_diagonal(self):
        # FIXME: does scale factor in? surely it does somewhere?
        D = np.abs(self.complex.stencil).sum() / 2
        diag = self.complex.form(0, init='ones') * D + self.constraint
        return 1 / diag

    def residual(self, x, y):
        return self.complex.laplace_0(x) + self.constraint * x - y

    def jacobi(self, x, y, relaxation: float=1):
        """Jacobi iteration. """
        residual = self.residual(x, y)
        print(np.abs(residual).sum())
        return x - self.inverse_diagonal * residual * (relaxation / 2)

    def overrelax(self, x, y, knots):
        """overrelax, forcing the eigencomponent to zero at the specified overrelaxation knots

        Parameters
        ----------
        x : ndarray
        knots : List[float]
            for interpretation, see self.descent
        """
        for s in knots:
            x = self.jacobi(x, y, s)
        return x

    def smooth(self, x, y, base=1):
        """Basic smoother; inspired by time integration of heat diffusion equation

        Notes
        -----
        base relaxation rate needs to be lower in case of a multi-block jacobi
        not yet fully understood

        """
        knots = np.linspace(1, 4, 3, endpoint=True) * base
        return self.overrelax(x, y, knots)

    def solve(self, y, x=None, iterations=10):
        if x is None:
            x = y * 0
        for i in range(iterations):
            x = self.smooth(x, y)
        return x

    def restrict(self, x):
        return self.complex.coarsen[0](x)

    def interpolate(self, x):
        return self.complex.refine[0](x)

    def coarsen(self):
        """Coarsen the equation object"""
        source = self.restrict(self.source)
        constraint = self.restrict(self.constraint)
        return type(self)(
            complex=self.complex.coarse,
            source=source,
            constraint=constraint
        )