from typing import Tuple

import numpy as np
from scipy import ndimage
from pycomplex.stencil.operator import StencilOperator, SymmetricOperator, ComposedOperator, DiagonalOperator

from pycomplex.stencil.util import generate, smoother


class StencilComplex(object):

    """Abstract factory class for stencil based dec operations and mg hierarchies

    The defining distinction of the stencil based approach is that all topology is implicit
    other than its shape it has no associated memory requirements

    make component-first or component-last configurable?

    """

    def __init__(self, shape: Tuple[int], scale=1, boundary: str = 'periodic'):
        self.boundary = boundary
        self.shape = shape
        self.scale = scale
        # these are more like properties
        self.ndim = len(shape)
        self.symbols, self.terms, self.axes, self.parities = generate(self.ndim)

    @property
    def coarse(self):
        """Construct coarse counterpart"""
        assert all(s % 2 == 0 for s in self.shape)
        return type(self)(
            boundary=self.boundary,
            shape=tuple([s // 2 for s in self.shape]),
            scale=self.scale * 2,
        )

    @property
    def fine(self):
        """Construct fine counterpart"""
        return type(self)(
            boundary=self.boundary,
            shape=tuple([s * 2 for s in self.shape]),
            scale=self.scale / 2,
        )

    def form(self, n, dtype=np.float32, init='zeros'):
        """Allocate an n-form

        Returns
        -------
        list of len n_components of ndarrays of shape self.shape
        """
        assert self.boundary == 'periodic'
        # with periodic boundaries all forms have the same shape
        components = len(self.symbols[n])
        # in more general case, different forms and different components of forms may have different shape
        if init == 'zeros':
            return np.zeros((components, ) + self.shape, dtype=dtype)
        if init == 'empty':
            return np.empty((components, ) + self.shape, dtype=dtype)
        raise Exception

    @property
    def primal(self):
        """Primal derivative operators

        Returns
        -------
        array_like, [ndim], operator
            list of stencil operators mapping primal n-form to primal n+1 form
        """
        assert self.boundary == 'periodic'
        def conv(*args, **kwargs):
            return ndimage.convolve1d(*args, **kwargs, mode='wrap')
        def corr(*args, **kwargs):
            return ndimage.correlate1d(*args, **kwargs, mode='wrap')

        def derivative(n, symbols, terms, axes, parities, func):
            """Exterior derivative operator"""
            def inner(f):
                d = self.form(n + 1, init='empty')
                # write once; update afterwards
                initialized = [False] * len(d)
                # loop over all components of the new derived form
                for c, (T, A, P) in enumerate(zip(terms, axes, parities)):
                    # loop over all terms that the new symbol is composed of
                    for t, a, p in zip(T, A, P):
                        i = symbols.index(t)
                        weights = [-1, +1] if p else [+1, -1]
                        if initialized[c]:
                            d[c] += func(f[i], axis=a, weights=weights)
                        else:
                            func(f[i], axis=a, output=d[c], weights=weights)
                            initialized[c] = True
                return d
            return inner

        return [
            StencilOperator(
                left=derivative(n, symbols, terms, axes, parities, conv),
                right=derivative(n, symbols, terms, axes, parities, corr),
                shape=(self.form(n + 1).shape, self.form(n).shape)
            )
            for n, (symbols, terms, axes, parities)
            in enumerate(zip(self.symbols, self.terms, self.axes, self.parities))
        ]

    @property
    def dual(self):
        """Dual derivative operators

        Returns
        -------
        array_like, [ndim-1], operator
            list of operators mapping dual n-form to dual n+1 form

        Notes
        -----
        A simple transpose suffices in the periodic case
        """
        assert self.boundary == 'periodic'
        return [c.transpose for c in self.primal]

    @property
    def hodge(self):
        """Operators that map primal to dual

        Returns
        -------
        array_like, [ndim], DiagonalOperator
            for each level of form, an array broadcast-compatible with the domain

        Notes
        -----
        only pure regular grids now; add support for more complex metrics?
        """
        def h(n):
            scale = (self.scale ** (self.ndim - n * 2))
            def inner(x):
                return x * scale
            return inner
        def hi(n):
            scale = (self.scale ** (self.ndim - n * 2))
            def inner(x):
                return x / scale
            return inner
        return [
            DiagonalOperator(
                h(n), hi(n),
                self.form(n).shape,
            )
            for n in range(self.ndim + 1)
        ]

    def smoothers(self, scale):
        """

        Returns
        -------
        list of smoothers for each primal n-form
        """
        def smooth(n):
            symbols = self.symbols[n]
            sm = smoother(1)
            if not scale:
                sm = sm / sm.max()

            def inner(f):
                """smooth a primal n-form along directions normal to it"""
                smoothed = np.copy(f)
                # loop over all components of the form
                for c, s in enumerate(symbols):
                    # process the form potentially in all directions
                    for i in range(self.ndim):
                        # do not smooth in direction we have taken a derivative in
                        if i in s:
                            continue
                        smoothed[c] = ndimage.convolve1d(smoothed[c], weights=sm, axis=i, mode='wrap')
                return smoothed
            return inner

        return [
            SymmetricOperator(
                smooth(n),
                shape=self.form(n).shape
            )
            for n in range(self.ndim + 1)
        ]

    @property
    def transfers(self):
        """pure direct parent-child transfer operators"""

        def bin(n):
            def inner(fine):
                # apply skipping to some axes and summing to others
                from pycomplex.stencil.util import binning
                coarse = self.coarse.form(n, init='empty')
                for c, s in enumerate(self.symbols[n]):
                    bins = [2 if i in s else 1 for i in range(self.ndim)]
                    skips = [slice(None, None, 1 if i in s else 2) for i in range(self.ndim)]
                    coarse[c] = binning(fine[c][skips], bins)
                return coarse
            return inner

        def tile(n):
            def inner(coarse):
                # insert zeros in some axes and tile others
                fine = self.form(n, init='zeros')
                for c, s in enumerate(self.symbols[n]):
                    repeats = [2 if i in s else 1 for i in range(self.ndim)]
                    view = coarse[c]
                    view = np.ndarray(
                        buffer=view.data,
                        dtype=view.dtype,
                        strides=[q for p in view.strides for q in (p, 0)],
                        shape=[q for p in zip(view.shape, repeats) for q in p],
                    )
                    view = view.reshape([s*r for s, r in zip(coarse[c].shape, repeats)])
                    skips = [slice(None, None, 1 if i in s else 2) for i in range(self.ndim)]
                    fine[c][skips] = view
                return fine
            return inner

        return [
            # NOTE: if  scaling were to be absorbed here if would not longer be a true transpose relationship
            StencilOperator(
                left=tile(n),
                right=bin(n),
                shape=(self.coarse.form(n).shape, self.form(n).shape)
            )
            for n in range(self.ndim + 1)
        ]

    @property
    def coarsen(self):
        """
        Returns
        -------
        array_like, [ndim + 1], Operator

        Notes
        -----
        transpose of refine exacept for scale
        """
        T = self.transfers
        S = self.smoothers(scale=True)
        return [
            ComposedOperator(T[n], S[n])
            for n in range(self.ndim + 1)
        ]

    @property
    def refine(self):
        """
        Returns
        -------
        array_like, [ndim + 1], Operator

        Notes
        -----
        transpose of coarsen exacept for scale
        """
        T = self.transfers
        S = self.smoothers(scale=False)
        return [
            ComposedOperator(T[n], S[n]).transpose
            for n in range(self.ndim + 1)
        ]


class StencilComplex2D(StencilComplex):
    """Make this as light as possible; mostly plotting code?"""

    def __init__(self, *args, **kwargs):
        super(self, StencilComplex).__init__(*args, **kwargs)
        assert self.ndim == 2


class StencilComplex3D(StencilComplex):
    def __init__(self, *args, **kwargs):
        super(self, StencilComplex).__init__(*args, **kwargs)
        assert self.ndim == 3