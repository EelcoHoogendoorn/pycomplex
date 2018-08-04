from typing import Tuple

import numpy as np
from scipy import ndimage
from pycomplex.stencil.operator import StencilOperator

from pycomplex.stencil.util import generate, smoother


class StencilComplex(object):

    """Abstract factory class for stencil based dec operations and mg hierarchies

    The defining distinction of the stencil based approach is that all topology is implicit
    other than its shape it has no associated memory requirements

    make component-first or component-last configurable?
    fix a convention for the meaning of all form components?
    anticommutative algebra works like this: xy = -yx, xx = 0
    at each derivative step, take derivative of each component wrt each dir.
    drop zero terms, aggregate identical symbols

    """

    def __init__(self, shape: Tuple[int], boundary: str = 'periodic'):
        self.boundary = boundary
        self.shape = shape
        self.ndim = len(shape)
        self.symbols, self.terms, self.axes, self.parities = generate(self.ndim)

    def form(self, n, dtype=np.float32, init='zeros'):
        """Allocate an n-form"""
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
        def conv(*args, **kwargs):
            return ndimage.convolve1d(*args, **kwargs, mode='wrap')
        def corr(*args, **kwargs):
            return ndimage.correlate1d(*args, **kwargs, mode='wrap')

        ops = []
        for n, (symbols, terms, axes, parities) in enumerate(zip(self.symbols, self.terms, self.axes, self.parities)):
            def wrapper(n, symbols, terms, axes, parities, func):
                def foo(f):
                    d = self.form(n + 1, init='empty')
                    # write once; update afterwards
                    inits = [False] * len(d)
                    # loop over all components of the new derived form
                    for c, (T, A, P) in enumerate(zip(terms, axes, parities)):
                        # loop over all terms that the new symbol is composed of
                        for t, a, p in zip(T, A, P):
                            i = symbols.index(t)
                            weights = [-1, +1] if p else [+1, -1]
                            if inits[c]:
                                d[c] += func(f[i], axis=a, weights=weights)
                            else:
                                func(f[i], axis=a, output=d[c], weights=weights)
                                inits[c] = True
                    return d
                return foo

            op = StencilOperator(
                wrapper(n, symbols, terms, axes, parities, conv),
                wrapper(n, symbols, terms, axes, parities, corr),
                shape=(self.form(n+1).shape, self.form(n).shape)
            )
            ops.append(op)
        return ops

    @property
    def dual(self):
        """Dual derivative operators

        Returns
        -------
        array_like, [ndim-1], operator
            list of operators mapping dual n-form to dual n+1 form
        """
        return [c.transpose for c in self.primal]

    @property
    def hodge(self):
        """

        Returns
        -------
        array_like, [ndim], n-form
            for each level of form, an array broadcast-compatible with the domain
        """

    @property
    def smoother(self):
        return smoother(self.ndim)

    @property
    def transfer(self):
        # FIXME: which of two versions? smoothing in all directions, or only those containing zeros?
        raise NotImplementedError

    @property
    def coarsen(self):
        """
        Returns
        -------
        array_like, [ndim], n-form
        """
        return self.transfer

    @property
    def refine(self):
        # assume galerkin transfer operators as a default
        return [c.transpose for c in self.coarsen]


class StencilComplex2D(StencilComplex):
    """Make this as light as possible; mostly plotting code?"""

    def __init__(self, *args, **kwargs):
        super(self, StencilComplex).__init__(*args, **kwargs)
        assert self.ndim == 2

    def coarsen(self):
        """

        Returns
        -------

        Notes
        -----
        conv and corr are the same here, considering symmetric smoothing stencil
        """
        def c0(p0):
            s = ndimage.convolve(p0, weights=self.smoother, mode='wrap')
            return s[:, ::2, ::2]
        def c1(p1):
            s = np.copy(p1)
            ndimage.convolve(p1[0], output=s[0], weights=self.smoother, mode='wrap')
            ndimage.convolve(p1[1], output=s[1], weights=self.smoother, mode='wrap')
            x = s[0, :, ::2]
            y = s[1, ::2, :]
            return np.array([
                binning(x, [2, 1]),
                binning(y, [1, 2]),
            ])
        def c2(p2):
            s = ndimage.convolve(p2[0], weights=self.smoother, mode='wrap')
            return binning(s, [2, 2])

        return [c0, c1, c2]

    def refine(self):
        """

        Returns
        -------

        Notes
        -----
        These are effectively transposes of the coarsen operator
        """
        def r0(p0):
            # tile with zeros inbetween; then convolve
            return ndimage.convolve(s, weights=self.smoother, mode='wrap')

        def r1(p1):
            pass

        def r2(p2):
            s = np.tile(p2, (2, 2))
            return ndimage.convolve(s, weights=self.smoother, mode='wrap')

        return [r0, r1, r2]



class StencilComplex3D(StencilComplex):
    def __init__(self, *args, **kwargs):
        super(self, StencilComplex).__init__(*args, **kwargs)
        assert self.ndim == 3
