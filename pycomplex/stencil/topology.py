from typing import Tuple

import numpy as np
from cached_property import cached_property
from scipy import ndimage

from pycomplex.stencil.operator import DerivativeOperator, StencilOperator
from pycomplex.stencil.util import generate


class StencilTopology(object):
    def __init__(self, shape: Tuple[int], boundary: str = 'periodic'):
        self.boundary = boundary
        self.shape = shape
        # these are more like properties
        self.n_dim = len(shape)
        self.symbols, self.terms, self.axes, self.parities = generate(self.n_dim)

    @cached_property
    def is_even(self):
        return all(s % 2 == 0 for s in self.shape)

    @cached_property
    def coarse(self):
        """Construct coarse counterpart"""
        # FIXME: add pointer back to fine parent?
        assert self.is_even
        return type(self)(
            boundary=self.boundary,
            shape=tuple([s // 2 for s in self.shape]),
        )

    @cached_property
    def fine(self):
        """Construct fine counterpart"""
        return type(self)(
            boundary=self.boundary,
            shape=tuple([s * 2 for s in self.shape]),
        )

    @cached_property
    def n_elements(self):
        """Number of elements in each direction
        That is, the shape of each n-form

        Returns
        -------
        List, [n_dim + 1], Tuple[int]
        """
        return [(len(s), ) + self.shape for s in self.symbols]

    def form(self, n, dtype=np.float32, init='zeros'):
        """Allocate an n-form

        Returns
        -------
        list of len n_components of ndarrays of shape self.shape
        """
        assert self.boundary == 'periodic'
        # with periodic boundaries all forms have the same shape
        shape = self.n_elements[n]
        # in more general case, different forms and different components of forms may have different shape
        if init == 'zeros':
            return np.zeros(shape, dtype=dtype)
        if init == 'empty':
            return np.empty(shape, dtype=dtype)
        if init == 'ones':
            return np.ones(shape, dtype=dtype)
        raise Exception

    @cached_property
    def primal(self):
        """Primal derivative operators

        Returns
        -------
        List, [ndim], operator
            list of stencil operators mapping primal n-form to primal n+1 form
        """
        assert self.boundary == 'periodic'
        def conv(*args, **kwargs):
            return ndimage.convolve1d(*args, **kwargs, mode='wrap')
        def corr(*args, **kwargs):
            return ndimage.correlate1d(*args, **kwargs, mode='wrap')

        def primal(n, terms, axes, parities):
            def inner(f):
                d = self.form(n + 1, init='empty')
                # write once; update afterwards
                initialized = [False] * len(d)
                # loop over all components of the new derived form
                for c, (T, A, P) in enumerate(zip(terms, axes, parities)):
                    # loop over all terms that the new symbol is composed of
                    for t, a, p in zip(T, A, P):
                        weights = [-1, +1] if p else [+1, -1]
                        if initialized[c]:
                            d[c] += conv(f[t], axis=a, weights=weights)
                        else:
                            conv(f[t], axis=a, output=d[c], weights=weights)
                            initialized[c] = True
                return d
            return inner
        def dual(n, terms, axes, parities):
            def inner(f):
                d = self.form(n, init='empty')
                # write once; update afterwards
                initialized = [False] * len(d)
                # loop over all components of the new derived form
                for c, (T, A, P) in enumerate(zip(terms, axes, parities)):
                    # loop over all terms that the new symbol is composed of
                    for t, a, p in zip(T, A, P):
                        weights = [-1, +1] if p else [+1, -1]
                        if initialized[t]:
                            d[t] += corr(f[c], axis=a, weights=weights)
                        else:
                            corr(f[c], axis=a, output=d[t], weights=weights)
                            initialized[t] = True
                return d
            return inner

        return [
            DerivativeOperator(
                # in boundary-free case, left operator equals dual derivative
                # not just a conv/corr difference; also need to transpose summing logic!
                left=dual(n, terms, axes, parities),
                right=primal(n, terms, axes, parities),
                shape=(self.n_elements[n+1], self.n_elements[n])
            )
            for n, (terms, axes, parities)
            in enumerate(zip(self.terms, self.axes, self.parities))
        ]

    @cached_property
    def dual(self):
        """Dual derivative operators

        Returns
        -------
        array_like, [ndim], operator
            list of operators mapping dual n-form to dual n+1 form

        Notes
        -----
        A simple transpose suffices in the periodic case
        """
        assert self.boundary == 'periodic'
        return [c.transpose() for c in self.primal]

    @cached_property
    def transfers(self):
        """Pure direct parent-child transfer operators
        That is, this transfer operators only relates n-cubes that share a parent-child relationship,
        or c-cubes on different levels which in a geometrical sense share overlap

        Returns
        -------
        array_like, [ndim + 1], StencilOperator
            transfer-component of multigrid-transfer operators, for each primal n-form

        """

        def bin(n):
            def inner(fine):
                # apply skipping to some axes and summing to others
                from pycomplex.stencil.util import binning
                coarse = self.coarse.form(n, init='empty')
                for c, s in enumerate(self.symbols[n]):
                    bins = [2 if i in s else 1 for i in range(self.n_dim)]
                    skips = [slice(None, None, 1 if i in s else 2) for i in range(self.n_dim)]
                    coarse[c] = binning(fine[c][skips], bins)
                return coarse
            return inner

        def tile(n):
            def inner(coarse):
                # insert zeros in some axes and tile others
                fine = self.form(n, init='zeros')
                for c, s in enumerate(self.symbols[n]):
                    repeats = [2 if i in s else 1 for i in range(self.n_dim)]
                    view = coarse[c]
                    view = np.ndarray(
                        buffer=view.data,
                        dtype=view.dtype,
                        strides=[q for p in view.strides for q in (p, 0)],
                        shape=[q for p in zip(view.shape, repeats) for q in p],
                    )
                    view = view.reshape([s*r for s, r in zip(coarse[c].shape, repeats)])
                    skips = [slice(None, None, 1 if i in s else 2) for i in range(self.n_dim)]
                    fine[c][skips] = view
                return fine
            return inner

        return [
            # NOTE: this is not a true transpose relationship when the call to binning is using a mean reduction
            StencilOperator(
                left=tile(n),
                right=bin(n),
                shape=(self.coarse.n_elements[n], self.n_elements[n])
            )
            for n in range(self.n_dim + 1)
        ]

    @cached_property
    def averaging_operators_0(self):
        """Average 0-forms onto n-forms

        Returns
        -------
        array_like, [n_dim + 1], StencilOperator
            n-th operator averages zero-forms to n-forms
        """
        def average(n):
            def stencil(axes):
                shape = np.ones(self.n_dim, np.int)
                shape[list(axes)] = 2
                return np.ones(shape) / np.prod(shape)
            # one stencil for each output component
            stencils = [stencil(a) for a in self.symbols[n]]

            def inner(f0):
                output = self.form(n, dtype=f0.dtype)
                for i, stencil in enumerate(stencils):
                    ndimage.convolve(f0[0], stencil, output[i], mode='wrap')
                return output
            return inner

        return [
            StencilOperator(
                right=average(n),
                left=None,  # FIXME: implement transpose?
                shape=(self.n_elements[n], self.n_elements[0])
            )
            for n in range(self.n_dim + 1)
        ]

    @cached_property
    def averaging_operators_N(self):
        """Not much different to implement from 0-averaging; just take inverse of symbol-set

        however; may even be possible to implement n-to-n averaging operators;
        just consider diff of input and output order symbols
        """
        raise NotImplementedError

    def explicit(self):
        """convert to explicit topology representation"""
        raise NotImplementedError
        from pycomplex.topology.cubical import TopologyCubical
        return TopologyCubical()