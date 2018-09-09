from typing import Tuple
from cached_property import cached_property

import numpy as np
from scipy import ndimage

from pycomplex.stencil.operator import StencilOperator, SymmetricOperator, HodgeOperator, DerivativeOperator
from pycomplex.stencil.util import generate, smoother


class StencilComplex(object):
    """Base class for stencil based DEC operations and multigrid hierarchies

    The defining distinction of the stencil based approach is that all topology is implicit
    other than its shape it has no associated memory requirements

    The only implemented variant of this stencil based approach is one with a periodic boundary,
    or toroidal global topology. The appeal of this is that all forms have the exact same spatial extent,
    and there is no need to deal with domain boundaries; allowing us to focus on immersed boundaries instead,
    which are required to maximize the usefulness of such a regular grid based method.

    TODO
    ----
    emit code instead of making callables?
    might eliminate quite some overhead and expose parralelism?
    """

    def __init__(self, shape: Tuple[int], scale=1, boundary: str = 'periodic'):
        self.boundary = boundary
        self.shape = shape
        self.scale = scale
        # these are more like properties
        self.n_dim = len(shape)
        self.symbols, self.terms, self.axes, self.parities = generate(self.n_dim)

    @cached_property
    def is_even(self):
        return all(s % 2 == 0 for s in self.shape)

    @cached_property
    def coarse(self):
        """Construct coarse counterpart"""
        # FIXME: add pointer back to parent?
        assert self.is_even
        return type(self)(
            boundary=self.boundary,
            shape=tuple([s // 2 for s in self.shape]),
            scale=self.scale * 2,
        )

    @cached_property
    def fine(self):
        """Construct fine counterpart"""
        return type(self)(
            boundary=self.boundary,
            shape=tuple([s * 2 for s in self.shape]),
            scale=self.scale / 2,
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
    def metric(self):
        """would be cool if we could supply a variable spacing for each primal edge / axis,
        and translate that into primal and dual n-cube metrics.

        This might quite substantially improve the efficiency of simulating on a fixed-topology grid in practice
        """
        raise NotImplementedError

    @cached_property
    def hodge(self):
        """Operators that map primal to dual

        Returns
        -------
        array_like, [ndim + 1], HodgeOperator
            for each level of form, a pointwise-Operator broadcast-compatible with the domain

        Notes
        -----
        only pure regular grids now, with only a single scalar parameter;
        add support for more complex metrics?
        """
        return [
            HodgeOperator(
                self.scale ** (self.n_dim - n * 2),
                self.n_elements[n],
            )
            for n in range(self.n_dim + 1)
        ]

    def smoothers(self, scale: bool):
        """

        Parameters
        ----------
        scale : bool
            determines normalisation of the smooth kernel

        Returns
        -------
        array_like, [ndim + 1], SymmetricOperator
            smoothing-component of multigrid-transfer operators, for each primal n-form
        """
        def smooth(n):
            symbols = self.symbols[n]
            sm = smoother(1)
            if not scale:
                sm = sm / sm.max()
            else:
                sm = sm / sm.sum()

            def inner(f):
                """smooth a primal n-form along directions normal to it"""
                # FIXME: for n-form we need no smoothing and we can exit early without copy
                smoothed = np.copy(f)
                # loop over all components of the form
                for c, s in enumerate(symbols):
                    # process the form potentially in all directions
                    for i in range(self.n_dim):
                        # do not smooth in direction we have taken a derivative in
                        if i in s:
                            continue
                        smoothed[c] = ndimage.convolve1d(smoothed[c], weights=sm, axis=i, mode='wrap')
                return smoothed
            return inner

        return [
            SymmetricOperator(
                smooth(n),
                shape=self.n_elements[n]
            )
            for n in range(self.n_dim + 1)
        ]

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
    def coarsen(self):
        """Coarsen operators, mapping self to self.coarse

        Returns
        -------
        array_like, [ndim + 1], Operator
            n-th operator maps primal self.form(n) to primal self.coarse.form(n)

        Notes
        -----
        Transpose of refine except for a scale factor
        """
        return [
            T * S
            for T, S in zip(self.transfers, self.smoothers(scale=True))
        ]

    @cached_property
    def refine(self):
        """Refinement operators, mapping self.coarse to self

        Returns
        -------
        array_like, [ndim + 1], Operator
            n-th operator maps primal self.coarse.form(n) to primal self.form(n)

        Notes
        -----
        transpose of coarsen except for a scale factor
        """
        return [
            (T * S).transpose()
            for T, S in zip(self.transfers, self.smoothers(scale=False))
        ]

    def sample_0(self, f0, points, assume_interior=False):
        """Sample a 0-form at the given points, using linear interpolation

        Parameters
        ----------
        f0 : ndarray
            0-form
        points : ndarray, [n_points, n_dim], float
            coordinates in the space defined by the complex
        assume_interior : bool
            if True, it is assumed points are sampled from the interior,
            which allows for some optimisations

        Returns
        -------
        ndarray, [n_points], float
            the 0-form sampled for each row in `points`

        Notes
        -----
        map_coordinates(mode='wrap') is rather broken; or does not do what it should, hence the custom logic

        """
        assert self.boundary == 'periodic'
        points = np.asarray(points) / self.scale

        if not assume_interior:
            f0 = np.pad(f0[0], [(0, 1)] * self.n_dim, mode='wrap')
            points = np.remainder(points, self.shape)

        return ndimage.map_coordinates(f0, points.T, order=1)

    def primal_position(self):
        """Average position of all primal n-elements"""
        raise NotImplementedError

    def dual_position(self):
        raise NotImplementedError

    def averaging_operators(self):
        """implement analogues of topology operators here"""
        raise NotImplementedError

    def explicit(self):
        """convert to Regular complex with explicit topology representation"""
        # FIXME: should we add a topology subobject to this complex class?


class StencilComplex2D(StencilComplex):
    """Make this as light as possible; mostly plotting code?"""

    def __init__(self, *args, **kwargs):
        super(StencilComplex2D, self).__init__(*args, **kwargs)
        assert self.n_dim == 2

    def plot_0(self, f0):
        """Plot a 0-form"""
        import matplotlib.pyplot as plt
        assert f0.shape == self.n_elements[0], "Not a zero-form"
        f0 = f0[0]
        # enforce periodicity
        w, h = f0.shape
        f0 = np.pad(f0, [(0, 1)] * self.n_dim, mode='wrap')
        plt.figure()
        plt.imshow(f0, interpolation='bilinear', extent=[0, w, 0, h])
        plt.colorbar()
        # plt.show()


class StencilComplex3D(StencilComplex):
    """Adding a wavefront-based cuda/opencl kernel here would pay off the most"""
    def __init__(self, *args, **kwargs):
        super(StencilComplex3D, self).__init__(*args, **kwargs)
        assert self.n_dim == 3

    def segment(self, f0, level):
        """Segment a level-set contour on a 0-form associated with this complex

        Parameters
        ----------
        f0 : 0-form
        level : float

        Returns
        -------
        ComplexTriangularEuclidian3
        """
        assert f0.shape == self.n_elements[0], "Not a zero-form"

        from skimage.measure import marching_cubes_lewiner
        vertices, triangles, normals, values = marching_cubes_lewiner(f0, level=level, spacing=(self.scale, ) * 3)
        from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
        return ComplexTriangularEuclidian3(vertices=vertices, triangles=triangles)
