from typing import Tuple

import numpy as np
from cached_property import cached_property
from scipy import ndimage

from pycomplex.stencil.operator import SymmetricOperator, HodgeOperator
from pycomplex.stencil.util import smoother
from pycomplex.stencil.topology import StencilTopology


class StencilComplex(object):
    """Base class for stencil based DEC operations and multigrid hierarchies

    The defining distinction of the stencil based approach is that all topology is implicit
    other than its shape it has no associated memory requirements

    The only implemented variant of this stencil based approach is one with a periodic boundary,
    or toroidal global topology. The appeal of this is that all forms have the exact same spatial extent,
    and there is no need to deal with domain boundaries; allowing us to focus on immersed boundaries instead,
    which are required to maximize the usefulness of such a regular grid based method.
    """

    def __init__(self, topology, scale):
        self.topology = topology
        self.scale = scale

    @classmethod
    def from_shape(cls, shape: Tuple[int], scale=1):
        return cls(topology=StencilTopology(shape=shape), scale=scale)

    @property
    def n_dim(self):
        return self.topology.n_dim

    @property
    def shape(self):
        return self.topology.shape

    @property
    def n_elements(self):
        return self.topology.n_elements

    @cached_property
    def coarse(self):
        """Construct coarse counterpart"""
        # FIXME: add pointer back to parent?
        return type(self)(
            topology=self.topology.coarse,
            scale=self.scale * 2,
        )

    @cached_property
    def fine(self):
        """Construct fine counterpart"""
        return type(self)(
            topology=self.topology.fine,
            scale=self.scale / 2,
        )

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
        should probably be a seperate subclass
        """
        return [
            HodgeOperator(
                diagonal=self.scale ** (self.n_dim - n * 2),
                shape=self.n_elements[n],
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
            symbols = self.topology.symbols[n]
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
            for T, S in zip(self.topology.transfers, self.smoothers(scale=True))
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
            for T, S in zip(self.topology.transfers, self.smoothers(scale=False))
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
        assert self.topology.boundary == 'periodic'
        points = np.asarray(points) / self.scale

        if not assume_interior:
            f0 = np.pad(f0[0], [(0, 1)] * self.n_dim, mode='wrap')
            points = np.remainder(points, self.shape)

        return ndimage.map_coordinates(f0, points.T, order=1)

    @cached_property
    def primal_position(self):
        """Average position of all primal n-elements"""
        p = np.indices(self.shape) * self.scale
        p = np.moveaxis(p, 0, -1)
        return [
            p[None, ...]
        ]

    @cached_property
    def dual_position(self):
        raise NotImplementedError

    def explicit(self):
        """convert to Regular complex with explicit topology representation"""
        raise NotImplementedError
        from pycomplex.complex.cubical import ComplexCubical
        return ComplexCubical(
            vertices=self.primal_position[0],
            topology=self.topology.explicit()
        )


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
        plt.imshow(f0.T, interpolation='bilinear', extent=[0, w, 0, h])
        plt.colorbar()

    def plot_1(self, f1):
        """Visualize a 1-form as a streamplot"""
        import matplotlib.pyplot as plt
        assert f1.shape == self.n_elements[1], "Not a one-form"
        # FIXME: need to implement mapping from 1-form to vertex-based vector-field; method on complex
        u, v = f1.copy()
        u = u + np.roll(u, shift=1, axis=0)
        v = v + np.roll(v, shift=1, axis=1)
        x = np.arange(self.shape[0]) * self.scale
        y = np.arange(self.shape[1]) * self.scale
        plt.streamplot(x, y, u.T, v.T)

    def plot_2(self, f2):
        """Plot a 2-form"""
        import matplotlib.pyplot as plt
        assert f2.shape == self.n_elements[2], "Not a two-form"
        f2 = f2[0]
        plt.figure()
        w, h = f2.shape
        plt.imshow(f2.T, interpolation='nearest', extent=[0, w, 0, h])
        plt.colorbar()


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
