from abc import abstractmethod

import numpy as np
import scipy
from cached_property import cached_property
import numpy_indexed as npi


class BaseComplex(object):
    """Complex, regardless of embedding space or element type"""

    @cached_property
    def n_dim(self):
        """Dimensionality of the embedding space"""
        return self.vertices.shape[1]

    @cached_property
    def box(self):
        """Bounding box of the complex

        Returns
        -------
        ndarray, [2, n_dim], float
            aabb
        """
        return np.array([
            self.vertices.min(axis=0),
            self.vertices.max(axis=0),
        ])

    @abstractmethod
    def subdivide(self):
        """Return a complex of the same type, with one level of subdivision applied"""
        raise NotImplementedError

    def dual_position(self):
        """Positions of all dual elements; primal elements with boundary elements appended where required

        Returns
        -------
        list of element positions, length n_dim
            where th n-th element are the locations of the dual n-elements
            this consists of interior and boundary elements, in that order

        """
        # interior dual elements are located at their corresponding primal
        pp = self.primal_position()
        dp = []

        # location of dual boundary element is location of corresponding primal boundary element
        boundary = self.topology.boundary()
        for i, (e) in enumerate(self.topology.elements):
            if i == 0 or boundary is None:
                c = [pp[i]]
            else:
                idx = npi.indices(self.topology.elements[i-1], boundary.elements[i-1])
                b = pp[i-1][idx]
                c = [pp[i], b]
            dp.append(np.concatenate(c, axis=0))

        return dp[::-1]

    def smooth(self, creases=None):
        """Loop / catmul-clark / MLCA type smoothing

        Parameters
        ----------
        creases : dict of (int, ndarray), optional
            dict of n-chains, where nonzero elements denote crease elements

        Returns
        -------
        type(self)
            smoothed copy of self

        Notes
        -----
        MLCA implies no weights at all; Loop disagrees.
        Some form of weighting does intuitively make sense to me
        """
        # creasing default behavior; all n-elements are 'creases', and none of the other topological elements are
        C = [None] * self.topology.n_dim + [Ellipsis]
        if creases:
            for n, c in creases.items():
                C[n] = np.flatnonzero(c) if isinstance(c, np.ndarray) else c

        # accumulate new vertex positions
        vertices = self.vertices.copy()
        # start with n-elements, working backwards to vertices
        for corners, cr, centroids in reversed(list(zip(self.topology.corners, C, self.primal_position()))):
            if cr is not None:
                n_pts = corners.shape[1]
                corners = corners[cr].flatten()
                centroids = np.repeat(centroids[cr], n_pts, axis=0)
                vertex_i, vertex_p = npi.group_by(corners).mean(centroids)
                vertices[vertex_i] = vertex_p

        return type(self)(vertices=vertices, topology=self.topology)


class BaseComplexEuclidian(BaseComplex):

    def primal_position(self):
        """positions of all primal elements

        Returns
        -------
        pp : list of primal element positions, length n_dim
        """
        from pycomplex.geometry.euclidian import circumcenter
        return [circumcenter(self.vertices[c]) for c in self.topology.corners]


class BaseComplexCubical(BaseComplex):
    def primal_position(self):
        """positions of all primal elements

        Returns
        -------
        list of primal element positions, length n_dim
        """
        return [self.vertices[c].mean(axis=1) for c in self.topology.corners]


class BaseComplexSpherical(BaseComplex):
    def primal_position(self):
        """positions of all primal elements, determined as spherical circumcenters

        Returns
        -------
        list of primal element positions, length n_dim
        """
        from pycomplex.geometry.spherical import circumcenter
        return [circumcenter(self.vertices[c]) for c in self.topology.corners]


