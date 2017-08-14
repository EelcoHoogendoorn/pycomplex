from abc import abstractmethod
from cached_property import cached_property

import numpy as np
import scipy
import numpy_indexed as npi

from pycomplex.topology import sign_dtype
from pycomplex.sparse import normalize_l1
from pycomplex.math import linalg


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

    def select_subset(self, n_chain):
        """

        Parameters
        ----------
        n_chain : ndarray, [n_elements[-1], sign_type
            chain indicating which elements to select

        Returns
        -------
        type(self)
            The indicated subset
        """
        n_chain = np.asarray(n_chain)
        if not len(n_chain) == self.topology.n_elements[-1]:
            raise ValueError
        subset_topology = self.topology.select_subset(n_chain)
        return self.copy(topology=subset_topology, vertices=self.vertices[subset_topology.parent_idx[0]])

    @cached_property
    def boundary(self):
        boundary = self.topology.boundary
        if boundary is None:
            return None
        else:
            # FIXME: type self should be boundary_type
            return type(self)(vertices=self.vertices[boundary.parent_idx[0]], topology=boundary)

    def fix_orientation(self):
        return self.copy(topology=self.topology.fix_orientation())

    def copy(self, vertices=None, topology=None, **kwargs):
        c = type(self)(
            vertices=self.vertices if vertices is None else vertices,
            topology=self.topology if topology is None else topology,
            **kwargs
        )
        c.parent = self
        return c

    @cached_property
    def dual_position(self):
        """Positions of all dual elements; primal elements with boundary elements appended where required

        Returns
        -------
        list of element positions, length n_dim
            where th n-th element are the locations of the dual n-elements
            this consists of interior and boundary elements, in that order

        """
        # interior dual elements are located at their corresponding primal
        pp = self.primal_position

        # location of dual boundary element is location of corresponding primal boundary element
        boundary = self.topology.boundary
        if boundary is None:
            return pp[::-1]

        dp = []
        for i, (e) in enumerate(self.topology.elements):
            if i == 0:
                c = [pp[i]]
            else:
                idx = boundary.parent_idx[i-1]
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
        for corners, cr, centroids in reversed(list(zip(self.topology.corners, C, self.primal_position))):
            if cr is not None:
                n_pts = corners.shape[1]
                corners = corners[cr].flatten()
                centroids = np.repeat(centroids[cr], n_pts, axis=0)
                vertex_i, vertex_p = npi.group_by(corners).mean(centroids)
                vertices[vertex_i] = vertex_p

        return self.copy(vertices=vertices)

    def smooth_operator(self, creases=None):
        """Loop / catmul-clark / MLCA type smoothing operator

        Parameters
        ----------
        creases : dict of (int, ndarray), optional
            dict of n-chains, where nonzero elements denote crease elements

        Returns
        -------
        sparse array, [n_vertices, n_vertices]
            applies MLCA type smoothing to the primal vertices of a complex

        Notes
        -----
        This is now a method on complex; but really does not rely on the vertices in any way;
        perhaps better off in topology class

        """

        # creasing default behavior; all n-elements are 'creases', and none of the other topological elements are
        C = [np.zeros(n, dtype=sign_dtype) for n in self.topology.n_elements]
        C[-1][:] = 1
        # set crease overrides from the given dict
        if creases:
            for n, c in creases.items():
                C[n][:] = c != 0

        # need to decide for each fine vertex what n-elements it sources from; zero out others
        S = np.zeros((self.topology.n_dim + 1, self.topology.n_elements[0]), dtype=sign_dtype)
        for s, c, corners in zip(S, C, self.topology.corners):
            influence = np.unique(corners[c != 0])
            s[influence] = 1
        # let lower n creases override higher ones; this could be tweaked for partial creases
        for i, a in enumerate(S[:-1]):
            for b in S[i + 1:]:
                b[a != 0] = 0

        averaging = self.topology.averaging_operators_0
        smoothers = [scipy.sparse.diags(s) * a.T * scipy.sparse.diags(c) * a for a, s, c in zip(averaging, S, C)]
        return normalize_l1(sum(smoothers), axis=1)

    @cached_property
    def primal_metric(self):
        return self.metric[0]
    @cached_property
    def dual_metric(self):
        return self.metric[1]

    @cached_property
    def hodge_PD(self):
        """Hodges that map dual to primal forms. Indexed by primal element."""
        P, D = self.metric
        return [p / d for p, d in zip(P, D[::-1])]
    @cached_property
    def hodge_DP(self):
        """Hodges that map primal to dual forms. Indexed by primal element."""
        P, D = self.metric
        return [d / p for p, d in zip(P, D[::-1])]

    @abstractmethod
    def pick_primal(self, points):
        raise NotImplementedError
    @abstractmethod
    def pick_dual(self, points):
        raise NotImplementedError

    @cached_property
    def positive_dual_metric(self):
        """Returns true if all dual metrics are positive"""
        return all([np.all(m > 0) for m in self.dual_metric])


class BaseComplexEuclidian(BaseComplex):

    @cached_property
    def primal_barycentric(self):
        """barycentric positions of all primal elements

        Returns
        -------
        pp : list of primal element positions, length n_dim
        """
        from pycomplex.geometry import euclidian
        return [euclidian.circumcenter_barycentric(
                    self.vertices[c],
                    self.weights[c] if self.weights is not None else None)
                for c in self.topology.corners]

    @cached_property
    def primal_position(self):
        """positions of all primal elements

        Returns
        -------
        pp : list of primal element positions, length n_dim
        """
        return [np.einsum('...cn,...c->...n', self.vertices[c], b)
                for c, b in zip(self.topology.corners, self.primal_barycentric)]


class BaseComplexCubical(BaseComplex):

    @cached_property
    def primal_position(self):
        """positions of all primal elements

        Returns
        -------
        list of primal element positions, length n_dim
        """
        return [self.vertices[c].mean(axis=1) for c in self.topology.corners]


class BaseComplexSpherical(BaseComplex):

    @cached_property
    def primal_barycentric(self):
        """barycentric positions of all primal elements

        Returns
        -------
        pp : list of primal element positions, length n_dim
        """
        from pycomplex.geometry import euclidian
        return [euclidian.circumcenter_barycentric(
                    self.vertices[c],
                    self.weights[c] if self.weights is not None else None)
                for c in self.topology.corners]

    @cached_property
    def primal_position(self):
        """positions of all primal elements

        Returns
        -------
        pp : list of primal element positions, length n_dim
        """
        return [linalg.normalized(np.einsum('...cn,...c->...n', self.vertices[c], b))
                for c, b in zip(self.topology.corners, self.primal_barycentric)]
