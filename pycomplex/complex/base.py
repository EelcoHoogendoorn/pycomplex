
from abc import abstractmethod
from cached_property import cached_property
import functools
import operator

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
    def subdivide_cubical(self):
        """Returns a cubical complex"""
        raise NotImplementedError

    @abstractmethod
    def subdivide_fundamental(self):
        """Returns a simplicial complex"""
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
        # FIXME: this also copies the weights; also need their subset taken
        # not hard to fix, but radius in spherical case should be copied though
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

    def copy(self, **kwargs):
        import inspect
        args = inspect.signature(type(self)).parameters.keys()
        nkwargs = {}
        for a in args:
            if hasattr(self, a):
                nkwargs[a] = getattr(self, a)

        nkwargs.update(kwargs)
        c = type(self)(**nkwargs)
        c.parent = self
        return c

    def transform(self, transform):
        return self.copy(vertices=self.vertices.dot(transform))

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

        bp = [None] + [p[idx] for idx, p in zip(boundary.parent_idx, pp)]
        dp = [np.concatenate([p] + ([] if b is None else [b]), axis=0) for p, b in zip(pp, bp)]
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

        A = self.topology.averaging_operators_0
        smoothers = [scipy.sparse.diags(s) * a.T * scipy.sparse.diags(c) * a for a, s, c in zip(A, S, C)]
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

    def dual_edge_excess(self, signed=True):
        """Compute the 'dual edge excess'

        Used in both weight optimization as well as primal simplex picking

        Returns
        -------
        ndarray, [n_N-elements, n_corners], float
            how deep the circumcenter sits in the simplex, as viewed from each corner,
            or relative to the face opposite of that corner
            in units of distance-squared

        Notes
        -----
        euclidian metric is used here; need to compensate for that in spherical applications
        """
        # raise Exception('these PP should also be euclidian in the spherical case!')
        PP = self.primal_position
        B = self.topology._boundary[-1]
        delta = PP[-1][:, None, :] - PP[-2][B]
        d = linalg.dot(delta, delta)  #* sign
        if signed:
            d = d * np.sign(self.primal_barycentric[-1])
        return d

    def optimize_weights(self):
        """Optimize the weights of a simplicial complex, to improve positivity of the dual metric,
        and the condition number of equations based on their hodges.

        This method optimizes the

        Returns
        -------
        type(self)
            copy of self with optimized weights

        Notes
        -----
        This is a simplification of the method in the references below.

        There is a strong link between this and the dual optimization done to derive weights for primal simplex picking

        The logic here emphasises the balancing of dual edge excess at each edge.
        Each dual edge length is directly related to the primal vertex weights;
        and we obtain a set of equations in terms of primal vertex weights,
        by forming the normal equations.

        We solve these over-determined equations for the vertex weights using a few jacobi iterations,
        since global redistribution of weights is not desirable.

        References
        ----------
        http://www.geometry.caltech.edu/pubs/MMdGD11.pdf
        http://www.geometry.caltech.edu/pubs/dGMMD14.pdf
        """
        assert self.weights is None # not sure this is required
        # FIXME: behavior at the boundary is not ideal yet

        T = self.topology.matrices[0]
        P1P0 = T.T
        DNDn = T
        laplacian = DNDn * P1P0

        field = self.remap_boundary_0(self.dual_edge_excess())

        # We do only a few iterations of jacobi iterations to solve our equations,
        # since in practice local redistributions of dual edge length are the only ones of interest
        diag = self.topology.degree[0]
        field = field / diag
        rhs = field - field.mean()
        weights = np.zeros_like(field)
        for i in range(1):
            weights = (rhs - laplacian * weights + diag * weights) / diag
        weights = weights * diag
        return self.copy(weights=weights)

    def optimize_weights_metric(self):
        """Optimize the weights of a simplicial complex, to improve positivity of the dual metric,
        and the condition number of equations based on their hodges.

        This optimizes the weights such as to minimize the distance of the circumcenters to the barycenters.

        Indeed this objective is accomplished very well by the method, but it does feel a bit overly
        'aggressive' for many applications

        Returns
        -------
        type(self)
            copy of self with optimized weights

        References
        ----------
        http://www.geometry.caltech.edu/pubs/MMdGD11.pdf
        http://www.geometry.caltech.edu/pubs/dGMMD14.pdf
        """
        assert self.weights is None # not sure this is required

        # FIXME: spherical and simplicial should use the same code path; rethink class hierarchy
        if isinstance(self, BaseComplexEuclidian):
            euclidian = self
        else:
            euclidian = self.as_euclidian()

        T = self.topology.matrices[0]
        DnP1 = euclidian.hodge_DP[1]
        P1P0 = T.T
        DNDn = T
        laplacian = DNDn * scipy.sparse.diags(DnP1) * P1P0

        corners = self.vertices[self.topology.corners[-1]]
        barycenter = corners.mean(axis=1)
        PP = self.primal_position
        circumcenter = PP[-1]
        diff = circumcenter - barycenter    # this is what we seek to minimize
        B = self.topology._boundary[-1]

        # FIXME: using this as gradient directions is brittle; will fail around 0
        # delta = linalg.normalized(PP[-1][:, None, :] - PP[-2][B])

        boundary_normals = linalg.pinv(corners - barycenter[:, None, :])
        boundary_normals = np.swapaxes(boundary_normals, -2, -1)
        boundary_normals = linalg.normalized(boundary_normals)
        face_metric = euclidian.primal_metric[-2]
        field = linalg.dot(diff[:, None, :], boundary_normals) * face_metric[B] #* np.sign(self.primal_barycentric[-1])

        field = self.remap_boundary_0(field)

        rhs = field - field.mean()
        # We do only a few iterations of jacobi iterations to solve our equations,
        # since in practice local redistributions of dual edge length are the only ones of interest
        diag = laplacian.diagonal()
        weights = np.zeros_like(field)
        for i in range(1):
            weights = (rhs - laplacian * weights + diag * weights) / diag
        return self.copy(weights=weights)

    # FIXME: move to simplicial base class
    def optimize_weights_fundamental(self):
        """Optimize the weights of a complex derived from fundamental domain subdivision,
        such that the resulting primal simplices become strictly well-centered,
        in case it is derived form a simplex itself.
        Not true if it comes from a cubical complex embedded in space

        This can be accomplished simply by bumping the weights of fundamental vertices
        corresponding to parent vertices a little

        Returns
        -------
        copy of self, with optimized weights such that the dual vertex is well-centered
        """
        assert self.weights is None

        parent = self.topology.parent
        weights = self.topology.form(0)

        # calculate average length of edge connecting to vertex-vertex
        edge_length = self.unsigned_volume(self.vertices[self.topology.elements[1]])
        # edge_length = self.primal_metric[1]

        A = self.topology.averaging_operators_0[1].T
        scale = (A * edge_length ** 2) / (A * np.ones_like(edge_length))

        # best to only push away along primal edges, since dual edge lengths are much more variable
        weights[:parent.n_elements[0]] = scale[:parent.n_elements[0]] / 4
        # weights[-parent.n_elements[-1]:] = scale
        # weights[parent.n_elements[0]:-parent.n_elements[-1]] = -scale / 2

        return self.copy(weights=weights)

    def remap_boundary_N(self, field, oriented=True):
        """Given a quantity computed on each n-simplex-boundary, combine the contributions of each incident n-simplex

        Parameters
        ----------
        field : ndarray, [n_N-simplices, n_corners], float
            a quantity defined on each boundary of all simplices

        Returns
        -------
        field : ndarray, [n_n-simplices], float
            quantity defined on all boundaries
        """
        INn = self.topology._boundary[-1]
        if oriented:
            ONn = self.topology._orientation[-1]
            field = field * ONn
        _, field = npi.group_by(INn.flatten()).sum((field).flatten())
        return field

    def remap_boundary_0(self, field):
        """Given a quantity computed on each n-simplex-boundary,
        Sum around the vertices surrounded by this boundary

        Parameters
        ----------
        field : ndarray, [n_N-simplices, n_corners], float
            a quantity defined on each boundary of all N-simplices

        Returns
        -------
        field : ndarray, [n_0-simplices], float

        """
        IN0 = self.topology.elements[-1]
        _, field = npi.group_by(IN0.flatten()).sum((field).flatten())
        return field

    def weights_to_offsets(self, weights):
        """Convert power dual weights to offsets in an orthogonal coordinate

        Parameters
        ----------
        weights : ndarray, [n_vertices], float
            power dual weight in units of distance squared

        Returns
        -------
        offset : ndarray, [n_vertices], float
            distance away from the space of self.vertices in an orthogonal coordinate,
            required to realize the implied shift of dual edges in the offset=0 plane
            of the voronoi diagram
        """
        return np.sqrt(-weights + weights.max())

    @cached_property
    def positive_dual_metric(self):
        """Returns true if all dual metrics are positive"""
        return all([np.all(m > 0) for m in self.dual_metric])

    @cached_property
    def is_well_centered(self):
        """Test that all circumcenters are inside each simplex"""
        return all([np.all(b > 0) for b in self.primal_barycentric])

    @cached_property
    def is_pairwise_delaunay(self):
        """Test that adjacent circumcenters do not cross eachother, or that dual 1-metric is positive"""
        return np.all(self.remap_boundary_N(self.dual_edge_excess(), oriented=False) > 0)

    @cached_property
    def weighted_average_operators(self):
        """Weight averaging over the duals by their barycentric mean coordinates

        Divide by distance to dual vertex to make all other zero when approaching vertex
        Divide by distance to dual edge to make all other zero when approaching dual edge
        Divide by distance to dual face to make all other zero when approaching dual face
        and so on.

        General idea; divide perimeter of a dual element
        by the distance to all corners corresponding to lower order duals

        References
        ----------
        http://vcg.isti.cnr.it/Publications/2004/HF04/coordinates_aicm04.pdf
        https://www.researchgate.net/publication/2856409_Generalized_Barycentric_Coordinates_on_Irregular_Polygons

        Notes
        -----
        Requires a well-centered mesh.
        Getting this working on non-well centered meshes requires handling of sign in metric caluculations,
        but also replacing the division by the distance to each corner of a domain,
        with a product to all other distances in the cell, to avoid divisions by zero.
        This strikes me as a bit daunting still

        While I think its really neat how well this can be implemented for any dimension,
        the utility thereof isnt that obvious, considering the seeming difficulty of finding
        well-centered simplicial meshes in dimensions greater than 2.
        """

        topology = self.topology
        assert topology.is_oriented
        assert self.is_well_centered

        PP = self.primal_position
        domains = self.topology.fundamental_domains()

        domains = domains.reshape(-1, domains.shape[-1])
        corners = np.concatenate([p[d][:, None, :] for p, d in zip(PP, domains.T)], axis=1)

        unsigned = self.unsigned_volume

        # construct required distances; all edges lengths
        def edge_length(a, b):
            return unsigned(corners[:, [a, b]])

        def edge_length_prod(n):
            return functools.reduce(operator.mul, [edge_length(n, m + 1) for m in range(n, self.topology.n_dim)])

        # the perimeter of a dual element fundamental domain is given by this expression
        perimeter = [unsigned(corners[:, i + 1:]) for i in range(self.topology.n_dim)]

        # mean barycentric weights of an element is given by perimeter divided by the distance to all its corners
        W = [1] * (self.topology.n_dim + 1)
        for i in range(self.topology.n_dim):
            n = i + 1
            c = self.topology.n_dim - n
            W[n] = perimeter[c] / edge_length_prod(c)

        # we bias the averaging operators with these weights
        res = [1]
        # FIXME: a has bigger shape than w; this is the cause of divide by zero errors
        for i, (w, a) in enumerate(zip(W[1:], self.topology.dual.averaging_operators_0[1:])):
            M = scipy.sparse.coo_matrix((
                w,
                (domains[:, -(i + 2)], domains[:, -1])),
                shape=a.shape
            )
            q = a.multiply(M)
            res.append(normalize_l1(q, axis=1))

        return res

    @abstractmethod
    def unsigned_volume(self, pts):
        raise NotADirectoryError


class BaseComplexEuclidian(BaseComplex):

    def unsigned_volume(self, pts):
        from pycomplex.geometry import euclidian
        return euclidian.unsigned_volume(pts)

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

    def unsigned_volume(self, pts):
        from pycomplex.geometry import spherical
        return spherical.unsigned_volume(pts)

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
