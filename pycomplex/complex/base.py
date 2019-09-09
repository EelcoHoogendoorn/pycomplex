
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
        """Returns a simplicial complex

        Notes
        -----
        In the context of a simplicial complex, this is also known as a barycentric subdivision,
        but that does not generalize readily to a cubical complex.
        """
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
        weights = getattr(self, 'weights', None)
        kwargs = {}
        if weights is not None:
            kwargs['weights'] = weights[subset_topology.parent_idx[0]]
        return self.copy(
            topology=subset_topology,
            vertices=self.vertices[subset_topology.parent_idx[0]],
            **kwargs
        )

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
        import funcsigs
        args = funcsigs.signature(type(self)).parameters.keys()
        nkwargs = {}
        for a in args:
            if hasattr(self, a):
                nkwargs[a] = getattr(self, a)

        nkwargs.update(kwargs)
        c = type(self)(**nkwargs)
        c.parent = self
        return c

    def translate(self, vector):
        return self.copy(vertices=self.vertices + vector)

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
    def dual_metric_closed(self):
        """Dual metric, including dual boundary elements"""
        if self.boundary is None:
            return self.dual_metric
        return [np.concatenate(p) for p in zip(self.dual_metric, self.boundary.dual_metric + [[]])]

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

    @cached_property
    def positive_dual_metric(self):
        """Returns true if all dual metrics are positive"""
        return all([np.all(m > 0) for m in self.dual_metric])

    @cached_property
    def is_well_centered(self):
        """Test that all circumcenters are inside each element"""
        return all([np.all(b > 0) for b in self.primal_barycentric])

    @cached_property
    def is_pairwise_delaunay(self):
        """Test that adjacent circumcenters do not cross eachother, or that dual 1-metric is positive"""
        return np.all(self.remap_boundary_N(self.dual_edge_excess(), oriented=False) > 0)

    @cached_property
    def weighted_average_operators(self):
        """Weighted averaging of dual 0-forms by their barycentric mean coordinates
        In conjunction with the barycentric coordinates on each fundamental domain,
        this can be used to construct a barycentric interpolation of dual 0-forms;
        see `BaseComplex.sample_dual_0`

        This is equivalent to a mean-value barycentric interpolation,
        evaluated only at the position of primal/dual elements

        Returns
        -------
        array_like, [topology.n_dim+1] of sparse matrices
        n-th entry maps dual 0-form to dual of primal n-form (FIXME: returning reverse order is more natural)

        References
        ----------
        http://vcg.isti.cnr.it/Publications/2004/HF04/coordinates_aicm04.pdf
        https://www.researchgate.net/publication/2856409_Generalized_Barycentric_Coordinates_on_Irregular_Polygons

        Notes
        -----
        Divide by distance to dual vertex to make all other zero when approaching vertex
        Divide by distance to dual edge to make all other zero when approaching dual edge
        Divide by distance to dual face to make all other zero when approaching dual face
        and so on.

        General idea; divide perimeter of a dual element
        by the distance to all corners corresponding to lower order duals

        Requires a well-centered mesh.
        Getting this working on non-well centered meshes requires handling of sign in metric calculations,
        but also replacing the division by the distance to each corner of a domain,
        with a product to all other distances in the cell, to avoid divisions by zero.
        This strikes me as a bit daunting still

        While I think its really neat how well this can be implemented for any dimension,
        the utility thereof isnt that obvious, considering the seeming difficulty of finding
        well-centered simplicial meshes in dimensions greater than 2.
        """
        # FIXME: does this work for cubical, or should it be put in simplicial subclass?
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

    @cached_property
    def dual_flux_to_dual_velocity(self):
        """maps a dual-1-form (including boundary terms) representing tangent flux,
        to a vector field represented at dual vertices in the embedding space

        Parameters
        ----------
        flux_d1 : ndarray, [n_dual_edges, ...], float
            dual-1-form including boundary terms (which are currently ignored)

        Returns
        -------
        velocity_d0 : ndarray, [n_dual_vertices, n_dim, ...], float
            velocity vector in the embedding space at each dual vertex

        Notes
        -----
        This signature describes the docstring of the wrapped inner function
        """
        # FIXME: dual boundary fluxes are ignored, and boundary velocities are always zero.
        # FIXME: this is hardly ideal; we should be able to reason about dual fluxes incident on boundary edges as well
        # that said gradient on curved corner flux is not that well defined; but should the perfect be the enemy of the good?
        assert self.is_pairwise_delaunay    # required since we are dividing by dual edge lengths
        B = self.topology._boundary[-1]
        O = self.topology._orientation[-1]
        B = B.reshape(len(B), -1)
        O = O.reshape(len(O), -1)

        # from pycomplex.complex.simplicial.euclidian import ComplexSimplicialEuclidian
        from pycomplex.complex.simplicial.base import BaseComplexSimplicial
        from pycomplex.complex.regular import ComplexRegularMixin
        if isinstance(self, BaseComplexSimplicial):
            from pycomplex.geometry import euclidian
            # NOTE: analogous computation would be normal of cube faces multiplied with their area
            gradients = euclidian.simplex_gradients(self.vertices[self.topology.elements[-1]])

        elif isinstance(self, ComplexRegularMixin):
            pp = self.primal_position
            gradients = pp[-2][B] - pp[-1][:, None, :]
            # gradients -= gradients.mean(axis=1, keepdims=True)

            gradients = linalg.normalized(gradients)
            gradients *= self.primal_metric[-2][B][..., None]
        else:
            raise NotImplementedError


        u, s, v = np.linalg.svd(gradients)
        # only retain components in the plane of the element
        s[:, self.topology.n_dim:] = np.inf
        s = 1 / s
        pinv = np.einsum('...ij,...j,...jk->...ki', u[..., :s.shape[-1]], s, v)
        # gradients.dot(velocity) = normal_flux

        # for incompressible flows on simplicial topologies, there exists a 3-vector at the dual vertex,
        # which projected on the dual edges forms the dual fluxes. on a sphere the third component is not determined
        # approximate inverse would still make sense in cubical topology however;
        # here each velocity component should reduce to an averaging over the opposing faces
        # tangent_directions.dot(velocity) = tangent_velocity_component

        def block_diag(blocks):
            """Reorganize a multilinear operator of shape [b, r, c]
            as a sparse matrix with b diagonal blocks of shape [r, c]

            Parameters
            ----------
            blocks : ndarray, [b, r, c], float
                list of matrices to be used like y = einsum('brc,bc->br', blocks, x)

            Returns
            -------
            blocked : scipy.sparse, [b * r, b * c]
                equivalent linear operator to be used as y.flatten() == (blocked * x.flatten())
            """
            b, r, c = np.indices(blocks.shape)
            r = r + b * blocks.shape[1]
            c = c + b * blocks.shape[2]
            # cast to bsr matrix? perhaps little point since only precomputation
            return scipy.sparse.coo_matrix((blocks.flatten(), (r.flatten(), c.flatten())))

        def signed_selector(B, O):
            c = B
            r = np.arange(len(c), dtype=c.dtype)
            return scipy.sparse.coo_matrix((O.flatten(), (r.flatten(), c.flatten())))

        bpinv = block_diag(pinv)
        s = signed_selector(B.flatten(), O.flatten())       # this assembles all fluxes of each n-cube in 2*n rows per n_cube
        pd = scipy.sparse.diags(self.hodge_PD[-2])
        # core maps from interior dual edges to interior dual vertices * ndim
        core = (bpinv * s * pd).tocsr()    # perhaps drop near-zero terms?
        # NOTE: dual selectors are indexed by primal element order!
        S = self.topology.dual.selector_interior[-2]     # map from dual-1-elements to primal-n-1-elements; drop boundary fluxes
        P = self.topology.dual.selector_interior[-1].T   # map from primal-n-elements to dual-0-elements; pad boundary with zeros

        def dual_flux_to_dual_velocity(flux_d1):
            # lots of reshaping to support gufunc dimensions; otherwise quite simple product
            gu = flux_d1.shape[1:]
            gun = np.prod(gu, dtype=int)
            s1 = len(flux_d1), gun
            s2 = self.topology.n_elements[-1], self.n_dim * gun
            s3 = (self.topology.dual.n_elements[0], self.n_dim) + gu
            return (P * (core * (S * flux_d1.reshape(s1))).reshape(s2)).reshape(s3)
        return dual_flux_to_dual_velocity

    @abstractmethod
    def pick_primal(self, points):
        raise NotImplementedError
    @abstractmethod
    def pick_dual(self, points):
        raise NotImplementedError
    @abstractmethod
    def pick_fundamental(self, points):
        raise NotImplementedError

    @abstractmethod
    def sample_dual_0(self, d0, points):
        """Sample a dual 0-form at the given points in the embedding space,
        using linear interpolation over fundamental domains"""
        raise NotImplementedError

    @abstractmethod
    def unsigned_volume(self, pts):
        raise NotImplementedError

    @property
    @abstractmethod
    def metric(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def multigrid_transfers(self):
        raise NotImplementedError

    def plot_dual_flux(self, d1, **kwargs):
        """Visualize a dual 1-form flux by warping the complex in accordance with said flux"""
        defaults = dict(plot_dual=False, plot_vertices=False, plot_lines=False)
        defaults.update(kwargs)

        # map dual flux to primal velocity; kinda ugly
        S = self.topology.dual.selector_interior
        d0 = S[-1] * self.dual_flux_to_dual_velocity(S[1].T * d1)
        p0 = self.topology.averaging_operators_N[0] * d0
        # plot warped mesh
        return self.copy(vertices=self.primal_position[0] + p0).plot(**defaults)
