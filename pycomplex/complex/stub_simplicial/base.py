
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.spatial

from pycomplex.geometry import spherical
from pycomplex.math import linalg

from pycomplex.complex.base import BaseComplex


class BaseComplexSimplicial(BaseComplex):
    """Place things common to spherical and euclidian simplicial complexes here;
    scope of that duplicated code keeps growing"""

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

        Notes
        -----
        Well-centeredness is not a necessity for picking; delaunay would be fine
        a more intelligent algorithm here wouldnt hurt; this can fail for adversarial input

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

    def homogenize(self):
        raise NotImplementedError

    def pick_primal_brute(self, points):
        """Added for debugging purposes"""
        _, basis = self.pick_precompute
        baries = np.einsum('bcv,pc->bpv', basis, points)
        quality = (baries * (baries < 0)).sum(axis=-1)
        simplex_index = np.argmax(quality, axis=0)
        return simplex_index

    @cached_property
    def pick_primal_precomp(self):
        """Precomputations for primal picking

        Notes
        -----
        Requires pairwise delaunay complex
        """
        assert self.is_pairwise_delaunay    # if centroids cross eachother, this method fails
        ee = self.dual_edge_excess(signed=False)

        corners = self.vertices[self.topology.elements[-1]]
        dual_vertex = np.einsum('...cn,...c->...n', corners, self.primal_barycentric[-1])

        # sum these around each n-1-simplex, or bounding face, to get n-1-form
        S = self.topology.selector[-2]  # only consider interior simplex boundaries
        q = S * self.remap_boundary_N(ee, oriented=True)
        T = S * self.topology.matrices[-1]
        # solve T * w = q; that is,
        # difference in desired weights on simplices over faces equals difference in squared distance over boundary between simplices
        L = T.T * T
        rhs = T.T * q
        rhs = rhs - rhs.mean()  # this might aid numerical stability of minres
        weight = scipy.sparse.linalg.minres(L, rhs, tol=1e-12)[0]

        offset = self.weights_to_offsets(weight)
        augmented = np.concatenate([dual_vertex, offset[:, None]], axis=1)
        tree = scipy.spatial.cKDTree(augmented)

        basis = np.linalg.inv(self.homogenize(corners))

        return tree, basis

    def pick_primal(self, points, simplex_idx=None):
        """Picking of primal simplex by means of a point query wrt its dual vertex

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
        simplex_idx : ndarray, [n_points], index_dtype
            initial guess

        Returns
        -------
        simplex_idx : ndarray, [n_points], index_dtype
        bary : ndarray, [n_points, topology.n_dim + 1], float
            barycentric coordinates

        """
        assert self.is_pairwise_delaunay
        tree, basis = self.pick_primal_precomp

        def query(points):
            augmented = np.concatenate([points, np.zeros_like(points[:, :1])], axis=1)
            dist, idx = tree.query(augmented)
            homogeneous = np.concatenate([points], axis=1)
            baries = np.einsum('tcv,tc->tv', basis[idx], homogeneous)
            return idx, baries

        if simplex_idx is None:
            simplex_idx, baries = query(points)
        else:
            homogeneous = np.concatenate([points, np.ones_like(points[:, :1])], axis=1)
            baries = np.einsum('tcv,tc->tv', basis[simplex_idx], homogeneous)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                simplex_idx = simplex_idx.copy()
                s, b = query(points[update])
                simplex_idx[update] = s
                baries[update] = b

        baries /= baries.sum(axis=1, keepdims=True)

        return simplex_idx, baries

    @cached_property
    def pick_precompute(self):
        """Cached precomputations for picking operations"""
        c = self.primal_position[0]
        if self.weights is not None:
            # NOTE: if using this for primal simplex picking, we could omit the weights
            offsets = self.weights_to_offsets(self.weights)
            c = np.concatenate([c, offsets[:, None]], axis=1)
        tree = scipy.spatial.cKDTree(c)
        basis = np.linalg.inv(self.vertices[self.topology.elements[-1]])
        return tree, basis

    def pick_dual(self, points):
        """Pick the dual elements. By definition of the voronoi dual,
        this lookup can be trivially implemented as a closest-point query

        Returns
        -------
        ndarray, [self.topology.n_elements[0]], index_dtype
            primal vertex / dual element index
        """
        if self.weights is not None:
            points = np.concatenate([points, np.zeros_like(points[:, :1])], axis=1)
        tree, _ = self.pick_precompute
        # finding the dual face we are in is as simple as finding the closest primal vertex,
        # by virtue of the definition of duality
        _, dual_element_index = tree.query(points)

        return dual_element_index

    @cached_property
    def pick_fundamental_precomp(self):
        # this may be a bit expensive; but hey it is cached; and it sure is elegant
        fundamental = self.subdivide_fundamental(oriented=True).optimize_weights_fundamental()
        domains = self.topology.cubical_domains()
        domains = domains.reshape(-1, self.topology.n_dim + 1)
        return fundamental, domains

    def pick_fundamental(self, points, domain_idx=None):
        """Pick the fundamental domain

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick
        domain_idx : ndarray, [n_points], index_dtype, optional
            feed previous returned idx in to exploit temporal coherence in queries

        Returns
        -------
        domain_idx : ndarray, [n_points], index_dtype, optional
            returned if domain_idx input is not None
        baries: ndarray, [n_points, n_dim] float
            barycentric weights corresponding to the domain indices
        domains : ndarray, [n_points, n_dim], index_dtype
            n-th column corresponds to indices of n-element

        """
        assert self.is_pairwise_delaunay
        fundamental, domains = self.pick_fundamental_precomp

        domain_idx, bary = fundamental.pick_primal(points, simplex_idx=domain_idx)
        return domain_idx, bary, domains[domain_idx]

    def sample_dual_0(self, d0, points, weighted=True):
        """Sample a dual 0-form at the given points, using linear interpolation over fundamental domains

        Parameters
        ----------
        d0 : ndarray, [topology.dual.n_elements[0]], float
        points : ndarray, [n_points, self.n_dim], float

        Returns
        -------
        ndarray, [n_points], float
            dual 0-form sampled at the given points
        """
        if weighted:
            A = self.weighted_average_operators
        else:
            A = self.topology.dual.averaging_operators_0
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = [a * d0 for a in A][::-1]

        # pick fundamental domains
        domain_idx, bary, domain = self.pick_fundamental(points)

        # reverse flips made for orientation preservation
        flip = np.bitwise_and(domain_idx, 1) == 0
        temp = bary[flip, -2]
        bary[flip, -2] = bary[flip, -1]
        bary[flip, -1] = temp

        # do interpolation over fundamental domain
        i = [(dual_forms[i][domain[:, i]].T * bary[:, i].T).T
            for i in range(self.topology.n_dim + 1)]
        return sum(i)

    def sample_primal_0(self, p0, points):
        """Sample a primal 0-form at the given points, using linear interpolation over n-simplices

        Parameters
        ----------
        p0 : ndarray, [topology.n_elements[0]], float
        points : ndarray, [n_points, self.n_dim], float

        Returns
        -------
        ndarray, [n_points], float
        """
        simplex_idx, bary = self.pick_primal(points)
        simplices = self.topology.elements[-1]
        vertex_idx = simplices[simplex_idx]
        return (p0[vertex_idx] * bary).sum(axis=1)

    @cached_property
    def metric(self):
        """Compute metrics from fundamental domain contributions

        Notes
        -----
        There is a lot of duplication in metric calculation this way.
        Would it pay to construct the fundamental topology first?

        Only works up to ndim=2 for now. general spherical case seems hard
        we might approximate by doing some additional subdivision steps using cubes, and approximating with euclidian measure
        Or we could simply cast to euclidian, and return the metric of that
        """
        topology = self.topology
        assert topology.is_oriented
        assert self.is_well_centered
        assert self.topology.n_dim <= 2     # what to do in higher dims? numerical quadrature?

        PP = self.primal_position
        domains = self.topology.cubical_domains()

        domains = domains.reshape(-1, domains.shape[-1])
        corners = np.concatenate([p[d][:, None, :] for p, d in zip(PP, domains.T)], axis=1)

        PN = topology.n_elements
        DN = PN[::-1]

        # metrics
        PM = [np.zeros(n) for n in PN]
        PM[0][...] = 1
        DM = [np.zeros(n) for n in DN]
        DM[0][...] = 1

        raise Exception
        unsigned = spherical.unsigned_volume
        from scipy.misc import factorial
        groups = [npi.group_by(c) for c in domains.T]   # cache groupings since they may get reused

        for i in range(1, self.topology.n_dim):
            n = i + 1
            d = self.topology.n_dim - i
            PM[i] = groups[i].mean(unsigned(corners[:, :n]))[1] * factorial(n)
            DM[i] = groups[d].sum (unsigned(corners[:, d:]))[1] / factorial(d+1)

        V = spherical.unsigned_volume(corners)
        PM[-1] = groups[-1].sum(V)[1]
        DM[-1] = groups[+0].sum(V)[1]

        return (
            [m * (self.radius ** i) for i, m in enumerate(PM)],
            [m * (self.radius ** i) for i, m in enumerate(DM)]
        )
