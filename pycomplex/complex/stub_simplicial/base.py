
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.spatial

from pycomplex.geometry import spherical

from pycomplex.complex.base import BaseComplex


class BaseComplexSimplicial(BaseComplex):
    """Place things common to spherical and euclidian simplicial complexes here;
    scope of that duplicated code keeps growing"""

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
        """Cached precomputations for spherical picking operations"""
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
