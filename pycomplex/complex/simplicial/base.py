
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.spatial

from pycomplex.math import linalg
from pycomplex.complex.base import BaseComplex


class BaseComplexSimplicial(BaseComplex):
    """(weighted) simplicial complex

    This is an abstract base class
    """

    def subdivide_fundamental(self, oriented=True):
        return self.copy(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_fundamental(oriented),
            weights=None
        )

    def subdivide_fundamental_transfer(self):
        raise NotImplementedError

    def subdivide_simplicial(self):
        PP = self.primal_position
        return self.copy(
            vertices=np.concatenate([PP[0], PP[-1]], axis=0),
            topology=self.topology.subdivide_simplicial(),
            weights=None
        )

    def subdivide_cubical(self):
        """Subdivide the simplicial complex into a cubical complex"""
        from pycomplex.complex.cubical import ComplexCubical
        return ComplexCubical(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_cubical(),
        )

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
        euclidian metric is used here; also usefull in spherical applications
        """
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

        try:
            # for spherical complexes, it suffices to simply view them as euclidian for the present purpose
            euclidian = self.as_euclidian()
        except:
            euclidian = self

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
        # since in practice local redistributions of dual edge length are the only ones of interest,
        # and more aggresive solvers tend to just make things worse by messing with the null space
        diag = laplacian.diagonal()
        weights = np.zeros_like(field)
        for i in range(1):
            weights = (rhs - laplacian * weights + diag * weights) / diag
        return self.copy(weights=weights)

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
        # FIXME: not sure why this check made sense? why cant parent by cubical?
        from pycomplex.topology.simplicial import TopologySimplicial
        assert isinstance(parent, TopologySimplicial)

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

    def homogenize(self, points):
        """Homogenize a set of points.
        In the euclidian case, this means adding a column of ones.
        In the spherical case, this is an identity mapping

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float

        Returns
        -------
        points : ndarray, [n_points, n_homogeneous_coordinates], float
        """
        raise NotImplementedError

    def augment(self, points, weights=None):
        """Augment a set of points with a weight-based coordinate

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float

        Returns
        -------
        points : ndarray, [n_points, n_dim + 1], float
        """
        if weights is None:
            offsets = np.zeros_like(points[..., 0])
        else:
            offsets = self.weights_to_offsets(weights)
        return np.concatenate([points, offsets[..., None]], axis=-1)

    def pick_primal_brute(self, points):
        """Added for debugging purposes"""
        _, basis = self.pick_precompute
        baries = np.einsum('bcv,pc->bpv', basis, points)
        quality = (baries * (baries < 0)).sum(axis=-1)
        simplex_index = np.argmax(quality, axis=0)
        return simplex_index

    @cached_property
    def pick_precompute(self):
        """Cached precomputations for picking operations

        Returns
        -------
        tree : CKDTree
            tree over augmented primal vertices
        basis : ndarray, [n_N-elements, n_homogeneous_coordinates, n_homogeneous_coordinates]
            pre-inverted basis of each N-simplex,
            for fast calculation of barycentric coordinates from homogeneous coordinates
        """
        points = self.primal_position[0]
        if self.weights is not None:
            points = self.augment(points, self.weights)
        tree = scipy.spatial.cKDTree(points)
        corners = self.vertices[self.topology.elements[-1]]
        basis = np.linalg.inv(self.homogenize(corners))
        return tree, basis

    def pick_primal_precomp(self, weight):
        """Precomputations for primal picking

        Returns
        -------
        tree : CKDTree
            tree over augmented dual vertices
        basis : ndarray, [n_N-elements, n_homogeneous_coordinates, n_homogeneous_coordinates]
            pre-inverted basis of each n-element,
            for fast barycentric calculation

        Notes
        -----
        Requires pairwise delaunay complex
        """
        assert self.is_pairwise_delaunay  # if centroids cross eachother, this method fails

        corners = self.vertices[self.topology.elements[-1]]
        dual_vertex = np.einsum('...cn,...c->...n', corners, self.primal_barycentric[-1])

        if weight:
            ee = self.dual_edge_excess(signed=False)
            # sum these around each n-1-simplex, or bounding face, to get n-1-form
            S = self.topology.selector_interior[-2]  # only consider interior simplex boundaries
            q = S * self.remap_boundary_N(ee, oriented=True)
            T = S * self.topology.matrices[-1]
            # solve T * w = q; that is,
            # difference in desired weights on simplices over faces equals difference in squared distance over boundary between simplices
            L = T.T * T
            rhs = T.T * q
            rhs = rhs - rhs.mean()  # this might aid numerical stability of minres
            # FIXME: could we use amg here?
            weight = scipy.sparse.linalg.minres(L, rhs, tol=1e-12)[0]
            points = self.augment(dual_vertex, weight)
        else:
            points = dual_vertex

        tree = scipy.spatial.cKDTree(points)
        basis = np.linalg.inv(self.homogenize(corners))
        return tree, basis

    @cached_property
    def pick_primal_precomp_weighted(self):
        return self.pick_primal_precomp(weight=True)

    @cached_property
    def pick_primal_precomp_unweighted(self):
        return self.pick_primal_precomp(weight=False)

    def pick_primal(self, points, simplex_idx=None, weighted=True):
        """Picking of primal simplex by means of a point query wrt its dual vertex
        Note that the crux of this functionality lies in the precomputation step,
        where the coordinates are augmented with an extra dimension to make this possible

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            query points in embedding space
        simplex_idx : ndarray, [n_points], index_dtype, optional
            can be used to exploit temporal coherence in queries

        Returns
        -------
        simplex_idx : ndarray, [n_points], index_dtype
            index of the primal simplex being picked
        bary : ndarray, [n_points, n_dim], float
            barycentric coordinates
            note that they are not normalized yet!
            this means they only sum to one of the query point lies exactly on the simplex
        """
        assert self.is_pairwise_delaunay
        if weighted:
            tree, basis = self.pick_primal_precomp_weighted
        else:
            tree, basis = self.pick_primal_precomp_unweighted

        def query(points):
            augmented = self.augment(points) if weighted else points
            dist, idx = tree.query(augmented)
            homogeneous = self.homogenize(points)
            baries = np.einsum('tcv,tc->tv', basis[idx], homogeneous)
            return idx, baries

        if simplex_idx is None:
            simplex_idx, baries = query(points)
        else:
            homogeneous = self.homogenize(points)
            baries = np.einsum('tcv,tc->tv', basis[simplex_idx], homogeneous)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                simplex_idx = simplex_idx.copy()
                s, b = query(points[update])
                simplex_idx[update] = s
                baries[update] = b

        baries /= baries.sum(axis=1, keepdims=True)

        return simplex_idx, baries

    def pick_dual(self, points):
        """Pick the dual elements. By definition of the voronoi dual,
        this lookup can be trivially implemented as a closest-point query

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            query points in embedding space

        Returns
        -------
        dual_element_idx : ndarray, [n_points], index_dtype
            primal vertex / dual N-simplex index
        """
        if self.weights is not None:
            points = self.augment(points)
        tree, _ = self.pick_precompute
        _, dual_element_index = tree.query(points)
        return dual_element_index

    @cached_property
    def pick_fundamental_precomp(self):
        """Perform precomputations for fundamental domain picking

        Returns
        -------
        subdivided : type(self)
            fundamental domain subdivision of self,
            with fundamental domain weight optimization applied,
            such that dual vertices do not coincide
        domains : ndarray, [n_points, n_dim], index_dtype
            n-th column corresponds to indices of n-element

        Notes
        -----
        This may be a bit expensive; but hey it is cached; and it sure is elegant
        """
        subdivided = self.subdivide_fundamental(oriented=True).optimize_weights_fundamental()
        domains = self.topology.fundamental_domains()
        domains = domains.reshape(-1, self.topology.n_dim + 1)
        return subdivided, domains

    def pick_fundamental(self, points, domain_idx=None):
        """Pick the fundamental domains

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick
        domain_idx : ndarray, [n_points], index_dtype, optional
            can be used to exploit temporal coherence in queries

        Returns
        -------
        domain_idx : ndarray, [n_points], index_dtype
            element idx of the fundamental domain being picked
        baries : ndarray, [n_points, n_dim] float
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
            averaging = self.weighted_average_operators
        else:
            averaging = self.topology.dual.averaging_operators_0
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = [a * d0 for a in averaging][::-1]

        # pick fundamental domains
        domain_idx, bary, domain = self.pick_fundamental(points)

        # reverse flips made for orientation preservation
        flip = np.bitwise_and(domain_idx, 1) == 1
        temp = bary[flip, -2]
        bary[flip, -2] = bary[flip, -1]
        bary[flip, -1] = temp

        # do interpolation over fundamental domain
        i = ((dual_forms[i][domain[:, i]].T * bary[:, i].T).T
            for i in range(self.topology.n_dim + 1))
        # sum over the contributions of all corners of the fundamental domain
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
