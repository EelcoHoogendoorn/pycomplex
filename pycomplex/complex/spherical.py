
import functools
import operator
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.spatial

from pycomplex.complex.base import BaseComplexSpherical
from pycomplex.geometry import spherical
from pycomplex.math import linalg
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial
from pycomplex.topology import index_dtype
from pycomplex.sparse import normalize_l1


class ComplexSpherical(BaseComplexSpherical):
    """Complex on an n-sphere"""

    def __init__(self, vertices, simplices=None, topology=None, radius=1, weights=None):
        self.vertices = np.asarray(vertices)
        self.weights = weights
        if topology is None:
            topology = TopologySimplicial.from_simplices(simplices).fix_orientation()
            assert topology.is_oriented
        self.topology = topology
        self.radius = radius

    def plot(self, plot_dual=True, backface_culling=False, plot_vertices=True, ax=None, primal_color='b', dual_color='r'):
        """Visualize a complex on a 2-sphere; a little more involved than the other 2d cases"""
        import matplotlib.pyplot as plt
        import matplotlib.collections

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        def from_se(s, e):
            return np.concatenate([s[:, None, :], e[:, None, :]], axis=1)

        def subdivide(edges, steps):
            f = np.linspace(0, 1, steps)
            i = np.array([f, 1-f])
            edges = edges[:, :, None, :] * i[None, :, :, None]
            edges = edges.sum(axis=1)
            s = edges[:, :-1, None, :]
            e = edges[:, +1:, None, :]
            edges = np.concatenate([s, e], axis=2)
            edges = edges.reshape(-1, 2, edges.shape[-1])
            return linalg.normalized(edges)

        def plot_edge(ax, lines, **kwargs):
            if backface_culling:
                z = lines[..., 2]
                drawn = (z > 0).all(axis=1)
                lines = lines[drawn]
            lc = matplotlib.collections.LineCollection(lines[..., :2], **kwargs)
            ax.add_collection(lc)

        def plot_vertex(ax, points, **kwargs):
            if backface_culling:
                z = points[..., 2]
                drawn = z > 0
                points = points[drawn]
            ax.scatter(*points.T[:2], **kwargs)

        # plot outline of embedding space
        angles = np.linspace(0, 2*np.pi, 1000)
        ax.plot(np.cos(angles), np.sin(angles), color='k')

        # plot primal edges
        edges = self.topology.corners[1]
        steps = int(1000 / len(edges)) + 2
        e = subdivide(self.vertices[edges], steps=steps*2)
        plot_edge(ax, e, color=primal_color, alpha=0.5)
        if plot_vertices:
            plot_vertex(ax, self.vertices, color=primal_color)

        if plot_dual:
            # plot dual edges
            dual_vertices, dual_edges = self.dual_position[:2]
            dual_topology = self.topology.dual
            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:, 0], de[:, 1]
            s = subdivide(from_se(dual_edges, s), steps=steps)
            plot_edge(ax, s, color=dual_color, alpha=0.5)
            e = subdivide(from_se(dual_edges, e), steps=steps)
            plot_edge(ax, e, color=dual_color, alpha=0.5)
            if plot_vertices:
                plot_vertex(ax, dual_vertices, color=dual_color)

        plt.axis('equal')

    def subdivide_fundamental(self, oriented=True):
        return type(self)(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_fundamental(oriented)
        )

    def as_2(self):
        return ComplexSpherical2(
            vertices=self.vertices, topology=self.topology.as_2(), weights=self.weights)

    def as_euclidian(self):
        from pycomplex.complex.simplicial import ComplexSimplicialEuclidian
        return ComplexSimplicialEuclidian(vertices=self.vertices, topology=self.topology)

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

        homogeneous = np.concatenate([corners], axis=-1)    # no extra coord here
        basis = np.linalg.inv(homogeneous)

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
        domains = self.topology.fundamental_domains()
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
        domains = self.topology.fundamental_domains()

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

    # def sample_dual_0(self, d0, points):
    #     """Sample a dual 0-form at the given points, using linear interpolation over fundamental domains
    #
    #     Parameters
    #     ----------
    #     d0 : ndarray, [topology.dual.n_elements[0], ...], float
    #     points : ndarray, [n_points, self.n_dim], float
    #
    #     Returns
    #     -------
    #     ndarray, [n_points, ...], float
    #     """
    #     # extend dual 0 form to all other dual elements by averaging
    #     dual_forms = [a * d0 for a in self.weighted_average_operators]
    #     domain, bary = self.pick_fundamental(points)
    #     # do interpolation over fundamental domain
    #     return sum([(dual_forms[i][domain[:, i]].T * bary[:, i]).T
    #                 for i in range(self.topology.n_dim + 1)])
    #
    # def sample_primal_0(self, p0, points):
    #     """Sample a primal 0-form at the given points, using linear interpolation over n-simplices
    #
    #     Parameters
    #     ----------
    #     p0 : ndarray, [topology.n_elements[0]], float
    #     points : ndarray, [n_points, self.n_dim], float
    #
    #     Returns
    #     -------
    #     ndarray, [n_points], float
    #     """
    #     element, bary = self.pick_primal(points)
    #     IN0 = self.topology.incidence[-1, 0]
    #     verts = IN0[element]
    #     return (p0[verts] * bary).sum(axis=1)


class ComplexCircular(ComplexSpherical):
    """Simplicial complex on the surface of a 1-sphere
    cant really think of any applications, other than testing purposes

    """

    def subdivide(self):
        return self.subdivide_fundamental()


class ComplexSpherical2(ComplexSpherical):
    """Simplicial complex on the surface of a 2-sphere"""

    def __init__(self, vertices, simplices=None, topology=None, radius=1, weights=None):
        self.vertices = np.asarray(vertices)
        self.weights = weights

        if topology is None:
            topology = TopologyTriangular.from_simplices(simplices)
        else:
            try:
                topology = topology
            except:
                pass

        assert isinstance(topology, TopologyTriangular)
        self.topology = topology
        self.radius = radius

    def subdivide(self):
        """Subdivide the complex, returning a refined complex where each edge inserts a vertex

        This is a loop-like subdivision

        """
        return ComplexSpherical2(
            vertices=np.concatenate(self.primal_position[:2], axis=0),
            topology=self.topology.subdivide()
        )

    def as_euclidian(self):
        from pycomplex.complex.simplicial import ComplexTriangularEuclidian3
        return ComplexTriangularEuclidian3(vertices=self.vertices, topology=self.topology)

    def multigrid_transfers(self):
        """Port multigrid transfers from escheresque.
        Can we re-use weighted averaging bary logic for this?
        No, seems we are mostly stuck with existing logic.
        Crux is in the middle triangle.
        We can query coarse dual vertices with fine dual vertices of central triangles to figure out the conditional
        """


class ComplexSpherical3(ComplexSpherical):
    """Figuring out the metric computations for this will be a hard one.
    But doing physical simulations in a curved space should be fun
    """
    pass