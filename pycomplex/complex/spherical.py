
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

    def as_2(self):
        return ComplexSpherical2(vertices=self.vertices, topology=self.topology.as_2())

    def as_euclidian(self):
        from pycomplex.complex.simplicial import ComplexSimplicial
        return ComplexSimplicial(vertices=self.vertices, topology=self.topology)

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

    def pick_primal_brute(self, points):
        """Added for debugging purposes"""
        _, basis = self.pick_precompute
        baries = np.einsum('bcv,pc->bpv', basis, points)
        quality = (baries * (baries < 0)).sum(axis=-1)
        simplex_index = np.argmax(quality, axis=0)
        return simplex_index

    def pick_primal(self, points, simplex_idx=None):
        """Pick triangles and their barycentric coordinates on the sphere

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points on the sphere to pick
        simplex_idx : ndarray, [n_points], index_dtype, optional
            guesses as to the simplex that contains the point;
            can be used to exploit temporal coherence

        Returns
        -------
        simplex_idx : ndarray, [n_points], index_dtype
        bary : ndarray, [n_points, n_dim], float

        Notes
        -----
        Probably not super efficient, but it is fully vectorized, and fully n-dim

        """
        assert self.is_well_centered    # this is needed such that the ring of triangles logic works
        tree, basis = self.pick_precompute

        def query(points):
            vertex_index = self.pick_dual(points)
            # construct all point-simplex combinations we need to test for;
            # matrix is compressed-row so row indexing should be efficient
            T = self.topology.matrix(self.topology.n_dim, 0)[vertex_index].tocoo()
            point_idx, simplex_index = T.row, T.col
            baries = np.einsum('tcv,tc->tv', basis[simplex_index], points[point_idx])
            # pick the one with the least-negative coordinates
            quality = (baries * (baries < 0)).sum(axis=1)
            _, best = npi.group_by(point_idx).argmax(quality)   # point_idx already sorted; can we make an optimized index for that?
            simplex_index, baries = simplex_index[best], baries[best]
            return simplex_index, baries

        if simplex_idx is None:
            simplex_idx, baries = query(points)
        else:
            baries = np.einsum('tcv,tc->tv', basis[simplex_idx], points)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                simplex_idx = simplex_idx.copy()
                s, b = query(points[update])
                simplex_idx[update] = s
                baries[update] = b

        baries /= baries.sum(axis=1, keepdims=True)
        return simplex_idx, baries

    def subdivide_fundamental(self):
        """Perform fundamental-domain subdivision"""
        return type(self)(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_fundamental()
        )

    def subdivide_cubical(self):
        """Subdivide the spherical simplical complex into a cubical complex"""
        from pycomplex.complex.cubical import ComplexCubical
        return ComplexCubical(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_cubical()
        )

    @cached_property
    def pick_fundamental_precomp(self):
        domains = self.topology.fundamental_domains()
        PP = self.primal_position
        p = np.empty(domains.shape + (self.n_dim,))
        for i in range(self.topology.n_dim + 1):
            p[..., i, :] = PP[i][domains[..., i]]

        domains_index = npi.as_index(domains.reshape(-1, domains.shape[-1]))

        return domains, np.linalg.inv(p).astype(np.float32), domains_index

    def pick_fundamental(self, points, domain_idx=None):
        """Pick the fundamental domain

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick
        domain_idx : ndarray, [n_points], index_dtype, optional
            feed previous returned idx in to exploit temporal coherence in queries
            if True, idx are returned despite not being given

        Returns
        -------
        domains : ndarray, [n_points, n_dim], index_dtype
            n-th column corresponds to indices of n-element
        baries: ndarray, [n_points, n_dim] float
            barycentric weights corresponding to the domain indices
        domain_idx : ndarray, [n_points], index_dtype, optional
            returned if domain_idx input is not None

        Notes
        -----
        Using a weighted fundamental subdivision, directly picking the right domain should in fact be possible
        Certainly in a euclidian space; need to think about the spherical case a little more
        """
        assert self.positive_dual_metric
        domains, basis, domains_index = self.pick_fundamental_precomp

        def query(points):
            n_points, n_dim = points.shape
            # FiXME: can be unified with its own tree; midpoint between primals and duals
            if False:
                primal, bary = self.pick_primal_alt(points) # FIXME: still have failures for 4-spheres sometimes. would picking primal/dual together fix this?
            else:
                primal, bary = self.pick_primal(points)
            dual = self.pick_dual(points)

            # get all fundamental domains that match both primal and dual, and brute-force versus their precomputed inverses
            d = domains[primal].reshape(n_points, -1, n_dim)
            b = basis  [primal].reshape(n_points, -1, n_dim, n_dim)
            s = np.where(d[:, :, 0] == dual[:, None])
            d = d[s].reshape(n_points, -1, n_dim)
            b = b[s].reshape(n_points, -1, n_dim, n_dim)

            # now get the best fitting domain from the selected set
            # FIXME: how to handle inverted domains?
            baries = np.einsum('tpcv,tc->tpv', b, points)
            quality = (baries * (baries < 0)).sum(axis=2)
            best = np.argmax(quality, axis=1)
            r = np.arange(len(points), dtype=index_dtype)
            d = d[r, best]
            baries = baries[r, best]
            return d, baries

        if domain_idx is None:
            domain, baries = query(points)
            baries /= baries.sum(axis=1, keepdims=True)
            return domain, baries
        elif domain_idx is True:
            domain, baries = query(points)
            baries /= baries.sum(axis=1, keepdims=True)
            domain_idx = npi.indices(domains_index, domain)
            return domain, baries, domain_idx
        else:
            baries = np.einsum('tcv,tc->tv', basis.reshape((-1, ) + basis.shape[-2:])[domain_idx], points)
            domain = domains.reshape(-1, domains.shape[-1])[domain_idx]
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                domain_idx = domain_idx.copy()
                d, b = query(points[update])
                domain_idx[update] = npi.indices(domains_index, d)
                baries[update] = b
                domain[update] = d
                baries /= baries.sum(axis=1, keepdims=True)
            return domain, baries, domain_idx

    @cached_property
    def pick_primal_alt_precomp(self):
        """Can we find a power/weight for each dual such that nearest weighted dual vertex gives us the primal element?
        think so! for each element, compute the distance to all its bounding elements.
        the diff of that distance is the required diff in weight over a dual edge.
        to find weights satisfying that diff, is essentially the dual of streamfunction;
        we have a dual 1-form that is closed wrt dual 2-form by construction

        how to handle boundary? just discard negative baries?

        can we generalize this to any element? query nearest edges, f.i?
        edge query seems hard. we can split our simplex however, by inserting a vert in the middle only
        that should give a new mesh that we can apply the same primal-picking logic to
        that then gives face and edge in one swoop. tri would have angle >> 90 degree typically though,
        so does not work great with this picking strategy

        Notes
        -----
        As it turned out, the logic for optimizing the primal vertex weights is eerily similar;
        not sure I fully grasp all the implications thereof.
        """
        assert self.topology.is_closed
        euc = self.as_euclidian().copy(weights=self.weights)
        assert self.is_pairwise_delaunay    # if centroids cross eachother, this method fails
        # FIXME as euclidian appears to do the trick; does lead to cracks tho, but more understabale. think about this aspect more!
        # worst case, crack can be solved by considering opposing simplex in case of negative bary
        # edge excess is the squared distance of each N-dual to its bounding n-1 duals; shape [n_N-simplices, N+1]
        ee = euc.dual_edge_excess(signed=False)
        # this is like primal position, but without the normalization. otherwise, our dual edge excess should also be in spherical coordinates
        corners = self.vertices[self.topology.elements[-1]]
        dual_vertex = np.einsum('...cn,...c->...n', corners, euc.primal_barycentric[-1])

        # sum these around each n-1-simplex, or bounding face, to get n-1-form
        q = self.remap_boundary_N(ee, oriented=True)
        T = self.topology.matrices[-1]
        # solve T * w = q; that is,
        # difference in desired weights on simplices over faces equals difference in squared distance over boundary
        L = T.T * T
        rhs = T.T * q
        # rhs = rhs - rhs.mean()
        weight = scipy.sparse.linalg.minres(L, rhs, tol=1e-16)[0]
        # print(weight.min(), weight.max())
        offset = self.weights_to_offsets(weight)

        # PP = self.primal_position
        augmented = np.concatenate([dual_vertex, offset[:, None]], axis=1)
        tree = scipy.spatial.cKDTree(augmented)

        basis = np.linalg.inv(corners)

        return tree, basis

    def pick_primal_alt(self, points, simplex=None):
        """Picking of primal simplex by means of a point query wrt its dual vertex

        Parameters
        ----------
        points
        simplex

        Returns
        -------

        Notes
        -----
        Seems to work fine; except when it does not.
        This strikes me as a numerical stability issue rather than an error in the logic here
        """
        assert self.is_pairwise_delaunay
        tree, basis = self.pick_primal_alt_precomp

        def query(points):
            augmented = np.concatenate([points, np.zeros_like(points[:, :1])], axis=1)
            dist, idx = tree.query(augmented)
            baries = np.einsum('tcv,tc->tv', basis[idx], points)
            return idx, baries

        if simplex is None:
            simplex, baries = query(points)
        else:
            baries = np.einsum('tcv,tc->tv', basis[simplex], points)
            update = np.any(baries < 0, axis=1)
            # print('misses: ', update.mean())
            if np.any(update):
                simplex = simplex.copy()
                s, b = query(points[update])
                simplex[update] = s
                baries[update] = b

        baries /= baries.sum(axis=1, keepdims=True)

        return simplex, baries

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
    def pick_cube_precompute(self):
        """Pick the `cube` of a set of points

        A cube here is defined as the intersection between a primal and dual element.
        For a simplex, this always forms a cubical region

        A cube is indicated
        """
        assert self.topology.is_closed
        # assert self.positive_dual_metric  # implement spherical metric in higher dimensions so we can enable this

        q = self.remap_boundary_N(self.dual_edge_excess())

        T = self.topology.matrices[-1]
        L = T.T * T
        rhs = T.T * q

        weight = scipy.sparse.linalg.minres(L, rhs, tol=1e-16)[0]

        # offset = self.weights_to_offsets(weight) / 8 * 7
        offset = self.weights_to_offsets(weight) / 2

        # PP = self.primal_position
        PP = [(np.einsum('...cn,...c->...n', self.vertices[c], b))
                for c, b in zip(self.topology.corners, self.primal_barycentric)]

        IN0 = self.topology.elements[-1]
        n_corners = IN0.shape[-1]
        P_idx = np.arange(IN0.size) // n_corners
        D_idx = IN0.flatten()
        # FIXME: how to locate the sampling points? midpoints of unweighted centroids?
        mid = linalg.normalized(PP[0][D_idx] + PP[-1][P_idx] * 1)

        # FIXME: can we construct some type of basis that will inform us of fundamental domain we are in, in a single call?
        # all fundamental domains meet at both primal and dual vertex. lines seperating them must be linear equations
        augmented = np.concatenate([mid, np.repeat(offset[:, None], n_corners, axis=1).reshape(-1, 1)], axis=1)
        tree = scipy.spatial.cKDTree(augmented)
        return tree, P_idx, D_idx

    def pick_cube(self, points):
        points = np.concatenate([points, np.zeros_like(points[:, :1])], axis=1)

        tree, P_idx, D_idx = self.pick_cube_precompute
        _, idx = tree.query(points)

        return idx

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

    def sample_dual_0(self, d0, points):
        """Sample a dual 0-form at the given points, using linear interpolation over fundamental domains

        Parameters
        ----------
        d0 : ndarray, [topology.dual.n_elements[0]], float
        points : ndarray, [n_points, self.n_dim], float

        Returns
        -------
        ndarray, [n_points], float
        """
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = [a * d0 for a in self.weighted_average_operators]
        domain, bary = self.pick_fundamental(points)
        # do interpolation over fundamental domain
        return sum([dual_forms[i][domain[:, i]] * bary[:, [i]]
                    for i in range(self.topology.n_dim + 1)])

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
        element, bary = self.pick_primal(points)
        IN0 = self.topology.incidence[-1, 0]
        verts = IN0[element]
        return (p0[verts] * bary).sum(axis=1)


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
                topology = topology.as_2()
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