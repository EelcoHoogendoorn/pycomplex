
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

    def __init__(self, vertices, simplices=None, topology=None, radius=1):
        self.vertices = np.asarray(vertices)
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

    @cached_property
    def primal_lookup(self):
        """Cached precomputations for spherical picking operations"""
        tree = scipy.spatial.cKDTree(self.primal_position[0])
        basis = np.linalg.inv(self.vertices[self.topology.elements[-1]])
        return tree, basis

    def pick_primal(self, points, simplex=None):
        """Pick triangles and their barycentric coordinates on the sphere

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points on the sphere to pick
        simplex : ndarray, [n_points], index_dtype, optional
            guesses as to the simplex that contains the point;
            can be used to exploit temporal coherence

        Returns
        -------
        simplex : ndarray, [n_points], index_dtype
        bary : ndarray, [n_points, n_dim], float

        Notes
        -----
        Probably not super efficient, but it is fully vectorized, and fully n-dim

        """
        tree, basis = self.primal_lookup
        n_points, n_dim = points.shape

        def query(points):
            _, vertex_index = tree.query(points)
            # construct all point-simplex combinations we need to test for;
            # matrix is compressed-row so row indexing should be efficient
            T = self.topology.matrix(self.topology.n_dim, 0)[vertex_index].tocoo()
            point_idx, simplex_index = T.row, T.col
            baries = np.einsum('tcv,tc->tv', basis[simplex_index], points[point_idx])
            # pick the one with the least-negative coordinates
            quality = (baries * (baries < 0)).sum(axis=1)
            _, best = npi.group_by(point_idx).argmax(quality)   # point_idx already sorted; can we make an optimized index for that?
            simplex_index, baries = simplex_index[best], baries[best]
            # normalize
            return simplex_index, baries

        if simplex is None:
            simplex, baries = query(points)
        else:
            baries = np.einsum('tcv,tc->tv', basis[simplex], points)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                simplex = simplex.copy()
                s, b = query(points[update])
                simplex[update] = s
                baries[update] = b

        baries /= baries.sum(axis=1, keepdims=True)
        return simplex, baries

    def subdivide_fundamental(self):
        return type(self)(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_fundamental()
        )

    def remap_boundary(self, field):
        """Given a quantity computed on each n-simplex-boundary, combine the contributions of each incident n-simplex

        Parameters
        ----------
        field : ndarray, [n_simplices, n_corners], float
            a quantity defined on each boundary of all simplices

        Returns
        -------
        field : ndarray, [n_boundary_simplices], float
        """
        INn = self.topology._boundary[-1]
        ONn = self.topology._orientation[-1]
        _, field = npi.group_by(INn.flatten()).sum((field * ONn).flatten())
        return field

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
        what about constructing the actual fundamental domain? has a 90 deg angle, resulting in two
        points located at the midpoint of edge joining primal and dual.

        could consider those as a single point, and pick doubled fundamental domain instead.
        by equilaterality, all such weights should be constant over the simplex
        this might work as a method of picking primal and dual at the same time, but still no edge sidedness


        tiling edge domain with two tris does result in well conditioned tris; and an interesting split generally
        not entirely sure what to do at boundary, but seems like we should have the dofs there to make it work
        split does not map tet meshes to tet meshes though..


        aside from that, is there a simple rule for getting the nearest edge point in 2d case?
        yes; we have simplex baries. it is not the edge opposite dual index.
        wait; easier to just brute-force all candidate-fundamental domains at this point
        how to generate fundamental domains? list of n-element indices?
        having those would make n-dim metric a lot easier too

        """
        assert self.topology.is_closed
        # DP = self.dual_position
        PP = self.primal_position
        tri_edge = PP[-2][self.topology._boundary[-1]]
        delta = PP[-1][:, None, :] - tri_edge
        d = np.linalg.norm(delta, axis=2) ** 2        # fixme: this should be signed distance
        q = self.remap_boundary(d)
        T = self.topology.matrices[-1]
        rhs = T.T * q
        L = T.T * T
        power = scipy.sparse.linalg.minres(L, rhs, tol=1e-16)[0]
        # print(np.abs(T * power - q).max())

        # power += power.min()
        power -= power.max()
        augmented = np.concatenate([PP[-1], ((-power) ** 0.5)[:, None]], axis=1)
        tree = scipy.spatial.cKDTree(augmented)

        basis = np.linalg.inv(self.vertices[self.topology.elements[-1]])

        return tree, basis

    @cached_property
    def pick_fundamental_precomp(self):
        domains = self.topology.fundamental_domains()
        PP = self.primal_position
        p = np.empty(domains.shape + (self.n_dim,))
        for i in range(self.topology.n_dim + 1):
            p[..., i, :] = PP[i][domains[..., i]]
        return domains, np.linalg.inv(p).astype(np.float32)

    def pick_fundamental(self, points, domain=None):
        """Pick the fundamental domain

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick

        Returns
        -------
        domains : ndarray, [n_points, n_dim], index_dtype
            n-th column corresponds to indices of n-element
        baries: ndarray, [n_points, n_dim] float
            barycentric weights corresponding to the domain indices

        """
        domains, basis = self.pick_fundamental_precomp

        def query(points):
            n_points, n_dim = points.shape
            # FiXME: can be unified with its own tree; midpoint between primals and duals
            primal, bary = self.pick_primal(points)     # FIXME: use alt here? add kwarg for fast or stable
            dual = self.pick_dual(points)

            # get all fundamental domains that match both primal and dual, and brute-force versus their precomputed inverses
            d = domains[primal].reshape(n_points, -1, n_dim)
            b = basis  [primal].reshape(n_points, -1, n_dim, n_dim)
            s = np.where(d[:, :, 0] == dual[:, None])
            d = d[s].reshape(n_points, -1, n_dim)
            b = b[s].reshape(n_points, -1, n_dim, n_dim)

            # now get the best fitting domain from the selected set
            baries = np.einsum('tpcv,tc->tpv', b, points)
            quality = (baries * (baries < 0)).sum(axis=2)
            best = np.argmax(quality, axis=1)
            r = np.arange(len(points), dtype=index_dtype)
            d = d[r, best]
            baries = baries[r, best]
            return d, baries

        if domain is None:
            domain, baries = query(points)
        else:
            raise NotImplementedError('Need to think about how to cache fundamental domain hits. cache the primal/dual query?')
            baries = np.einsum('tcv,tc->tv', basis[domain], points)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                domain = domain.copy()
                d, b = query(points[update])
                domain[update] = d
                baries[update] = b

        baries /= baries.sum(axis=1, keepdims=True)
        return domain, baries

    def pick_primal_alt(self, points, simplex=None):
        """

        Parameters
        ----------
        points
        simplex

        Returns
        -------

        """
        tree, basis = self.pick_primal_alt_precomp

        def query(points):
            augmented = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
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
        tree, _ = self.primal_lookup
        # finding the dual face we are in is as simple as finding the closest primal vertex,
        # by virtue of the definition of duality
        _, dual_face_index = tree.query(points)

        return dual_face_index
        # to get the dual baries, would ideally do something like this:
        # https://pdfs.semanticscholar.org/6150/43145ebd38e2ae1fcf714f1d445c2d3a4308.pdf
        # but something simpler might suffice for now
        # http://www.geometry.caltech.edu/pubs/BLTD16.pdf
        # this is also looking good
        # http://www.geometry.caltech.edu/pubs/MMdGD11.pdf
        # as is this...

    @cached_property
    def metric(self):
        """Compute metrics from fundamental domain contributions

        Notes
        -----
        There is a lot of duplication in metric calculation this way.
        Would it pay to construct the fundamental topology first?
        """
        topology = self.topology
        assert topology.is_oriented
        assert self.is_acute
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

    @cached_property
    def is_acute(self):
        """Test that circumcenter is inside each simplex

        Notes
        -----
        code duplication with simplicial complex; need a mixin class i think
        """
        import pycomplex.geometry.euclidian
        corners = self.vertices[self.topology.corners[-1]]
        mean = corners.mean(axis=-2, keepdims=True)  # centering
        corners = corners - mean
        bary = pycomplex.geometry.euclidian.circumcenter_barycentric(corners)
        return np.all(bary >= 0)

    def weighted_average_operators(self):
        """Weight averaging over the duals by their barycentric coordinates

        Using mean coordinates for now, since they are simple to implement in a vectorized an nd-manner

        General logic is pretty simple; take the perimeter of an n-element, and divide it by the distance to
        all lower-order elements.

        Divide by distance to dual vertex to make all other zero when approaching vertex
        Divide by distance to dual edge to make all other zero when approaching dual edge
        Divide by distance to dual face to make all other zero when approaching dual face
        and so on.

        References
        ----------
        http://vcg.isti.cnr.it/Publications/2004/HF04/coordinates_aicm04.pdf
        https://www.researchgate.net/publication/2856409_Generalized_Barycentric_Coordinates_on_Irregular_Polygons

        FIXME: only closed cases for now; need to add boundary handling

        This also is pure duplication relative to simplicial case, except for choice of metric

        """
        topology = self.topology
        assert topology.is_oriented
        assert self.topology.is_closed
        assert self.is_acute

        PP = self.primal_position
        domains = self.topology.fundamental_domains()

        domains = domains.reshape(-1, domains.shape[-1])
        corners = np.concatenate([p[d][:, None, :] for p, d in zip(PP, domains.T)], axis=1)

        unsigned = spherical.unsigned_volume

        # construct required distances; all edges lengths
        def edge_length(a, b):
            return unsigned(corners[:, [a, b]])
        def edge_length_prod(n):
            return functools.reduce(operator.mul, [edge_length(n, m + 1) for m in range(n, self.topology.n_dim)])

        perimiter = [unsigned(corners[:, i+1:]) for i in range(self.topology.n_dim)]

        W = [1] * (self.topology.n_dim + 1)
        for i in range(self.topology.n_dim):
            n = i + 1
            c = self.topology.n_dim - n
            W[n] = perimiter[c] / edge_length_prod(c)

        res = [1]
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
    def cached_averages(self):
        # note: weighted average is more correct, but the difference appears very minimal in practice
        # return self.weighted_average_operators()
        return self.topology.dual.averaging_operators_0()

    def average_dual(self, d0):
        return [a * d0 for a in self.cached_averages]

    def sample_dual_0(self, d0, points):
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = self.average_dual(d0)[::-1]
        domain, bary = self.pick_fundamental(points)
        # do interpolation over fundamental domain
        return sum([dual_forms[i][domain[:, i]] * bary[:, [i]]
                    for i in range(self.topology.n_dim + 1)])
    def sample_primal_0(self, p0, points):
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

    def __init__(self, vertices, simplices=None, topology=None, radius=1):
        self.vertices = np.asarray(vertices)

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