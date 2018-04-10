
from cached_property import cached_property

import numpy as np
import numpy_indexed as npi
import scipy.spatial

from pycomplex.complex.simplicial.base import BaseComplexSimplicial
from pycomplex.geometry import spherical
from pycomplex.math import linalg
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial


class ComplexSpherical(BaseComplexSimplicial):
    """Simplicial complex on an n-sphere"""

    def __init__(self, vertices, simplices=None, topology=None, radius=1, weights=None):
        self.vertices = np.asarray(vertices)
        self.weights = weights
        if topology is None:
            topology = TopologySimplicial.from_simplices(simplices).fix_orientation()
            assert topology.is_oriented
        self.topology = topology
        self.radius = radius

    def as_euclidian(self):
        from pycomplex.complex.simplicial.euclidian import ComplexSimplicialEuclidian
        return ComplexSimplicialEuclidian(vertices=self.vertices, topology=self.topology, weights=self.weights)

    def homogenize(self, points):
        return points

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
            from pycomplex.topology.util import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:, 0], de[:, 1]
            s = subdivide(from_se(dual_edges, s), steps=steps)
            plot_edge(ax, s, color=dual_color, alpha=0.5)
            e = subdivide(from_se(dual_edges, e), steps=steps)
            plot_edge(ax, e, color=dual_color, alpha=0.5)
            if plot_vertices:
                plot_vertex(ax, dual_vertices, color=dual_color)

        ax.axis('equal')

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
        assert self.is_well_centered    # could be relaxed once signs are handled properly
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


class ComplexCircular(ComplexSpherical):
    """Simplicial complex on the surface of a 1-sphere
    cant really think of any applications, other than testing purposes

    """

    def subdivide(self):
        """1d simplicial has a natural correspondence to fundamental subdivision"""
        return self.subdivide_fundamental()


class ComplexSpherical2(ComplexSpherical):
    """Simplicial complex on the surface of a 2-sphere"""

    def __init__(self, vertices, simplices=None, topology=None, radius=1, weights=None):
        # FIXME: do we expect all vertices to have radius one, and do we enfroce this?
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

    def subdivide_loop(self):
        """Subdivide the complex, returning a refined complex where each edge inserts a vertex

        This is a loop-like subdivision

        """
        return self.copy(
            vertices=np.concatenate(self.primal_position[:2], axis=0),
            topology=self.topology.subdivide_loop()
        )

    def subdivide_loop_direct(self):
        """Subdivide the complex, returning a refined complex where each edge inserts a vertex

        This is a loop-like subdivision

        """
        return self.copy(
            vertices=np.concatenate(self.primal_position[:2], axis=0),
            topology=self.topology.subdivide_loop_direct()
        )

    def as_euclidian(self):
        from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
        return ComplexTriangularEuclidian3(vertices=self.vertices, topology=self.topology)

    @staticmethod
    def multigrid_transfer_dual(coarse, fine):
        """Construct dual transfer operator between spherical triangular complex
        and its loop subdivision

        Returns
        -------
        sparse matrix, [fine.n_vertices, coarse.n_vertices], float
            transfer matrix. each entry of the matrix represents the
            area overlap between fine and coarse dual cells

        Notes
        -----
        This operator is defined by the overlap between coarse and fine dual cells;
        computing those overlaps is the challenge addressed by this function

        The corner fine triangles of each coarse triangle are always trivial
        Crux is in the middle triangle, where fine and coarse dual vertex meet

        Can generalize this to euclidian complexes as well, by passing in the metric functions

        clrm are the center, left, right and mid of the fine central triangle (edge-midpoints of fine)
        CLRM are the center, left, right and mid of the coarse elements (vertices; edge-midpoints of coarse)
        where left-right is as judged looking from the picked coarse vertex down to the middle tri

        """

        assert fine.radius == coarse.radius
        assert fine.is_well_centered
        assert coarse.is_well_centered

        from pycomplex.geometry.spherical import triangle_area_from_corners, intersect_edges

        def coo_matrix(data, row, col):
            """construct a coo_matrix from data and index arrays"""
            return scipy.sparse.coo_matrix(
                (data.ravel(), (row.ravel(), col.ravel())),
                shape=(coarse.topology.n_elements[0], fine.topology.n_elements[0]))

        all_tris = np.arange(fine.topology.n_elements[2]).reshape(coarse.topology.n_elements[2], 4)
        # NOTE: this organization represents an assumption on relation between fine and coarse complexes
        central_tris = all_tris[:,0].flatten()
        corner_tris  = all_tris[:,1:].flatten()

        # for corner tris, we can simply work with fundamental domains
        # FIXME: is it not easier to define central triangle relative to these contributions too?
        domains = fine.topology.fundamental_domains()
        domains = domains[corner_tris].reshape(-1, 3)
        areas = triangle_area_from_corners(*[fine.primal_position[i][domains[:, i]] for i in range(3)])

        corner_transfer = coo_matrix(
            areas,
            # for every fine edge, decide what coarse dual cell it belongs to
            # then knowing the fine edge, we can decide on the dual cell
            coarse.pick_dual(fine.primal_position[1])[domains[:, 1]],
            domains[:, 0],
        )


        # now work on central triangle; this is the hard part
        i20 = fine.topology.incidence[2, 0]
        i21 = fine.topology.incidence[2, 1]
        I20 = coarse.topology.incidence[2, 0]
        I21 = coarse.topology.incidence[2, 1]

        # get the positions of the elements of the central fine triangle
        p0 = fine.primal_position[0][i20[central_tris]]
        p1 = fine.primal_position[1][i21[central_tris]]
        p2 = fine.primal_position[2][central_tris]
        P2 = coarse.primal_position[2]    # coarse dual vertex position

        # find the coarse dual cell the central fine dual vertex resides in
        # `m` is the column in our incidence arrays representing the middle
        a, m = np.where(I20 == coarse.pick_dual(p2)[:, None])
        assert np.array_equiv(a, np.arange(len(a))), "Fine central dual vertex not in any of the three coarse dual cells it is expected to be in"
        # compute columns left and right of the middle one
        # when referring to fine edge midpoints, L/R is reversed
        R = l = m - 1
        L = r = m - 2

        # compute the two intersection points
        pl = +intersect_edges(p0[a, l], P2, p1[a, L], p2)   # right fine edge is left coarse edge / left fine vertex
        pr = -intersect_edges(p0[a, r], P2, p1[a, R], p2)   # left fine edge if right coarse edge / right fine vertex

        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
            coarse.plot(ax=ax)
            fine.plot(ax=ax, primal_color='c', dual_color='m')
            ax.scatter(*pl[:, :2].T, c='y')
            ax.scatter(*pr[:, :2].T, c='g')
            plt.show()

        A = triangle_area_from_corners

        wedge_l = A(P2, p1[a, L], pl)
        wedge_r = A(P2, p1[a, R], pr)
        diamond_l = A(P2, pl, p2)
        diamond_r = A(P2, pr, p2)
        sliver_l = A(p0[a, l], pl, p2)
        sliver_r = A(p0[a, r], pr, p2)

        # assemble area contributions of the middle triangle
        areas = np.empty((len(a), 3, 3))     # coarsetris x coarsevert x finevert
        # the non-overlapping parts
        areas[a, R, l] = 0
        areas[a, L, r] = 0
        # triangular slivers disjoint from the m,m intersection
        areas[a, L, l] = A(p0[a, l], p1[a, L], pl)
        areas[a, R, r] = A(p0[a, r], p1[a, R], pr)
        # subset of coarse tri bounding sliver
        areas[a, L, m] = A(p0[a, m], p1[a, L], P2) + wedge_l
        areas[a, R, m] = A(p0[a, m], p1[a, R], P2) + wedge_r
        # subset of fine tri bounding sliver
        areas[a, m, l] = A(p0[a, l], p1[a, m], p2) + sliver_l
        areas[a, m, r] = A(p0[a, r], p1[a, m], p2) + sliver_r
        # square middle region; may compute as fine or coarse minus its flanking parts
        areas[a, m, m] = diamond_l + diamond_r

        assert(np.all(areas > -1e-13))

        # finally, we have the transfer matrix of the central fine triangles
        center_transfer = coo_matrix(
            areas,
            np.broadcast_to(I20[:           , :, None], areas.shape),
            np.broadcast_to(i20[central_tris, None, :], areas.shape),
        )

        return (center_transfer + corner_transfer).tocsr()

    @cached_property
    def coarse_to_fine(coarse):
        """Maps primal-0-form on coarse to primal-0-form on fine

        Returns
        -------
        sparse matrix, [fine.n_vertices, coarse.n_vertices], float
        """
        from pycomplex.sparse import normalize_l1
        return normalize_l1(coarse.multigrid_transfer_dual)
    @cached_property
    def fine_to_coarse(coarse):
        """Maps primal-0-form on fine to primal-0-form on coarse

        Returns
        -------
        sparse matrix, [coarse.n_vertices, fine.n_vertices], float
        """
        from pycomplex.sparse import normalize_l1
        return normalize_l1(coarse.multigrid_transfer_dual.T)


class ComplexSpherical3(ComplexSpherical):
    """Figuring out the metric computations for this is nontrivial.
    But doing physical simulations in a curved 3d space should be fun
    """
    pass