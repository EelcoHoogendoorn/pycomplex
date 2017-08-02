
import numpy as np
import numpy_indexed as npi
import scipy.spatial

from cached_property import cached_property

from pycomplex.complex.base import BaseComplexSpherical
from pycomplex.geometry import spherical
from pycomplex.math import linalg
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial
from pycomplex.topology import index_dtype


class ComplexSpherical(BaseComplexSpherical):
    """Complex on an n-sphere"""

    def __init__(self, vertices, simplices=None, topology=None, radius=1):
        self.vertices = np.asarray(vertices)
        if topology is None:
            topology = TopologySimplicial.from_simplices(simplices).fix_orientation()
            assert topology.is_oriented
        self.topology = topology
        self.radius = radius

    def plot(self, plot_dual=True, backface_culling=False, plot_vertices=True):
        """Visualize a complex on a 2-sphere; a little more involved than the other 2d cases"""
        import matplotlib.pyplot as plt
        import matplotlib.collections

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

        fig, ax = plt.subplots(1, 1)
        # plot outline of embedding space
        angles = np.linspace(0, 2*np.pi, 1000)
        ax.plot(np.cos(angles), np.sin(angles), color='k')

        # plot primal edges
        edges = self.topology.corners[1]
        steps = int(1000 / len(edges)) + 1
        e = subdivide(self.vertices[edges], steps=steps*2)
        plot_edge(ax, e, color='b', alpha=0.5)
        if plot_vertices:
            plot_vertex(ax, self.vertices, color='b')

        if plot_dual:
            # plot dual edges
            dual_vertices, dual_edges = self.dual_position[:2]
            dual_topology = self.topology.dual
            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:, 0], de[:, 1]
            s = subdivide(from_se(dual_edges, s), steps=steps)
            plot_edge(ax, s, color='r', alpha=0.5)
            e = subdivide(from_se(dual_edges, e), steps=steps)
            plot_edge(ax, e, color='r', alpha=0.5)
            if plot_vertices:
                plot_vertex(ax, dual_vertices, color='r')

        plt.axis('equal')
        plt.show()

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


class ComplexCircular(ComplexSpherical):
    """Simplicial complex on the surface of a 1-sphere; cant really think of any applications"""
    pass


class ComplexSpherical2(ComplexSpherical):
    """Simplicial complex on the surface of a 2-sphere"""

    def __init__(self, vertices, simplices=None, topology=None, radius=1):
        self.vertices = np.asarray(vertices)

        if topology is None:
            topology = TopologyTriangular.from_simplices(simplices)

        assert isinstance(topology, TopologyTriangular)
        self.topology = topology
        self.radius = radius

    @cached_property
    def metric(self):
        """Calc metric properties of a spherical complex

        Parameters
        ----------
        radius : float
            The radius of the n-sphere

        Notes
        -----
        This currently assumes triangle circumcenters are inside their triangles
        However, it should not be too hard to generalize it with signed metric calculations
        """
        def gather(idx, vals):
            """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
            return vals[idx]
        def scatter(idx, vals, target):
            """target[idx] += vals. """
            np.add.at(target.ravel(), idx.ravel(), vals.ravel())

        topology = self.topology
        PP = self.primal_position

        #metrics
        PN = topology.n_elements
        DN = PN[::-1]

        PM = [np.zeros(n) for n in PN]
        PM[0][...] = 1
        DM = [np.zeros(n) for n in DN]
        DM[0][...] = 1

        # precomputations
        E21  = topology.incidence[2, 1]  # [faces, e3]
        E10  = topology.incidence[1, 0]  # [edges, v2]
        E210 = E10[E21]                  # [faces, e3, v2]

        PP10  = PP[0][E10]                 # [edges, v2, c3]
        PP210 = PP10[E21]                  # [faces, e3, v2, c3]
        PP21  = PP[1][E21]                 # [faces, e3, c3] ; face-edge midpoints

        # calculate areas; devectorization over e makes things a little more elegant, by avoiding superfluous stacking
        for e in range(3):
            # this is the area of two fundamental domains
            # note that it is assumed here that the primal face center lies within the triangle
            # could we just compute a signed area and would it generalize?
            areas = spherical.triangle_area_from_corners(PP210[:,e,0,:], PP210[:,e,1,:], PP[2])
            PM[2] += areas                    # add contribution to primal face
            scatter(E210[:,e,0], areas/2, DM[2])
            scatter(E210[:,e,1], areas/2, DM[2])

        # calc edge lengths
        PM[1] += spherical.edge_length(PP10[:,0,:], PP10[:,1,:])
        for e in range(3):
            # note: this calc would need to be signed too, to support external circumcenters
            scatter(
                E21[:,e],
                spherical.edge_length(PP21[:,e,:], PP[2]),
                DM[1])

        return ([m * (self.radius ** i) for i, m in enumerate(PM)],
                [m * (self.radius ** i) for i, m in enumerate(DM)])

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
