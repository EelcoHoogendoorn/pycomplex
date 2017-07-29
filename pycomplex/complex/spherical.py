import numpy as np

from pycomplex.complex.base import BaseComplexSpherical
from pycomplex.geometry import spherical
from pycomplex.math import linalg
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial
from pycomplex.topology import index_dtype


class ComplexSpherical(BaseComplexSpherical):
    """Complex on an n-sphere"""

    def __init__(self, vertices, simplices=None, topology=None):
        self.vertices = np.asarray(vertices)
        if topology is None:
            topology = TopologySimplicial.from_simplices(simplices).fix_orientation()
            assert topology.is_oriented
        self.topology = topology

    def plot(self, plot_dual=True, backface_culling=False):
        """Visualize a complex on a 2-sphere; a little more involved than the other 2d cases"""
        import matplotlib.pyplot as plt
        import matplotlib.collections

        def from_se(s, e):
            return np.concatenate([s[:, None, :], e[:, None, :]], axis=1)

        def subdivide(edges, steps=10):
            f = np.linspace(0, 1, steps)
            i = np.array([f, 1-f])
            edges = edges[:, :, None, :] * i[None, :, :, None]
            edges = edges.sum(axis=1)
            s = edges[:, :-1, None, :]
            e = edges[:, +1:, None, :]
            edges = np.concatenate([s, e], axis=2)
            edges = edges.reshape(-1, 2, edges.shape[-1])
            return linalg.normalized(edges)

        def plot_edges(ax, lines, **kwargs):
            if backface_culling:
                z = lines[..., 2]
                drawn = (z > 0).all(axis=1)
                lines = lines[drawn]
            lc = matplotlib.collections.LineCollection(lines[..., :2], **kwargs)
            ax.add_collection(lc)

        def plot_vertices(ax, points, **kwargs):
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
        e = subdivide(self.vertices[edges], steps=20)
        plot_edges(ax, e, color='b', alpha=0.5)
        plot_vertices(ax, self.vertices, color='b')

        if plot_dual:
            # plot dual edges
            dual_vertices, dual_edges = self.dual_position()[:2]
            dual_topology = self.topology.dual
            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:, 0], de[:, 1]
            s = subdivide(from_se(dual_edges, s))
            plot_edges(ax, s, color='r', alpha=0.5)
            e = subdivide(from_se(dual_edges, e))
            plot_edges(ax, e, color='r', alpha=0.5)

            plot_vertices(ax, dual_vertices, color='r')

        plt.axis('equal')
        plt.show()

    def as_2(self):
        return ComplexSpherical2(vertices=self.vertices, topology=self.topology.as_2())

class ComplexCircular(ComplexSpherical):
    """Simplicial complex on the surface of a 1-sphere; cant really think of any applications"""
    pass



class ComplexSpherical2(ComplexSpherical):
    """Simplicial complex on the surface of a 2-sphere"""

    def __init__(self, vertices, simplices=None, topology=None):
        self.vertices = np.asarray(vertices)

        if topology is None:
            topology = TopologyTriangular.from_simplices(simplices)

        self.topology = topology

    def metric(self):
        """Calc metric properties of a spherical complex

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
        PP0, PP1, PP2 = self.primal_position()

        #metrics
        P0, P1, P2 = topology.n_elements
        D0, D1, D2 = P2, P1, P0
        MP0 = np.ones (P0)
        MP1 = np.zeros(P1)
        MP2 = np.zeros(P2)
        MD0 = np.ones (D0)
        MD1 = np.zeros(D1)
        MD2 = np.zeros(D2)

        # precomputations
        E21  = topology.incidence[2, 1]  # [faces, e3]
        E10  = topology.incidence[1, 0]  # [edges, v2]
        E210 = E10[E21]                  # [faces, e3, v2]

        PP10  = PP0[E10]                 # [edges, v2, c3]
        PP210 = PP10[E21]                # [faces, e3, v2, c3]
        PP21  = PP1[E21]                 # [faces, e3, c3] ; face-edge midpoints

        # calculate areas; devectorization over e makes things a little more elegant, by avoiding superfluous stacking
        for e in range(3):
            # this is the area of two fundamental domains
            # note that it is assumed here that the primal face center lies within the triangle
            # could we just compute a signed area and would it generalize?
            areas = spherical.triangle_area_from_corners(PP210[:,e,0,:], PP210[:,e,1,:], PP2)
            MP2 += areas                    # add contribution to primal face
            scatter(E210[:,e,0], areas/2, MD2)
            scatter(E210[:,e,1], areas/2, MD2)

        # calc edge lengths
        MP1 += spherical.edge_length(PP10[:,0,:], PP10[:,1,:])
        for e in range(3):
            # note: this calc would need to be signed too, to support external circumcenters
            scatter(
                E21[:,e],
                spherical.edge_length(PP21[:,e,:], PP2),
                MD1)

        self.primal_metric = [MP0, MP1, MP2]
        self.dual_metric = [MD0, MD1, MD2]

        self.hodge_from_metric()

    def hodge_from_metric(self):
        MP = self.primal_metric
        MD = self.dual_metric
        #hodge operators
        self.D2P0 = MD[2] / MP[0]
        self.P0D2 = MP[0] / MD[2]

        self.D1P1 = MD[1] / MP[1]
        self.P1D1 = MP[1] / MD[1]

        self.D0P2 = MD[0] / MP[2]
        self.P2D0 = MP[2] / MD[0]

    def subdivide(self):
        """Subdivide the complex, returning a refined complex where each edge inserts a vertex

        This is a loop-like subdivision

        """
        pp = self.primal_position()
        return ComplexSpherical2(
            vertices=np.concatenate([pp[0], pp[1]], axis=0),
            topology=self.topology.subdivide()
        )

    def as_euclidian(self):
        from pycomplex.complex.simplicial import ComplexTriangularEuclidian3
        return ComplexTriangularEuclidian3(vertices=self.vertices, topology=self.topology)
