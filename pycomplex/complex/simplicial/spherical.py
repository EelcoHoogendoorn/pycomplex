
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
        return ComplexSpherical2(
            vertices=np.concatenate(self.primal_position[:2], axis=0),
            topology=self.topology.subdivide_loop()
        )

    def as_euclidian(self):
        from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
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