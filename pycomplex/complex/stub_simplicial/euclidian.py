
import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.geometry import euclidian
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial

from pycomplex.complex.stub_simplicial.base import BaseComplexSimplicial


class ComplexSimplicialEuclidian(BaseComplexSimplicial):

    def plot(self, ax=None, plot_dual=True, plot_vertices=True, plot_lines=True, primal_color='b', dual_color='r'):
        """Plot projection on plane"""
        import matplotlib.pyplot as plt
        import matplotlib.collections
        edges = self.topology.elements[1]
        e = self.vertices[edges]

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if plot_lines:
            lc = matplotlib.collections.LineCollection(e[..., :2], color=primal_color, alpha=0.5)
            ax.add_collection(lc)
        if plot_vertices:
            ax.scatter(*self.vertices.T[:2], color=primal_color)

        if plot_dual:
            dual_vertices, dual_edges = self.dual_position[0:2]
            if plot_lines:
                dual_topology = self.topology.dual
                from pycomplex.topology import sparse_to_elements
                de = sparse_to_elements(dual_topology[0].T)

                de = dual_vertices[de]
                s, e = de[:, 0], de[:, 1]
                s = np.moveaxis(np.array([dual_edges, s]), 0, 1)
                lc = matplotlib.collections.LineCollection(s[..., :2], color=dual_color, alpha=0.5)
                ax.add_collection(lc)
                e = np.moveaxis(np.array([dual_edges, e]), 0, 1)
                lc = matplotlib.collections.LineCollection(e[..., :2], color=dual_color, alpha=0.5)
                ax.add_collection(lc)

            if plot_vertices:
                ax.scatter(*dual_vertices.T[:2], color=dual_color)
        plt.axis('equal')

    def plot_domains(self, ax):
        """Plot projection of fundamental domain onto plane"""
        import matplotlib.pyplot as plt
        import matplotlib.collections
        domain = self.topology.fundamental_domains()
        PP = self.primal_position

        # fig, ax = plt.subplots(1, 1)
        from pycomplex.math.combinatorial import combinations
        comb = combinations(list(range(self.topology.n_dim + 1)), 2)
        for (p, pair), c in zip(comb, 'rmbkyc'):
            i, j = pair
            # if i==1 or j ==1: continue
            s = PP[i][domain[..., i].flatten()[:]]
            e = PP[j][domain[..., j].flatten()[:]]
            edge = np.moveaxis(np.array([s, e]), 0, 1)

            lc = matplotlib.collections.LineCollection(edge[..., :2], color=c, alpha=0.5)
            ax.add_collection(lc)

        plt.axis('equal')

    def as_spherical(self):
        from pycomplex.complex.spherical import ComplexSpherical
        return ComplexSpherical(vertices=self.vertices, topology=self.topology)

    def as_2(self):
        return ComplexTriangularEuclidian(
            vertices=self.vertices, topology=self.topology.as_2(), weights=self.weights)

    def subdivide_fundamental(self, oriented=True):
        return type(self)(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_fundamental(oriented)
        )

    def subdivide_fundamental_transfer(self):
        return

    def subdivide_simplicial(self):
        PP = self.primal_position
        return type(self)(
            vertices=np.concatenate([PP[0], PP[-1]], axis=0),
            topology=self.topology.subdivide_simplicial()
        )

    def subdivide_cubical(self):
        """Subdivide the simplicial complex into a cubical complex"""
        from pycomplex.complex.cubical import ComplexCubical
        return ComplexCubical(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_cubical()
        )

    @cached_property
    def metric(self):
        """Compute metrics from fundamental domain contributions

        Notes
        -----
        according to [1], seems we can only have guaranteed non-negativity of vertex duals
        if the mesh is pairwise-delaunay; but it need not be strictly well-centered!
        quad subdivision surface to simplicial leads to negative vertex duals already.
        Optimization of hodges may make this problem go away; but not always it seems.
        Seems like there is still a use case for the simple barycentric dual vertex volumes

        References
        ----------
        [1] http://www.math.uiuc.edu/~hirani/papers/HiKaVa2013_CAD.pdf
        """
        topology = self.topology
        assert topology.is_oriented
        PP = self.primal_position
        domains = self.topology.fundamental_domains()

        corners = self.vertices[self.topology.corners[-1]]
        mean = corners.mean(axis=-2, keepdims=True)  # centering
        corners = corners - mean
        bary = euclidian.circumcenter_barycentric(corners)
        signs = np.sign(bary)   # if the face opposing this vert is inverted wrt the circumcenter
        signs = (signs.T * np.ones_like(domains[..., 0]).T).T.reshape(-1) # duplicate signs to all fundamental domains corresponding to each face

        domains = domains.reshape(-1, domains.shape[-1])
        corners = np.concatenate([p[d][:, None, :] for p, d in zip(PP, domains.T)], axis=1)

        PN = topology.n_elements
        DN = PN[::-1]

        # metrics
        PM = [np.zeros(n) for n in PN]
        PM[0][...] = 1
        DM = [np.zeros(n) for n in DN]
        DM[0][...] = 1

        unsigned = euclidian.unsigned_volume
        from scipy.misc import factorial
        groups = [npi.group_by(c) for c in domains.T]   # cache groupings since they can be reused

        for i in range(1, self.topology.n_dim):
            n = i + 1
            d = self.topology.n_dim - i
            PM[i] = groups[i].mean(unsigned(corners[:, :n]))[1] * factorial(n)  # FIXME: primal can be signed too! tri of tet for instance. need to work out signs logic better
            DM[i] = groups[d].sum (unsigned(corners[:, d:]) * signs)[1] / factorial(d+1)

        V = euclidian.unsigned_volume(corners) * signs
        PM[-1] = groups[-1].sum(V)[1]
        DM[-1] = groups[+0].sum(V)[1]

        return PM, DM
