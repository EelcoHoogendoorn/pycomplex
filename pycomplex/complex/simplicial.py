"""Triangular (2d simplicial) complex embedded in euclidian space"""

import numpy as np
import numpy_indexed as npi
from cached_property import cached_property

from pycomplex.complex.base import BaseComplexEuclidian
from pycomplex.geometry import euclidian
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial
from pycomplex.topology import index_dtype, sign_dtype


class ComplexSimplicial(BaseComplexEuclidian):

    def __init__(self, vertices, simplices=None, topology=None):
        self.vertices = np.asarray(vertices)
        if topology is None:
            topology = TopologySimplicial.from_simplices(simplices)
        self.topology = topology

    def plot(self, ax, plot_dual=True, plot_vertices=True, plot_lines=True):
        """Plot projection on plane"""
        import matplotlib.pyplot as plt
        import matplotlib.collections
        edges = self.topology.elements[1]
        e = self.vertices[edges]

        # fig, ax = plt.subplots(1, 1)
        if plot_lines:
            lc = matplotlib.collections.LineCollection(e[..., :2], color='b', alpha=0.5)
            ax.add_collection(lc)
        if plot_vertices:
            ax.scatter(*self.vertices.T[:2], color='b')

        if plot_dual:
            dual_vertices, dual_edges = self.dual_position[0:2]
            if plot_lines:
                dual_topology = self.topology.dual
                from pycomplex.topology import sparse_to_elements
                de = sparse_to_elements(dual_topology[0].T)

                de = dual_vertices[de]
                s, e = de[:, 0], de[:, 1]
                s = np.moveaxis(np.array([dual_edges, s]), 0, 1)
                lc = matplotlib.collections.LineCollection(s[..., :2], color='r', alpha=0.5)
                ax.add_collection(lc)
                e = np.moveaxis(np.array([dual_edges, e]), 0, 1)
                lc = matplotlib.collections.LineCollection(e[..., :2], color='r', alpha=0.5)
                ax.add_collection(lc)

            if plot_vertices:
                ax.scatter(*dual_vertices.T[:2], color='r')
        plt.axis('equal')
        # plt.show()

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
        # plt.show()

    def as_spherical(self):
        from pycomplex.complex.spherical import ComplexSpherical
        return ComplexSpherical(vertices=self.vertices, topology=self.topology)

    def as_2(self):
        return ComplexTriangular(vertices=self.vertices, topology=self.topology.as_2())


class ComplexTriangular(ComplexSimplicial):
    """Triangular simplicial complex"""

    def __init__(self, vertices, triangles=None, topology=None):
        self.vertices = np.asarray(vertices)
        if topology is None:
            topology = TopologyTriangular.from_simplices(triangles)
        self.topology = topology

    @cached_property
    def compute_triangle_angles(self):
        """Compute interior angles for each triangle-vertex

        Returns
        -------
        ndarray, [n_triangles, 3], float, radians
            interior angle of each vertex of each triangle
            the ith vertex-angle is the angle opposite from the ith edge of the face
        """
        return euclidian.triangle_angles(self.vertices[self.topology.triangles])

    @cached_property
    def compute_vertex_areas(self):
        # FIXME: this corresponds to barycentric dual, not circumcentric!
        _, vertex_areas = npi.group_by(self.topology.triangles.flatten()).sum(np.repeat(self.compute_triangle_areas, 3))
        return vertex_areas / 3

    @cached_property
    def compute_edge_ratio(self):
        """Relates primal 1-forms to their dual 1-forms

        Compute edge hodge based on cotan formula;
        This can be shown to be equivalent to a hodge defined as the ratio of the length of primal and dual edges,
        where the dual mesh is the circumcentric dual. The nice thing is however that in this form we dont actually
        need to construct the geometry of the dual mesh explicitly, which is a major complexity-win

        Returns
        -------
        ndarray, [n_edges], float
            ratio of primal/dual metric for each edge

        References
        ----------
        http://ddg.cs.columbia.edu/SGP2014/LaplaceBeltrami.pdf
        """
        cotan = 1. / np.tan(self.compute_triangle_angles)
        # sum the contribution for all faces incident to each edge
        B21 = self.topology._boundary[1]
        _, hodge = npi.group_by(B21.flatten()).sum(cotan.flatten())
        return hodge / 2

    @cached_property
    def compute_triangle_areas(self):
        """Compute area associated with each face

        Returns
        -------
        triangle_area : ndarray, [n_triangles], float
            vertex area per vertex
        """
        return euclidian.unsigned_volume(self.vertices[self.topology.triangles])

    @cached_property
    def metric(self):
        # FIXME: implement edge metrics; should not be hard
        # FIXME: implement true circumcentric-dual vertex areas
        PM = [self.topology.chain(0, fill=1), None, self.compute_triangle_areas]
        DM = [self.topology.chain(2, fill=1), None, 1 / self.compute_vertex_areas]
        return PM, DM

    @cached_property
    def hodge_DP(self):
        """Triangular complex overloads these for the time being"""
        return [self.compute_vertex_areas, self.compute_edge_ratio, 1 / self.compute_triangle_areas]
    @cached_property
    def hodge_PD(self):
        return [1 / h for h in self.hodge_DP]

    def subdivide(coarse, smooth=False, creases=None):
        """Loop subdivision

        """
        fine = type(coarse)(
            vertices=np.concatenate(coarse.primal_position[:2], axis=0),
            topology=coarse.topology.subdivide()
        )

        # propagate creases to lower level
        if creases is not None:
            creases = {n: fine.topology.transfer_matrices[n] * c
                       for n, c in creases.items()}

        if smooth:
            fine = fine.smooth(creases)

        return fine

    def to_cubical(self):
        """Convert the simplicial complex into a cubical complex,
        by forming 3 quads from each triangle and its dual position"""
        barycenters = [self.vertices[c].mean(axis=1) for c in self.topology.corners]

        from pycomplex.complex.cubical import ComplexCubical2
        return ComplexCubical2(
            vertices=np.concatenate(barycenters, axis=0),
            topology=self.topology.to_cubical()
        )

    def as_2(self):
        return ComplexTriangularEuclidian2(vertices=self.vertices, topology=self.topology)

    def as_3(self):
        return ComplexTriangularEuclidian3(vertices=self.vertices, topology=self.topology)


class ComplexTriangularEuclidian2(ComplexTriangular):
    """Triangular topology embedded in euclidian 2-space"""

    def plot_primal_0_form(self, c0, plot_contour=True, cmap='viridis', **kwargs):
        """plot a primal 0-form

        Parameters
        ----------
        c0 : ndarray, [n_vertices], float
            a primal 0-form

        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        triang = tri.Triangulation(*self.vertices[:, :2].T, triangles=self.topology.triangles)

        fig, ax = plt.subplots(1, 1)

        if plot_contour:
            levels = np.linspace(c0.min(), c0.max(), kwargs.get('levels', 20), endpoint=True)
            plt.tricontourf(triang, c0, cmap=cmap, levels=levels)
            plt.tricontour(triang, c0, colors='k', levels=levels)
        else:
            plt.tripcolor(triang, c0, cmap=cmap, **kwargs)

        plt.axis('equal')

    def plot_primal_2_form(self, p2, cmap='jet'):
        """plot a primal 2-form

        Parameters
        ----------
        p2 : ndarray, [n_vertices], float
            a primal 0-form

        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.cm import ScalarMappable

        fig, ax = plt.subplots(1, 1)
        cmap = plt.get_cmap(cmap)
        facecolors = ScalarMappable(cmap=cmap).to_rgba(p2)
        ax.add_collection(PolyCollection(self.vertices[self.topology.triangles], facecolors=facecolors, edgecolors=None))

        plt.axis('equal')


class ComplexTriangularEuclidian3(ComplexTriangular):
    """Triangular topology embedded in euclidian 3-space"""

    def triangle_normals(self):
        """Compute non-normalized triangle normals

        Returns
        -------
        ndarray, [n_triangles, 3], float
            normals
        """
        corners = self.vertices[self.topology.triangles]
        edges = np.diff(corners, axis=1)
        return np.cross(edges[:, 0], edges[:, 1])

    def triangle_centroids(self):
        return self.vertices[self.topology.corners[2]].mean(axis=1)

    def vertex_normals(self):
        """Compute non-normalized vertex normals

        Returns
        -------
        ndarray, [n_vertices, 3], float
            normals

        Notes
        -----
        This corresponds to the volume-gradient of the complex
        """
        t_n = self.triangle_normals()
        I20 = self.topology.incidence[2, 0]
        vertex_idx, v_n = npi.group_by(I20.flatten()).sum(np.repeat(t_n, 3, axis=0))
        return v_n

    def vertex_areas(self):
        """Compute circumcentric area here"""
        # FIXME: this is not a circumcentric area! wil do for now...
        I20 = self.topology.incidence[2, 0]
        t_a = np.linalg.norm(self.triangle_normals(), axis=1)
        vertex_idx, v_a = npi.group_by(I20.flatten()).sum(np.repeat(t_a, 3, axis=0))
        return v_a / 3
        raise NotImplementedError()

    def edge_lengths(self):
        """Compute primal edge lengths

        Returns
        -------
        ndarray, [n_edges], float
        """
        grad = self.topology.matrices[0].T
        return np.linalg.norm(grad * self.vertices, axis=1)

    def volume(self):
        """Return the volume enclosed by this complex

        Returns
        -------
        float
            The signed enclosed volume

        Raises
        ------
        ValueError
            If the manifold is not closed
        """
        if not self.topology.is_closed:
            raise ValueError('Computing volume requires a closed manifold')
        normals = self.triangle_normals()
        centroids = self.triangle_centroids()
        return (normals * centroids).sum() / self.n_dim

    def area(self):
        normals = self.triangle_normals()
        return np.linalg.norm(normals, axis=1).sum() / 2

    def plot_3d(self, backface_culling=True, plot_dual=True, plot_vertices=True):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.incidence[2, 1]
        vertices = self.topology.incidence[2, 0]
        if backface_culling:
            visible = self.triangle_normals()[:, 2] > 0
        else:
            visible = Ellipsis

        edges = edges[visible]
        vertices = vertices[visible]


        edges = np.unique(edges)
        vertices = np.unique(vertices)

        edge_positions = self.vertices[self.topology.edges[edges]]

        fig, ax = plt.subplots(1, 1)
        lc = matplotlib.collections.LineCollection(edge_positions[..., :2], color='b', alpha=0.5)
        ax.add_collection(lc)

        vertex_positions = self.vertices[vertices]
        if plot_vertices:
            ax.scatter(*vertex_positions.T[:2], color='b')

        if plot_dual:
            dual_vertices, dual_edges = self.dual_position[0:2]
            dual_topology = self.topology.dual

            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)
            fe = sparse_to_elements(dual_topology[0])
            fe = fe[visible]
            dual_vertices = dual_vertices[visible]

            # plot dual edges on a per-face basis; good for backface culling, not so much for boundaries
            for c in fe.T:
                e = np.concatenate([dual_vertices[:, None], dual_edges[c][:, None]], axis=1)
                lc = matplotlib.collections.LineCollection(e[..., :2], color='r', alpha=0.5)
                ax.add_collection(lc)

            if plot_vertices:
                ax.scatter(*dual_vertices.T[:2], color='r')

        plt.axis('equal')
        plt.show()

    def plot_primal_0_form(self, c0, backface_culling=True, plot_contour=True, cmap='viridis', **kwargs):
        """plot a primal 0-form

        Parameters
        ----------
        c0 : ndarray, [n_vertices], float
            a primal 0-form

        """
        # FIXME: add color mapping, including banded stripes
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        if backface_culling:
            visible = self.triangle_normals()[:, 2] > 0
        else:
            visible = Ellipsis

        triang = tri.Triangulation(*self.vertices[:, :2].T, triangles=self.topology.triangles, mask=visible)

        fig, ax = plt.subplots(1, 1)
        if plot_contour:
            plt.tricontourf(triang, c0, cmap=cmap, **kwargs)
            plt.tricontour(triang, c0, colors='k')
        else:
            plt.tripcolor(triang, c0, cmap=cmap, **kwargs)

        plt.axis('equal')
        # plt.show()

    def plot_primal_2_form(self, p2, backface_culling=True):
        """plot a primal 0-form

        Parameters
        ----------
        p2 : ndarray, [n_vertices], float
            a primal 2-form

        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.cm import ScalarMappable

        if backface_culling:
            visible = self.triangle_normals()[:, 2] > 0
        else:
            visible = Ellipsis

        tris = self.vertices[self.topology.triangles[visible]]

        fig, ax = plt.subplots(1, 1)
        cmap = plt.get_cmap('jet')
        facecolors = ScalarMappable(cmap=cmap).to_rgba(p2)
        ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=facecolors[visible], edgecolors=None))

        plt.axis('equal')
        plt.show()

    def as_spherical(self):
        return ComplexTriangular(vertices=self.vertices, topology=self.topology)
