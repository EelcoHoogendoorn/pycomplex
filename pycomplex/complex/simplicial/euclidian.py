
import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.math import linalg
from pycomplex.geometry import euclidian
from pycomplex.topology.simplicial import TopologyTriangular, TopologySimplicial

from pycomplex.complex.simplicial.base import BaseComplexSimplicial


class ComplexSimplicialEuclidian(BaseComplexSimplicial):
    """Simplicial complex embedded in euclidian space"""

    def __init__(self, vertices, simplices=None, topology=None, weights=None):
        self.vertices = np.asarray(vertices)
        self.weights = weights
        if topology is None:
            topology = TopologySimplicial.from_simplices(simplices)
        self.topology = topology

    def homogenize(self, points):
        return np.concatenate([points, np.ones_like(points[..., 0:1])], axis=-1)

    def plot(self, ax=None, plot_dual=True, plot_vertices=True, plot_lines=True, plot_arrow=False, primal_color='b', dual_color='r'):
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
        if plot_arrow:
            for edge in e[..., :2]:
                ax.arrow(*edge[0], *(edge[1]-edge[0]),
                         head_width=0.05, head_length=-0.1, fc='k', ec='k')



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
        ax.axis('equal')

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

        ax.axis('equal')

    def as_spherical(self):
        from pycomplex.complex.simplicial.spherical import ComplexSpherical
        return ComplexSpherical(vertices=self.vertices, topology=self.topology, weights=self.weights)

    def as_2(self):
        return ComplexTriangularEuclidian(
            vertices=self.vertices, topology=self.topology.as_2(), weights=self.weights)

    def unsigned_volume(self, pts):
        from pycomplex.geometry import euclidian
        return euclidian.unsigned_volume(pts)

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
        return [np.einsum('...cn,...c->...n', self.vertices[c], b)
                for c, b in zip(self.topology.corners, self.primal_barycentric)]

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


class ComplexTriangularEuclidian(ComplexSimplicialEuclidian):
    """Triangular simplicial complex in euclidian space"""

    def __init__(self, vertices, triangles=None, topology=None, weights=None):
        self.vertices = np.asarray(vertices)
        self.weights = weights
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
        # this may be a feature rather than a bug, however
        # FIXME: the below would work, if it were a summing rather than averaging operator. can be constructed by replacing data with zeros?
        # vertex_areas = self.topology.averaging_operators_N[0] * self.compute_triangle_areas
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

    # @cached_property
    # def metric(self):
    #     # FIXME: implement edge metrics; should not be hard
    #     # FIXME: implement true circumcentric-dual vertex areas
    #     PM = [self.topology.chain(0, fill=1), None, self.compute_triangle_areas]
    #     DM = [self.topology.chain(2, fill=1), None, 1 / self.compute_vertex_areas]
    #     return PM, DM
    #
    # @cached_property
    # def hodge_DP(self):
    #     """Triangular complex overloads these for the time being"""
    #     return [self.compute_vertex_areas, self.compute_edge_ratio, 1 / self.compute_triangle_areas]
    # @cached_property
    # def hodge_PD(self):
    #     return [1 / h for h in self.hodge_DP]

    def subdivide_loop(coarse, smooth=False, creases=None):
        """Loop subdivision

        Parameters
        ----------
        smooth : bool
            if False, newly created vertices are linearly interpolated from their parent
            if True, mesh is smoothed after subdivision
        creases : dict of (int, ndarray), optional
            dict of n-chains, where nonzero elements denote crease elements

        Returns
        -------
        fine : type(coarse)
        """
        fine = type(coarse)(
            vertices=np.concatenate(coarse.primal_position[:2], axis=0),
            topology=coarse.topology.subdivide_loop()
        )

        # propagate creases to lower level
        if creases is not None:
            creases = {n: fine.topology.transfer_matrices[n] * c
                       for n, c in creases.items()}

        if smooth:
            fine = fine.smooth(creases)

        return fine

    def subdivide_loop_operator(coarse, smooth=False, creases=None):
        """By constructing this in operator form, rather than subdividing directly,
        we can cache the expensive parts of this calculation,
        and achieve very fast updates to our subdivision curves under change of vertex position

        Parameters
        ----------
        creases : dict of (int: ndarray), optional
            dict of n to n-chains, where nonzero elements denote crease elements
        smooth : bool
            if true, smoothing is performed after subdivision

        Returns
        -------
        operator : sparse array, [coarse.n_vertices, fine.n_vertices]
            sparse array mapping coarse to fine vertices

        """
        coarse_averaging = scipy.sparse.vstack(coarse.topology.averaging_operators_0)

        if smooth:
            # NOTE: only difference with cubical case lies in this call
            fine = coarse.subdivide_loop()

            # propagate creases to lower level
            if creases is not None:
                creases = {n: fine.topology.transfer_matrices[n] * c
                           for n, c in creases.items()}

            operator = fine.smooth_operator(creases) * coarse_averaging

        else:
            operator = coarse_averaging

        return operator


    def subdivide_cubical(self):
        """Convert the simplicial-2 complex into a cubical-2 complex"""
        from pycomplex.complex.cubical import ComplexCubical2
        return ComplexCubical2(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_cubical()
        )

    def as_2(self):
        assert self.n_dim == 2
        return ComplexTriangularEuclidian2(vertices=self.vertices, topology=self.topology, weights=self.weights)

    def as_3(self):
        assert self.n_dim == 3
        return ComplexTriangularEuclidian3(vertices=self.vertices, topology=self.topology, weights=self.weights)

    def plot_dual_0_form_interpolated(self, d0, ax=None, weighted=False, **kwargs):
        if weighted:
            average = self.weighted_average_operators
        else:
            average = self.topology.dual.averaging_operators_0
        S = self.topology.dual.selector
        sub_form = np.concatenate([s * a * d0 for s, a in zip(S, average[::-1])], axis=0)
        sub = self.subdivide_fundamental()
        sub.plot_primal_0_form(sub_form, ax=ax, **kwargs)


class ComplexTriangularEuclidian2(ComplexTriangularEuclidian):
    """Triangular topology embedded in euclidian 2-space"""

    def plot_primal_0_form(self, c0, ax=None, plot_contour=True, cmap='viridis', **kwargs):
        """plot a primal 0-form

        Parameters
        ----------
        c0 : ndarray, [n_vertices], float
            a primal 0-form

        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        triang = tri.Triangulation(*self.vertices[:, :2].T, triangles=self.topology.triangles)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if plot_contour:
            levels = np.linspace(c0.min()-1e-6, c0.max()+1e-6, kwargs.get('levels', 20), endpoint=True)
            if cmap:
                ax.tricontourf(triang, c0, cmap=cmap, levels=levels)
            c = ax.tricontour(triang, c0, colors='k', levels=levels)
            # contour = c.allsegs[0][0]  # this is how a single contour can be extracted
        else:
            ax.tripcolor(triang, c0, cmap=cmap, **kwargs)

        ax.axis('equal')

    def plot_primal_2_form(self, p2, ax=None, cmap='jet'):
        """plot a primal 2-form

        Parameters
        ----------
        p2 : ndarray, [n_vertices], float
            a primal 0-form

        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        cmap = plt.get_cmap(cmap)
        facecolors = ScalarMappable(cmap=cmap).to_rgba(p2)
        ax.add_collection(PolyCollection(self.vertices[self.topology.triangles], facecolors=facecolors, edgecolors=None))

        ax.axis('equal')


class ComplexTriangularEuclidian3(ComplexTriangularEuclidian):
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
        # FIXME: this is not a circumcentric area! wil do for now... we have the circumcentric area now but not happy with it yet..
        I20 = self.topology.incidence[2, 0]
        t_a = np.linalg.norm(self.triangle_normals(), axis=1)
        vertex_idx, v_a = npi.group_by(I20.flatten()).sum(np.repeat(t_a, 3, axis=0))
        return v_a / 3

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
        """Compute total surface area"""
        normals = self.triangle_normals()
        return np.linalg.norm(normals, axis=1).sum() / 2

    def plot_3d(self, ax=None, plot_dual=True, plot_vertices=True, backface_culling=True, plot_lines=True, primal_color='b', dual_color='r'):
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

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if plot_lines:
            edge_positions = self.vertices[self.topology.edges[edges]]
            lc = matplotlib.collections.LineCollection(edge_positions[..., :2], color=primal_color, alpha=0.5)
            ax.add_collection(lc)
        if plot_vertices:
            vertex_positions = self.vertices[vertices]
            ax.scatter(*vertex_positions.T[:2], color=primal_color)

        if plot_dual:
            dual_vertices, dual_edges = self.dual_position[0:2]
            dual_topology = self.topology.dual

            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)
            fe = sparse_to_elements(dual_topology[0])
            fe = fe[visible]
            dual_vertices = dual_vertices[visible]

            if plot_lines:
                # plot dual edges on a per-face basis; good for backface culling, not so much for boundaries
                for c in fe.T:
                    e = np.concatenate([dual_vertices[:, None], dual_edges[c][:, None]], axis=1)
                    lc = matplotlib.collections.LineCollection(e[..., :2], color=dual_color, alpha=0.5)
                    ax.add_collection(lc)

            if plot_vertices:
                ax.scatter(*dual_vertices.T[:2], color=dual_color)

        ax.axis('equal')

    def plot_primal_0_form(self, c0, ax=None, backface_culling=True, flip_normals=False, plot_contour=True, cmap='viridis', **kwargs):
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
            normals = -self.triangle_normals() if flip_normals else self.triangle_normals()
            visible = normals[:, 2] > 0
        else:
            visible = None

        triang = tri.Triangulation(*self.vertices[:, :2].T, triangles=self.topology.triangles, mask=visible)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if plot_contour:
            ax.tricontourf(triang, c0, cmap=cmap, **kwargs)
            ax.tricontour(triang, c0, colors='k')
        else:
            ax.tripcolor(triang, c0, cmap=cmap, **kwargs)

        ax.autoscale(tight=True)
        ax.axis('equal')

    def plot_primal_2_form(self, p2, ax=None, backface_culling=True):
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

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        cmap = plt.get_cmap('jet')
        facecolors = ScalarMappable(cmap=cmap).to_rgba(p2)
        ax.add_collection(PolyCollection(tris[:, :, :2], facecolors=facecolors[visible], edgecolors=None))

        ax.autoscale(tight=True)
        ax.axis('equal')

    def as_spherical(self):
        radius = np.linalg.norm(self.vertices, axis=1)
        assert np.allclose(radius, radius.mean())
        from pycomplex.complex.simplicial.spherical import ComplexSpherical2
        return ComplexSpherical2(vertices=self.vertices, topology=self.topology)

    def save_STL(self, filename):
        """Save a mesh to binary STL.

        Parameters
        ----------
        filename : str
        """
        header      = np.zeros(80, '<c')
        triangles   = np.array(self.topology.n_elements[2], '<u4')
        dtype       = [('normal', '<f4', 3,), ('vertex', '<f4', (3, 3)), ('abc', '<u2', 1,)]
        data        = np.empty(triangles, dtype)

        data['abc']    = 0     #standard stl cruft; can store two bytes per triangle, but not really used for anything
        data['vertex'] = self.vertices[self.topology.corners[2]]
        data['normal'] = linalg.normalized(self.triangle_normals()) # normal also isnt really used

        with open(filename, 'wb') as fh:
            header.   tofile(fh)
            triangles.tofile(fh)
            data.     tofile(fh)

    @classmethod
    def load_STL(cls, filename):
        """Load an STL file from disk

        Parameters
        ----------
        filename : str

        Returns
        -------
        cls
        """
        dtype = [('normal', '<f4', 3,), ('vertex', '<f4', (3, 3)), ('abc', '<u2', 1,)]

        with open(filename, 'rb') as fh:
            header = np.fromfile(fh, '<c', 80)
            triangles = np.fromfile(fh, '<u4', 1)[0]
            data = np.fromfile(fh, dtype, triangles)

        vertices, triangles = npi.unique(data['vertex'].reshape(-1, 3), return_inverse=True)
        return cls(vertices, triangles=triangles.reshape(-1, 3))

    def extrude(self, other):
        """Extrude simplicial complex from self to other, where other is a displaced copy of self
        Boundary is stitched with a triangle strip

        Parameters
        ----------
        other : type(self)

        Returns
        -------
        solid : type(self)
        """
        assert self.topology is other.topology
        assert self.topology.is_oriented
        assert self.boundary

        n_vertices = len(self.vertices)
        boundary_edges = self.topology.incidence[1, 0][self.boundary.topology.parent_idx[1]]
        boundary_tris = np.concatenate((
            np.concatenate((boundary_edges[:, ::-1], boundary_edges[:, 0:1] + n_vertices), axis=1),
            np.concatenate((boundary_edges[:, ::+1] + n_vertices, boundary_edges[:, 1:2]), axis=1)
        ))

        self_tris = self.topology.incidence[2, 0]
        other_tris = self.topology.incidence[2, 0][:, ::-1] + n_vertices

        solid_points = np.concatenate((self.vertices, other.vertices))
        solid_tris   = np.concatenate((self_tris, other_tris, boundary_tris))

        solid = self.copy(vertices=solid_points, triangles=solid_tris, topology=None)
        assert solid.topology.is_oriented
        return solid
