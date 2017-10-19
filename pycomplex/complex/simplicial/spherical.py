
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

    def as_spherical(self):
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
        return type(self)(
            vertices=np.concatenate(self.primal_position[:2], axis=0),
            topology=self.topology.subdivide_loop()
        )

    def subdivide_loop_direct(self):
        """Subdivide the complex, returning a refined complex where each edge inserts a vertex

        This is a loop-like subdivision

        """
        return type(self)(
            vertices=np.concatenate(self.primal_position[:2], axis=0),
            topology=self.topology.subdivide_loop_direct()
        )

    def as_euclidian(self):
        from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
        return ComplexTriangularEuclidian3(vertices=self.vertices, topology=self.topology)

    @staticmethod
    def multigrid_transfer_d2(coarse, fine):
        """Construct multigrid transfer operator between spherical triangular complex
        and its direct loop subdivision

        Returns
        -------
        sparse matrix, [fine.n_vertices, coarse.n_vertices], float
            transfer operator

        Notes
        -----
        Corners of each fine triangle are always trivial
        Crux is in the middle triangle, where fine and coarse dual vertex meet
        We can query coarse dual vertices with fine dual vertices of central triangles to figure out the conditional

        """
        from pycomplex.geometry.spherical import triangle_area_from_corners, triangle_area_from_normals
        def triangle_areas_around_center(center, corners):
            """given a triangle formed by corners, and its dual point center,
            compute spherical area of the voronoi faces

            Parameters
            ----------
            center : ndarray, [..., 3], float
            corners : ndarray, [..., 3, 3], float

            Returns
            -------
            areas : ndarray, [..., 3], float
                spherical area opposite to each corner
            """
            areas = np.empty(corners.shape[:-1])
            for i in range(3):
                areas[:, :, i] = triangle_area_from_corners(center, corners[:, :, i - 2], corners[:, :, i - 1])
            # swivel equilaterals to vonoroi parts
            return (areas.sum(axis=2)[:, :, None] - areas) / 2

        def gather(idx, vals):
            return vals[idx]
        def coo_matrix(data, row, col):
            """construct a coo_matrix from data and index arrays"""
            return scipy.sparse.coo_matrix(
                (data.ravel(),(row.ravel(), col.ravel())),
                shape=(coarse.topology.n_elements[0], fine.topology.n_elements[0]))

        all_tris = np.arange(fine.topology.n_elements[2]).reshape(coarse.topology.n_elements[2], 4)
        central_tris = all_tris[:,0]
        corner_tris  = all_tris[:,1:]
        #first, compute contribution to transfer matrices from the central refined triangle

        coarse_dual   = coarse.primal_position[2]
        fine_dual     = fine.primal_position[2][central_tris]
        face_edge_mid = gather(fine.topology.elements[-1][0::4], fine.primal_position[0])

        fine_edge_normal = [np.cross(face_edge_mid[:,i-2,:], face_edge_mid[:,i-1,:]) for i in range(3)]
        fine_edge_mid    = [(face_edge_mid[:,i-2,:] + face_edge_mid[:,i-1,:])/2      for i in range(3)]
        fine_edge_dual   = [np.cross(fine_edge_mid[i], fine_edge_normal[i])          for i in range(3)]
        fine_edge_normal = np.array(fine_edge_normal)
        fine_edge_mid    = np.array(fine_edge_mid)
        fine_edge_dual   = np.array(fine_edge_dual)

        coarse_areas     = [triangle_area_from_corners(coarse_dual, face_edge_mid[:,i-2,:], face_edge_mid[:,i-1,:]) for i in range(3)]
        fine_areas       = [triangle_area_from_corners(fine_dual  , face_edge_mid[:,i-2,:], face_edge_mid[:,i-1,:]) for i in range(3)]
        fine_areas       = [(fine_areas[i-2]+fine_areas[i-1])/2 for i in range(3)]
        coarse_areas     = np.array(coarse_areas)
        fine_areas       = np.array(fine_areas)

        #normal of edge midpoints to coarse dual
        interior_normal = np.array([np.cross(face_edge_mid[:,i,:], coarse_dual) for i in range(3)])

        #the 0-3 index of the overlapping domains
        #biggest of the subtris formed with the coarse dual vertex seems to work; but cant prove why it is so...
        touching = np.argmax(coarse_areas, axis=0)

        # indexing arrays
        I = np.arange(len(touching))
        m = touching        # middle pair
        l = touching-1      # left-rotated pair
        r = touching-2      # right-rotated pair

        #compute sliver triangles
        sliver_r = triangle_area_from_normals(
            +fine_edge_normal[l, I],
            +fine_edge_dual  [l, I],
            +interior_normal [r, I])
        sliver_l = triangle_area_from_normals(
            +fine_edge_normal[r, I],
            -fine_edge_dual  [r, I],
            -interior_normal [l, I])


        assert(np.all(sliver_l > -1e-10))
        assert(np.all(sliver_r > -1e-10))


        # assemble area contributions of the middle triangle
        areas = np.empty((len(touching), 3, 3))     #coarsetris x coarsevert x finevert
        # the non-overlapping parts
        areas[I,l,l] = 0
        areas[I,r,r] = 0
        # triangular slivers disjoint from the m,m intersection
        areas[I,r,l] = sliver_l
        areas[I,l,r] = sliver_r
        # subset of coarse tri bounding sliver
        areas[I,r,m] = coarse_areas[r,I] - sliver_l
        areas[I,l,m] = coarse_areas[l,I] - sliver_r
        # subset of fine tri bounding sliver
        areas[I,m,l] = fine_areas[l,I] - sliver_l
        areas[I,m,r] = fine_areas[r,I] - sliver_r
        # square middle region; may compute as fine or coarse minus its flanking parts
        areas[I,m,m] = coarse_areas[m,I] - areas[I,m,l] - areas[I,m,r]

        # we may get numerical negativity for 2x2x2 symmetry, with equilateral fundemantal domain,
        # or high subdivision levels. or is error at high subdivision due to failing of touching logic?
        assert(np.all(areas > -1e-10))

        # areas maps between coarse vertices and fine edge vertices.
        # add mapping for coarse to fine vertices too

        # need to grab coarsetri x 3coarsevert x 3finevert arrays of coarse and fine vertices
        fine_vertex   = np.repeat( fine  .topology.elements[-1][0::4, None,    :], 3, axis=1)
        coarse_vertex = np.repeat( coarse.topology.elements[-1][:   , :   , None], 3, axis=2)


        center_transfer = coo_matrix(areas, coarse_vertex, fine_vertex)


        # add corner triangle contributions; this is relatively easy
        # coarsetri x 3coarsevert x 3finevert
        corner_vertex = gather(corner_tris, fine.topology.elements[-1])
        corner_dual   = gather(corner_tris, fine.dual)
        corner_primal = gather(corner_vertex, fine.primal)

        # coarsetri x 3coarsevert x 3finevert
        corner_areas    = triangle_areas_around_center(corner_dual, corner_primal)
        # construct matrix
        corner_transfer = coo_matrix(corner_areas, coarse_vertex, corner_vertex)
        return (center_transfer + corner_transfer).tocsr()

    @cached_property
    def stuff(self):
        # calc normalizations
        self.coarse_area = self.transfer   * np.ones(fine  .topology.D2)
        self.fine_area   = self.transfer.T * np.ones(coarse.topology.D2)

        self.f = np.sqrt( self.fine_area)[:,None]
        self.c = np.sqrt( self.coarse_area)[:,None]

        # test for consistency with metric calculations
        assert(np.allclose(self.coarse_area, coarse.D2P0, 1e-10))
        assert(np.allclose(self.fine_area  , fine  .D2P0, 1e-10))



class ComplexSpherical3(ComplexSpherical):
    """Figuring out the metric computations for this will be a hard one.
    But doing physical simulations in a curved space should be fun
    """
    pass