"""Regular grid complexes"""

import numpy as np
import numpy_indexed as npi

import pycomplex.topology
from pycomplex.topology import index_dtype, sign_dtype
import pycomplex.topology.base
from pycomplex.complex.base import BaseComplexCubical
from pycomplex.topology.cubical import TopologyCubical


class ComplexCubical(BaseComplexCubical):
    """Regular complex with euclidian embedding"""

    def __init__(self, vertices, cubes=None, topology=None):
        """

        Parameters
        ----------
        vertices : ndarray, [n_vertices, n_dim], float
            vertex positions in euclidian space
        cubes : ndarray, [n_regulars, 2 ** n_dim], index_type, optional
        topology : RegularTopology object, optional
        """
        self.vertices = np.asarray(vertices)
        if topology is None:
            topology = TopologyCubical.from_cubes(cubes)
            self.topology = topology
            self.cubes = cubes
        if cubes is None:
            self.topology = topology
            self.cubes = topology.elements[-1]

    def subdivide(self, creases=None, smooth=False):
        """Catmull–Clark subdivision; n-d case
        Each n-cube leads to a new vertex on the subdivided grid

        Parameters
        ----------
        creases : dict of (int: ndarray), optional
            dict of n to n-chains, where nonzero elements denote crease elements
        smooth : bool
            if true, smoothing is performed
        """

        divided = type(self)(
            vertices=np.concatenate(self.primal_position(), axis=0),
            topology=self.topology.subdivide()
        )

        # propagate creases to lower level
        if creases is not None:
            creases = {n: divided.topology.transfer_matrices[n] * c
                       for n, c in creases.items()}

        if smooth:
            divided = divided.smooth(creases)
        return divided

    def boundary(self):
        # FIXME: need to implement vertex subset selection! this only works if vertex-set remains stable
        return ComplexCubical(vertices=self.vertices, topology=self.topology.boundary())

    def product(self, other):
        """Construct the product of two regular complexes

        Parameters
        ----------
        self : ComplexCubical
        other ; RegularComplex

        Returns
        -------
        RegularComplex of dimension self.n_dim + other.n_dim
        """
        if not self.n_dim == self.topology.n_dim:
            raise ValueError
        if not other.n_dim == other.topology.n_dim:
            raise ValueError
        j, i = np.indices((len(other.vertices), len(self.vertices)))
        return ComplexCubical(
            vertices=np.concatenate([
                    self.vertices[i.flatten()],
                    other.vertices[j.flatten()]
                ],
                axis=1
            ),
            topology=self.topology.product(other.topology)
        )

    # cast to subtypes
    def as_12(self):
        if not (self.n_dim == 2 and self.topology.n_dim == 1):
            raise TypeError('invalid cast')
        return ComplexCubical1Euclidian2(vertices=self.vertices, topology=self.topology)

    def as_22(self):
        if not (self.n_dim == 2 and self.topology.n_dim == 2):
            raise TypeError('invalid cast')
        return ComplexCubical2Euclidian2(vertices=self.vertices, topology=self.topology)

    def as_23(self):
        if not (self.n_dim == 3 and self.topology.n_dim == 2):
            raise TypeError('invalid cast')
        return ComplexCubical2Euclidian3(vertices=self.vertices, topology=self.topology)

    def as_33(self):
        if not (self.n_dim == 3 and self.topology.n_dim == 3):
            raise TypeError('invalid cast')
        return ComplexCubical3Euclidian3(vertices=self.vertices, topology=self.topology)


class ComplexCubical2(ComplexCubical):
    """Specialization for 2d plotting"""

    def to_simplicial(self):
        """Convert the regular complex into a simplicial complex,
        by forming 4 tris from each quad and its dual position"""
        Q20 = self.topology.elements[2]
        n_e = self.topology.n_elements

        T20 = -np.ones((n_e[2], 4, 3), dtype=index_dtype)
        T20[:, 0, :2] = Q20[:, 0, ::+1]
        T20[:, 1, :2] = Q20[:, 1, ::-1]
        T20[:, 2, :2] = Q20[:, ::-1, 0]
        T20[:, 3, :2] = Q20[:, ::+1, 1]
        T20[:, :, 2] = np.arange(n_e[2], dtype=index_dtype)[:, None] + n_e[0]   # all tris connect to a vertex inserted at each triangle
        # FIXME: should not be restricted to 3d embedding space
        from pycomplex.complex.simplicial import ComplexTriangular
        # FIXME: construct a mapping from 0-forms in self to 0-forms in the returned topology?
        # FIXME: likely super useful when using this for rendering
        # can make for a legitimate step in subdivision too when combined with smoothing step
        # FIXME: split topological part
        # FIXME: add transfer operators here too
        v, e, f = self.primal_position()
        return ComplexTriangular(
            vertices=np.concatenate([v, f], axis=0),
            triangles=T20.reshape(-1, 3)
        )

    def metric(self):
        """
        calc metric properties and hodges for a 2d regular complex
        sum over subdomains.
        should be relatively easy to generalize to n-dimensions
        """
        from pycomplex.geometry import cubical

        def gather(idx, vals):
            """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
            return vals[idx]
        def scatter(idx, vals, target):
            """target[idx] += vals. """
            np.add.at(target.ravel(), idx.ravel(), vals.ravel())

        topology = self.topology
        dual_vertices, dual_edges, dual_faces = self.dual_position()
        primal_vertices, primal_edges, primal_faces = self.primal_position()

        #metrics
        P0, P1, P2 = topology.n_vertices, topology.n_edges, topology.n_faces
        D0, D1, D2 = P2, P1, P0
        MP0 = np.ones (P0)
        MP1 = np.zeros(P1)
        MP2 = np.zeros(P2)
        MD0 = np.ones (D0)
        MD1 = np.zeros(D1)
        MD2 = np.zeros(D2)

        #precomputations
        E21 = topology.elements[2, 1]     # [faces, e3]
        E10 = topology.elements[1, 0]     # [edges, v2]
        E10P  = self.vertices[E10] # [edges, v2, c3]
        E210P = E10P[E21]          # [faces, e3, v2, c3]
        FEM  = (E210P.mean(axis=2))  # face-edge midpoints; [faces, e3, c3]
        FEV  = E10[E21] # [faces, e3, v2]

        # calc areas of fundamental squares
        for d1 in range(2):
            for d2 in range(2):
                # this is the area of one fundamental domain
                # note that it is assumed here that the primal face center lies within the triangle
                # could we just compute a signed area and would it generalize?
                areas = cubical.area_from_corners(E210P[:, e, 0, :], E210P[:, e, 1, :], primal_faces)
                MP2 += areas                    # add contribution to primal face
                scatter(FEV[:,e,0], areas/2, MD2)

        # calc edge lengths
        MP1 += cubical.edge_length(E10P[:, 0, :], E10P[:, 1, :])
        for e in range(3):
            scatter(
                E21[:,e],
                cubical.edge_length(FEM[:, e, :], primal_faces),
                MD1)

        self.primal_metric = [MP0, MP1, MP2]
        self.dual_metric = [MD0, MD1, MD2]


class ComplexCubical2Euclidian2(ComplexCubical2):
    def plot(self, plot_dual=True):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        fig, ax = plt.subplots(1,1)
        lc = matplotlib.collections.LineCollection(e, color='b', alpha=0.5)
        ax.add_collection(lc)
        ax.scatter(*self.vertices.T, color='b')

        if plot_dual:
            # plot dual cells
            dual_vertices, dual_edges = self.dual_position()[0:2]
            dual_topology = self.topology.dual()
            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:,0], de[:,1]
            s = np.moveaxis(np.array([dual_edges, s]), 0, 1)
            lc = matplotlib.collections.LineCollection(s, color='r', alpha=0.5)
            ax.add_collection(lc)
            e = np.moveaxis(np.array([dual_edges, e]), 0, 1)
            lc = matplotlib.collections.LineCollection(e, color='r', alpha=0.5)
            ax.add_collection(lc)

            ax.scatter(*dual_vertices.T, color='r')

        plt.axis('equal')
        plt.show()


class ComplexCubical2Euclidian3(ComplexCubical2):
    """2 dimesional topology with 3d euclidian embedding"""

    def plot(self, plot_dual=True):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        fig, ax = plt.subplots(1,1)
        lc = matplotlib.collections.LineCollection(e[..., :2], color='b', alpha=0.5)
        ax.add_collection(lc)
        ax.scatter(*self.vertices.T[:2], color='b')

        # plot dual cells
        if plot_dual:
            dual_vertices, dual_edges = self.dual_position()[0:2]
            dual_topology = self.topology.dual()
            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:,0], de[:,1]
            s = np.moveaxis(np.array([dual_edges, s]), 0, 1)
            lc = matplotlib.collections.LineCollection(s[...,:2], color='r', alpha=0.5)
            ax.add_collection(lc)
            e = np.moveaxis(np.array([dual_edges, e]), 0, 1)
            lc = matplotlib.collections.LineCollection(e[...,:2], color='r', alpha=0.5)
            ax.add_collection(lc)

            ax.scatter(*dual_vertices.T[:2], color='r')

        plt.axis('equal')
        plt.show()


class ComplexCubical1Euclidian2(ComplexCubical):
    """Line in 2d euclidian space"""

    def plot(self, plot_vertices=True):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        fig, ax = plt.subplots(1,1)
        lc = matplotlib.collections.LineCollection(e, color='b', alpha=0.5)
        ax.add_collection(lc)
        if plot_vertices:
            ax.scatter(*self.vertices.T, color='b')

        plt.axis('equal')
        plt.show()


class ComplexCubical3Euclidian3(ComplexCubical):
    """Cubes in 3d euclidian space"""

    def plot(self, plot_dual=False):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        fig, ax = plt.subplots(1,1)
        lc = matplotlib.collections.LineCollection(e[..., :2], color='b', alpha=0.5)
        ax.add_collection(lc)
        ax.scatter(*self.vertices.T[:2], color='b')

        if plot_dual:
            # plot dual cells
            dual_vertices, dual_edges = self.dual_position()[0:2]
            dual_topology = self.topology.dual()
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

            ax.scatter(*dual_vertices.T[:2], color='r')

        plt.axis('equal')
        plt.show()
