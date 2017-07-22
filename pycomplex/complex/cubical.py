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
        cubes : ndarray, [n_cubes, 2 ** n_dim], index_type, optional
        topology : TopologyCubical object, optional
        """
        self.vertices = np.asarray(vertices)
        if topology is None:
            topology = TopologyCubical.from_cubes(cubes)
            self.topology = topology
        if cubes is None:
            self.topology = topology

    def subdivide(coarse, creases=None, smooth=False):
        """Catmull–Clark subdivision; n-d case
        Each n-cube leads to a new vertex on the subdivided grid

        Parameters
        ----------
        creases : dict of (int: ndarray), optional
            dict of n to n-chains, where nonzero elements denote crease elements
        smooth : bool
            if true, smoothing is performed after subdivision
        """

        fine = type(coarse)(
            vertices=np.concatenate(coarse.primal_position(), axis=0),    # every n-element spawns a new vertex
            topology=coarse.topology.subdivide()
        )

        # propagate creases to lower level
        if creases is not None:
            creases = {n: fine.topology.transfer_matrices[n] * c
                       for n, c in creases.items()}

        if smooth:
            fine = fine.smooth(creases)
        return fine

    def product(self, other):
        """Construct the product of two cubical complexes

        Parameters
        ----------
        self : ComplexCubical
        other : ComplexCubical

        Returns
        -------
        ComplexCubical of dimension self.n_dim + other.n_dim
        """
        if not self.n_dim == self.topology.n_dim:
            raise ValueError
        if not other.n_dim == other.topology.n_dim:
            raise ValueError
        # these vertex indices need to agree with the conventions employed in the topological product
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
    """Specialization for 2d quads"""

    def to_simplicial(self):
        """Convert the cubical complex into a simplicial complex,
        by forming 4 tris from each quad and its dual position"""
        v, e, f = self.primal_position()
        from pycomplex.complex.simplicial import ComplexTriangular  # Triangular and Quadrilateral or Simplical2 and Cubical2; pick one...
        return ComplexTriangular(
            vertices=np.concatenate([v, f], axis=0),
            topology=self.topology.as_2().to_simplicial()
        )

    def to_simplicial_transfer_0(self, c0):
        """map 0 form from cubical to simplical"""
        s0 = np.concatenate([c0, c0[self.topology.corners[2]].mean(axis=1)])
        return s0

    def to_simplicial_transfer_2(self, c2):
        """map 0 form from cubical to simplical"""
        s0 = np.repeat(c2, 4)
        return s0

class ComplexCubical2Euclidian2(ComplexCubical2):

    def as_regular(self):
        from pycomplex.complex.regular import ComplexRegular2
        return ComplexRegular2(vertices=self.vertices, topology=self.topology)
    def plot(self, plot_dual=True, plot_vertices=True):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        fig, ax = plt.subplots(1,1)
        lc = matplotlib.collections.LineCollection(e, color='b', alpha=0.5)
        ax.add_collection(lc)
        if plot_vertices:
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
            if plot_vertices:
                ax.scatter(*dual_vertices.T, color='r')

        plt.axis('equal')
        plt.show()


class ComplexCubical2Euclidian3(ComplexCubical2):
    """2 dimensional topology (quadrilateral) with 3d euclidian embedding"""

    def plot(self, plot_dual=True, plot_vertices=False):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        fig, ax = plt.subplots(1,1)
        lc = matplotlib.collections.LineCollection(e[..., :2], color='b', alpha=0.5)
        ax.add_collection(lc)
        if plot_vertices:
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

            if plot_vertices:
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
    """3-Cubes in 3d euclidian space"""

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

    def plot_slice(self, affine, ):
        pass
