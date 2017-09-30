"""Regular grid complexes"""

import numpy as np
import scipy.sparse
from cached_property import cached_property

from pycomplex.complex.base import BaseComplex
from pycomplex.topology.cubical import TopologyCubical


class ComplexCubical(BaseComplex):
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

    @cached_property
    def primal_position(self):
        """positions of all primal elements

        Returns
        -------
        list of primal element positions, length n_dim
        """
        return [self.vertices[c].mean(axis=1) for c in self.topology.corners]

    def subdivide_cubical(coarse, creases=None, smooth=False):
        """Cubical subdivision; n-d case
        Each n-cube in the coarse complex leads to a new vertex in the refined complex

        Parameters
        ----------
        creases : dict of (int: ndarray), optional
            dict of n to n-chains, where nonzero elements denote crease elements
        smooth : bool
            if true, smoothing is performed after subdivision

        """

        fine = type(coarse)(
            vertices=np.concatenate(coarse.primal_position, axis=0),    # every n-element spawns a new vertex
            topology=coarse.topology.subdivide_cubical()
        )

        # propagate creases to lower level
        if creases is not None:
            creases = {n: fine.topology.transfer_matrices[n] * c
                       for n, c in creases.items()}

        if smooth:
            fine = fine.smooth(creases)

        # FIXME: implement subdivide_transfer for cubes
        return fine

    def subdivide_operator(coarse, creases=None, smooth=False):
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

        Notes
        -----
        How to construct subdivision matrix?
        pure subdivision without smoothing:
            Just stack the topology.averaging_operators of the coarse meshes,
            to map coarse vertices to unsmooth fine vertices
        to add smoothing:
            add topology.averaging_operators of fine; mapping fine verts to centroids of fine n-elements
            multiplied by transpose; average centroids back to new vert positions
            transposed matrix product needs diagonal crease selector matrix inbetween

        """
        coarse_averaging = scipy.sparse.vstack(coarse.topology.averaging_operators_0)

        if smooth:
            fine = coarse.subdivide_cubical()

            # propagate creases to lower level
            if creases is not None:
                creases = {n: fine.topology.transfer_matrices[n] * c
                           for n, c in creases.items()}

            operator = fine.smooth_operator(creases) * coarse_averaging

        else:
            operator = coarse_averaging

        return operator

    def subdivide_fundamental(self):
        from pycomplex.complex.simplicial.euclidian import ComplexSimplicialEuclidian
        return ComplexSimplicialEuclidian(
            vertices=np.concatenate(self.primal_position, axis=0),
            topology=self.topology.subdivide_fundamental()
        )

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
        # FIXME: add transfer operators here too?
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
    def as_11(self):
        if not (self.n_dim == 1 and self.topology.n_dim == 1):
            raise TypeError('invalid cast')
        return ComplexCubical1Euclidian1(vertices=self.vertices, topology=self.topology)

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

    def as_44(self):
        if not (self.n_dim == 4 and self.topology.n_dim == 4):
            raise TypeError('invalid cast')
        return ComplexCubical4Euclidian4(vertices=self.vertices, topology=self.topology)

    def plot(self, plot_dual=True, plot_vertices=False, ax=None, primal_color='b', dual_color='r'):
        """Generic 2d projected plotting of primal and dual lines and edges"""
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        lc = matplotlib.collections.LineCollection(e[..., :2], color=primal_color, alpha=0.5)
        ax.add_collection(lc)
        if plot_vertices:
            ax.scatter(*self.vertices.T[:2], color=primal_color)

        # plot dual cells
        if plot_dual:
            dual_vertices, dual_edges = self.dual_position[0:2]
            dual_topology = self.topology.dual
            from pycomplex.topology import sparse_to_elements
            de = sparse_to_elements(dual_topology[0].T)

            de = dual_vertices[de]
            s, e = de[:,0], de[:,1]
            s = np.moveaxis(np.array([dual_edges, s]), 0, 1)
            lc = matplotlib.collections.LineCollection(s[...,:2], color=dual_color, alpha=0.5)
            ax.add_collection(lc)
            e = np.moveaxis(np.array([dual_edges, e]), 0, 1)
            lc = matplotlib.collections.LineCollection(e[...,:2], color=dual_color, alpha=0.5)
            ax.add_collection(lc)

            if plot_vertices:
                ax.scatter(*dual_vertices.T[:2], color=dual_color)

        plt.axis('equal')



class ComplexCubical1(ComplexCubical):
    """Specialization for 2d quads"""

    def to_simplicial(self):
        """Convert the cubical complex into a simplicial complex; trivial"""
        from pycomplex.complex.simplicial.euclidian import ComplexSimplicial1
        return ComplexSimplicial1(
            vertices=self.vertices,
            topology=self.topology.as_1().subdivide_simplicial()
        )


class ComplexCubical2(ComplexCubical):
    """Specialization for 2d quads"""

    def subdivide_simplicial(self):
        """Convert the cubical complex into a simplicial complex,
        by forming 4 tris from each quad and its dual position"""
        from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian  # Triangular and Quadrilateral or Simplical2 and Cubical2; pick one...
        return ComplexTriangularEuclidian(
            vertices=np.concatenate(self.primal_position[::2], axis=0),
            topology=self.topology.as_2().subdivide_simplicial()
        )


class ComplexCubical1Euclidian1(ComplexCubical1):

    def as_regular(self):
        from pycomplex.complex.regular import ComplexRegular1
        return ComplexRegular1(vertices=self.vertices, topology=self.topology)


class ComplexCubical2Euclidian2(ComplexCubical2):

    def as_regular(self):
        from pycomplex.complex.regular import ComplexRegular2
        return ComplexRegular2(vertices=self.vertices, topology=self.topology)


class ComplexCubical2Euclidian3(ComplexCubical2):
    """2 dimensional topology (quadrilateral) with 3d euclidian embedding"""


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

    def plot_slice(self, affine, ):
        pass

    def as_regular(self):
        from pycomplex.complex.regular import ComplexRegular3
        return ComplexRegular3(vertices=self.vertices, topology=self.topology)


class ComplexCubical4Euclidian4(ComplexCubical):
    """No use yet whatsoever, but nice to test on"""

    def as_regular(self):
        from pycomplex.complex.regular import ComplexRegular4
        return ComplexRegular4(vertices=self.vertices, topology=self.topology)


class ComplexCubicalToroidal(ComplexCubical):
    """Cubical complex with a toroidal topology, and corresponding metric"""