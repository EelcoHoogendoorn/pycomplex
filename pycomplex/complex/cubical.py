"""Regular grid complexes"""

import numpy as np
import scipy.sparse
from cached_property import cached_property

from pycomplex.complex.base import BaseComplex
from pycomplex.topology.cubical import TopologyCubical


class ComplexCubical(BaseComplex):
    """Regular complex with euclidian embedding"""

    def __init__(self, vertices, cubes=None, topology=None, parent=None):
        """

        Parameters
        ----------
        vertices : ndarray, [n_vertices, n_dim], float
            vertex positions in euclidian space
        cubes : ndarray, [n_cubes, 2 ** n_dim], index_type, optional
            if topology is not supplied
        topology : TopologyCubical object, optional
            if cubes is not supplied
        parent : BaseComplex, optional
            reference to the complex this one is derived from; usually a subdivision relation
        """
        self.parent = parent
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
        fine.parent = coarse
        return fine

    def subdivide_operator(coarse, creases=None, smooth=False):
        """By constructing this in operator form, rather than subdividing directly,
        we can cache the expensive parts of this calculation,
        and achieve very fast updates to our subdivision curves under change of vertex position

        Parameters
        ----------
        creases : dict of (int: ndarray), optional
            dict of n-chains, where nonzero elements denote crease elements
        smooth : bool
            if true, smoothing is performed after subdivision

        Returns
        -------
        operator : sparse array, [coarse.n_vertices, fine.n_vertices]
            sparse array mapping coarse to fine vertex positions

        """
        # relationship between coarse and fine vertices; without smoothing could not be simpler
        coarse_averaging = scipy.sparse.vstack(coarse.topology.averaging_operators_0)

        if smooth:
            # NOTE: only difference with triangular case lies in this call
            fine = coarse.subdivide_cubical()

            # propagate creases to lower level
            if creases is not None:
                creases = {n: fine.topology.transfer_matrices[n] * c
                           for n, c in creases.items()}

            operator = fine.smooth_operator(creases) * coarse_averaging

        else:
            operator = coarse_averaging

        return operator

    @cached_property
    def multigrid_transfers(self):
        """Multigrid transfer operators between the complex and its parent

        Returns
        -------
        List[scipy.sparse[fine.n_elements, coarse.n_elements]]
            n-th element in the list relates fine and coarse primal n-elements

        Notes
        -----
        These are geometric multigrid transfers; their coefficients are derived from
        smoothness of the k-form under the k-laplacian.
        If this is appropriate or not is problem-specific

        """
        # FIXME: is this really a property of the complex? maybe make it a free function in multigrid module
        # FIXME: This deals with primary elements only so far; can smoothing-based logic be extended to dual boundary as well?
        # FIXME: seems like dual would work along same principles. add boundary block to transfer matrix, and decide on desired number of smoothing steps
        # FIXME: pass in custom smoother? or even equation object itself?
        # FIXME: how to normalize? divide by metric, then normalize sum to one, and multiply with metric?
        # FIXME: what about vector forms? absolute sum? remember; goal is to have exact mapping etween (near) null spaces. but constructing these is hard
        # FIXME: note that transfer depends on metric; which may include vertex weights
        # does it generalize to simplicial? I do think so. seems like only the element-order logic needs specialization
        # also, the smoother used should include metric information then

        fine = self
        coarse = self.parent

        # compute the order and index of each fine cubes parent cube
        order, parent = coarse.topology.subdivide_cubical_relations(fine.topology)

        # these encode only direct ancestry relationships
        # could use these directly as coarsening operator, without the smoothing
        TT = fine.topology.transfer_matrices

        def laplacian(self, k):
            """Construct laplace-beltrami of order k on primal k-forms"""
            M = [0] + self.topology.matrices + [0]
            L, R = M[k:k + 2]
            LT, RT = np.transpose(L), np.transpose(R)
            return LT * L + R * RT

        def smoother(i):
            L = laplacian(fine, i)
            # seems to converge to 8 for all k for ndim=2
            l = scipy.sparse.linalg.eigsh((L * 1.0), k=1, which='LM', tol=1e-6, return_eigenvectors=False)[0]
            I = scipy.sparse.identity(L.shape[0])
            # construct smoother
            # FIXME: add mass term? not needed for cube; but at boundary?
            # cleaner to use the laplacian equation class here; another argument for moving this out of the complex
            # http://amath.colorado.edu/pub/multigrid/aSAm.pdf
            # suggests 4/3/lamda * L as a prolongation smoother
            r = 1 # 4/3
            S = I - (r/l) * L
            return S

        def smoothers(S, pre=0):
            # lazily compute powers of S
            SA = TM.T   # init with mapping onto coarse; reduce useless computation
            for _ in range(pre):
                SA = SA * S
            while True:
                yield SA
                SA = SA * S

        result = []
        for i, (TM, o, p) in enumerate(zip(TT, order, parent)):

            # element-order; what order of coarse cube a fine cube was inserted on
            eo = o.reshape(len(o), -1).min(axis=-1)
            # create selection matrices, to pick out fine cubes of a given order
            Select = (scipy.sparse.diags((eo == i) * 1) for i in np.unique(eo))

            # add these smoother matrices together again
            R = sum(smoother * select for select, smoother in zip(Select, smoothers(smoother(i))))

            result.append(R.T)
        return result

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
        # if not self.n_dim == self.topology.n_dim:
        #     raise ValueError
        # if not other.n_dim == other.topology.n_dim:
        #     raise ValueError
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

    def plot(self, plot_dual=True, plot_vertices=False, plot_arrow=False, ax=None, primal_color='b', dual_color='r', **kwargs):
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
        if plot_arrow:
            for edge in e[..., :2]:
                ax.arrow(*edge[0], *(edge[1]-edge[0]),
                         head_width=0.05, head_length=-0.1, fc='k', ec='k')

        # plot dual cells
        if plot_dual:
            dual_vertices, dual_edges = self.dual_position[0:2]
            dual_topology = self.topology.dual
            from pycomplex.topology.util import sparse_to_elements
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

        ax.axis('equal')


class ComplexCubical1(ComplexCubical):
    """Specialization for 1d lines"""

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
        from pycomplex.geometry.euclidian import segment_normals
        edges = self.vertices[self.topology.elements[1]]
        normals = segment_normals(edges)
        centroids = edges.mean(axis=1)
        return (normals * centroids).sum() / self.n_dim

    def plot(self, plot_vertices=True, plot_mean=False, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        edges = self.topology.elements[1]
        e = self.vertices[edges]

        lc = matplotlib.collections.LineCollection(e, **kwargs)
        ax.add_collection(lc)
        if plot_vertices:
            ax.scatter(*self.vertices.T, **kwargs)
        ax.axis('equal')

        from matplotlib.collections import PatchCollection
        if plot_mean:
            ax.add_collection(PatchCollection([plt.Circle(self.vertices.mean(axis=0, keepdims=True).T, 0.1)], alpha=0.95))


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
