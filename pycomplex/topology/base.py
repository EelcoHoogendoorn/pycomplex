from abc import abstractmethod
import pycosat

import scipy
import scipy.sparse

import numpy as np
import numpy_indexed as npi
import scipy
from cached_property import cached_property

from pycomplex.topology import ManifoldException, topology_matrix
from pycomplex.topology import sign_dtype, index_dtype, sparse_to_elements


class BaseTopology(object):
    """Stuff common to all topology objects in all dimensions

    An n-dimensional topology is defined by a sequence of n (sparse) topology matrices, T(n)
    T(n) defines the n-elements in terms of an oriented closed boundary of n-1 elements

    """

    def __init__(self, elements, boundary, orientation, element_labels=None, strict=False):
        """

        Parameters
        ----------
        elements : list of ndarrays
            element representation in terms of vertex indices; details depend on subtype
            n-th entry describes n-elements in terms of the indices of the 0-elements
            elements of the list are of type index_dtype
        boundary : list of ndarrays
            list of boundary index information
            n-th entry describes n+1-elements in terms of the indices of its n-element boundary
            elements of the list are of type index_dtype
        orientation : list of ndarrays
            list of orientation sign information
            n-th entry describes n+1-elements in terms of the relative orientation of its n-element boundary
            elements of the list are of type sign_dtype
        element_labels : labels associated with each topological element
        strict : bool, optional
            if True, strict checking for manifoldness is performed
        """
        self.strict = strict
        self._elements = elements
        self._orientation = orientation
        self._boundary = boundary
        self.n_dim = len(orientation)

        assert(len(boundary) == len(orientation))
        assert(len(elements) == self.n_dim + 1)
        for b, o in zip(boundary, orientation):
            if not b.shape == o.shape:
                raise ValueError

    @cached_property
    def elements(self):
        return self._elements

    @cached_property
    def corners(self):
        return [e.reshape(n, -1) for e, n in zip(self._elements, self.n_elements)]

    @cached_property
    def n_elements(self):
        return [len(e) for e in self._elements]

    @cached_property
    def incidence(self):
        """Incidence relations between topological elements"""
        I = np.zeros((self.n_dim + 1, self.n_dim + 1), dtype=np.object)
        for i, e in enumerate(self.elements):
            I[i, 0] = e
        for i, b in enumerate(self._boundary):
            I[i+1, i] = b
        return I

    # FIXME: add caching here?
    def matrix(self, n, m=None):
        """Construct topological relations between elements as sparse matrix"""
        if n < 0:
            n = self.n_dim + n
        if m is None:
            m = n + 1
        elif m == 0:
            i = self._elements[n]
            o = np.ones_like(i)

        if m == n + 1:
            i = self._boundary[n]
            o = self._orientation[n]

        return topology_matrix(i, o)

    def vertex_degree(self):
        """Compute the degree of each vertex; or the number of adjecent n-elements"""
        IN0 = self.incidence[-1, 0]
        vertex_idx, count = npi.count(IN0.flatten())
        c = np.zeros(self.n_vertices)
        c[vertex_idx] = count
        return c

    @cached_property
    def regions_per_vertex(self):
        """Slow but readable code for eaching how many connected regions each vertex has"""
        # convert to column format for fast face-slicing
        TnN = self.matrix(-1).tocsc()
        # iterate over the faces of each vertex
        T0N = self.matrix(0, -1).tocoo()
        I0, IN = T0N.row, T0N.col
        regions = np.zeros(self.n_vertices, np.int)
        for i0, iN in zip(*npi.group_by(I0, IN)):
            # edge-face incidence matrix of faces incident to this vertex
            subset = TnN[:, iN]
            # face adjacency matrix of the faces incident to this vertex
            g = subset.T * subset
            n_components = scipy.sparse.csgraph.connected_components(g, return_labels=False)
            regions[i0] = n_components
        return regions

    # @cached_property
    # def regions_per_vertex_vec(self):
    #     # use npi for vectorized graph algorithm; do floodfill on the n-simplices around each 0-simplex
    #     N = self.n_dim
    #     TnN = self.topology(N-1, N).tocsc()
    #     # iterate over the faces of each vertex
    #     T0N = self.topology(0, N).tocoo()
    #
    #     V, F = T0N.row, T0N.col
    #
    #     P = inc.T * inc
    #     P = P.tocoo()
    #     Pr = P.row
    #     Pc = P.col
    #
    #     inc = inc.tocoo()
    #     # faces with a matching edge index are in contact
    #     EI, FI = inc.row, inc.col
    #     FF = npi.group_by(EI).split_array_as_array(FI)
    #     while True:
    #         # a = npi.indices(F, FF[:, 0])
    #         # b = npi.indices(F, FF[:, 1])
    #         # in I, we remap a to b
    #         # if two face-indices are in the same V-group,
    #         # and they form an a-b tuple, flow I betweeen them
    #
    #         # expand V, F, I by expansion via inc matrix, copying I and V values
    #         # we have multiple faces in F and multiple faces per row of P
    #         # new faces should be the product of those two groups
    #         iF = npi.as_index(F)
    #         iP = npi.as_index(Pr)
    #         iV = npi.as_index(V)
    #         V = V[Pr]
    #         F = Pc
    #         I = I[Pr]
    #         for i in range(3):
    #             v = self.faces[:, i]
    #             f = np.arange(len(v))
    #
    #
    #         # until I converges
    #         (V, F), I = npi.group_by((V, F)).max(I)
    #
    #         break
    #     if len(npi.unique((V, I))[0]) != self.n_vertices:
    #         raise Exception('WRONG')
    #     print()

    def split_connected_components(self):
        """Split the mesh in disjoint connected components

        Returns
        -------
        components : list of Mesh objects
            all disjoint components, sorted from most to least vertices

        """
        TnN = self.matrix(-1)
        graph = TnN.T * TnN     # graph where N-elements are nodes and n-elements are edges
        n_components, labels = scipy.sparse.csgraph.connected_components(graph)
        idx = np.arange(self.n_elements[-1])
        _, elements = npi.group_by(labels).split(idx)
        # FIXME: this presumes a given method of construction... should probably leave this to subclass!
        raise NotImplementedError
        components = [type(self).from_elements(elements=e) for e in elements]
        return sorted(components, key=lambda c: -c.n_elements)

    @cached_property
    def is_connected(self):
        """Returns true of the mesh consists of a single connected component

        Returns
        -------
        bool
            Whether the segment consists of a single connected component
        """
        return len(self.split_connected_components()) == 1

    @abstractmethod
    def select_subchain(self, chain, ndim):
        raise NotImplementedError

    @cached_property
    def is_manifold(self):
        try:
            self.check_manifold()
            return True
        except ManifoldException:
            return False

    def chain(self, n, fill=1, dtype=sign_dtype):
        """allocate an n-chain"""
        c = np.empty(self.n_elements[n], dtype=dtype)
        c.fill(fill)
        return c

    def range(self, n):
        return np.arange(self.n_elements[n], dtype=index_dtype)

    def boundary_indices(self):
        """Return integer indices denoting n-1-elements forming the boundary"""
        chain_N = self.chain(-1, fill=1)

        chain_n = np.abs(self.matrix(-1)) * chain_N
        b_idx = chain_n == 1
        return np.flatnonzero(b_idx)

    @abstractmethod
    def boundary(self):
        raise NotImplementedError

    @cached_property
    def is_closed(self):
        return self.boundary() is None

    @cached_property
    def is_oriented(self):
        """Return true if the topology is oriented"""
        B10 = self._orientation[0]
        B10 = np.sort(B10, axis=1)
        return np.alltrue(B10 == [[-1, +1]])

    def dual(self):
        """return dual topology with closed boundary

        Returns
        -------
        list of dual topology matrices

        Notes
        -----
        dual topology is tranpose of primal topology plus transpose of primal boundary topology
        boundary does not add any dual n-elements
        Every topology matrix needs an identity term added to link the new boundary elements with their
        adjecent dual internal topology
        that is, every T matrix has a block structure, such that:
        [T.T, 0   ]
        [I,   B.T]

        3d example:
        D01 = T23.T + I + B12.T
        D12 = T12.T + I + B01.T
        D23 = T01.T + I

        D0 = P3 + B2
        D1 = P2 + B1
        D2 = P1 + B0
        D3 = P0
        NOTE: D elements generally cannot be constructed in the same layout as the primal ones,
        for lack of uniform valence

        2d example:
        D01 = T12.T + I + B01.T
        D12 = T01.T + I

        """

        def dual_T(T, E, B, Be):
            """Compose dual topology matrix in presence of boundaries

            FIXME: make block structure persistent? would be more self-documenting to vectors
            also would be cleaner to split primal topology in interior/boundary blocks first

            To what extent do we care about relations between dual boundary elements?
            only really care about the caps to close the dual; interrelations appear irrelevant
            """

            idx = npi.indices(E, Be)
            orientation = np.ones_like(idx) # FIXME: this is obviously nonsense; need to work out signs
            I = scipy.sparse.coo_matrix(
                (orientation,
                 (np.arange(len(idx)), idx )),
                shape=(B.shape[0], T.shape[1]) if not B is None else (len(idx), T.shape[1])
            )
            if B is None:
                blocks = [
                    [T],
                    [I]
                ]
            else:
                blocks = [
                    [T, None],
                    [I, B]
                ]
            return scipy.sparse.bmat(blocks)
        boundary = self.boundary()
        CBT = []
        T = [self.matrix(i) for i in range(self.n_dim)]
        E = self.elements
        if not boundary is None:
            BT = [boundary.matrix(i) for i in range(boundary.n_dim)]
            BE = boundary.elements

        for d in range(len(T)):
            CBT.append(
                T[::-1][d].T if boundary is None else
                dual_T(
                    T[::-1][d].T,
                    E[::-1][d+1],
                    BT[::-1][d].T if d < len(BT) else None,
                    BE[::-1][d]
                )
            )
        return CBT

    def relative_orientation(self):
        """Try to find the relative orientation of all n-elements

        Returns
        -------
        n_chain : ndarray, [n_elements], bool

        Raises
        ------
        ValueError
            if the topology is not orientable
        """
        inc = self.matrix(-1)

        # filter out relevant edges
        interior = inc.getnnz(axis=1) == 2
        inc = inc[interior]
        # warning: obviously correct code ahead
        clauses = ((inc.indices + 1) * inc.data).reshape(-1, 2)
        if not len(clauses):
            # FIXME: this is needed for loose triangles; but what about loose disjoint triangle plus other connection component?
            # would prob fail; need to handle remapping back from tris lost in interior filtering
            return self
        orientation = pycosat.solve(clauses.tolist() + (-clauses).tolist())
        if orientation == 'UNSAT':
            raise ValueError('Topology is a non-orientable manifold.')

        # reorient the faces according to the solution found
        orientation = np.array(orientation)[:, None] > 0
        return orientation
