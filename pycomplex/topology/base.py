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
    @classmethod
    def from_elements(cls, elements):
        raise NotImplementedError

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
            if not b.dtype == index_dtype:
                raise ValueError
            if not o.dtype == sign_dtype:
                raise ValueError
        for e in elements:
            if not e.dtype == index_dtype:
                raise ValueError

    @cached_property
    def elements(self):
        return self._elements

    @cached_property
    def corners(self):
        return [e.reshape(n, -1) for e, n in zip(self._elements, self.n_elements)]

    @cached_property
    def n_elements(self):
        """List of ints; n-th entry is number of n-elements"""
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

    @cached_property
    def is_manifold(self):
        try:
            self.check_manifold()
            return True
        except ManifoldException:
            return False

    def chain(self, n, fill=1, dtype=sign_dtype):
        """Construct an n-chain"""
        c = np.empty(self.n_elements[n], dtype=dtype)
        c.fill(fill)
        return c
    def range(self, n):
        """Construct an n-chain filled with an integer range"""
        return np.arange(self.n_elements[n], dtype=index_dtype)

    @cached_property
    def is_closed(self):
        return self.boundary is None

    @cached_property
    def is_oriented(self):
        """Return true if the topology is oriented"""
        return npi.all_equal(self.relative_orientation())

    @cached_property
    def dual(self):
        """Return dual topology object, that closes all boundaries"""
        from pycomplex.topology.dual import Dual
        return Dual(self)

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
            return self.chain(-1, fill=0)
        orientation = pycosat.solve(clauses.tolist() + (-clauses).tolist())
        if orientation == 'UNSAT':
            raise ValueError('Topology is a non-orientable manifold.')

        # reorient the faces according to the solution found
        orientation = np.array(orientation)[:, None] > 0
        return orientation

    def find_correspondence(self, other, mapping):
        """

        Parameters
        ----------
        self : BaseTopology
        other : BaseTopology
            subset of self
        mapping : ndarray, [other.n_elements[0]], index_type
            indices of self.vertices for each of other.vertices

        Returns
        -------
        idx : list of ndarray
            ith entry of the list contains the index of other in self of the i-elements
        """
        idx = []
        for C, c in zip(self.corners, other.corners):
            c = np.sort(mapping[c], axis=1)
            C = np.sort(C, axis=1)
            # FIXME: find and do something with relative orientation here?
            idx.append(npi.indices(C, c))
        return idx

    def select_subset(self, n_chain):
        """

        Parameters
        ----------
        n_chain : ndarray, [n_elements[-1], sign_type
            chain indicating which elements to select

        Returns
        -------
        type(self)
            Topology of the indicated subset,
            including a precomputed mapping back to the original topology
        """
        n_chain = np.asarray(n_chain)
        if not len(n_chain) == self.n_elements[-1]:
            raise ValueError

        elements = self.elements[-1][np.flatnonzero(n_chain)]   # FIXME: discarding sign info of chain here...

        mapping, inverse = np.unique(elements.flatten(), return_inverse=True)
        elements = inverse.reshape(elements.shape).astype(index_dtype)

        subset = type(self).from_elements(elements)
        subset.parent_idx = self.find_correspondence(subset, mapping)
        return subset

    @cached_property
    def boundary(self):
        """Return n-1-topology representing the boundary

        Returns
        -------
        self.boundary_type
            Boundary topology, with attribute parent_idx referring back to the parent elements
        """

        chain_N = self.chain(-1, fill=1)
        chain_n = self.matrix(-1) * chain_N
        b_idx = np.flatnonzero(chain_n)
        orientation = chain_n[b_idx]
        if len(b_idx) == 0:
            # topology is closed
            return None
        # construct boundary
        elements = self.elements[-2][b_idx]

        # FIXME: enabling this orientation logic breaks subdivision\letter example; need to figure out why
        # shape = np.asarray(elements.shape)
        # shape[1:] = 1
        # elements = np.where(orientation.reshape(shape)==1, elements, elements[:, ::-1])
        mapping, inverse = np.unique(elements.flatten(), return_inverse=True)
        elements = inverse.reshape(elements.shape).astype(index_dtype)

        B = self.boundary_type().from_elements(elements)
        B.parent_idx = self.find_correspondence(B, mapping)
        B.parent = self
        return B
