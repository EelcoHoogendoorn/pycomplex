
import operator
import pycosat

import numpy as np
import numpy_indexed as npi
import scipy
import scipy.sparse
from cached_property import cached_property

import pycomplex.sparse
from pycomplex.topology import ManifoldException, sign_dtype, index_dtype
from pycomplex.util import accumulate


class BaseTopology(object):
    """An n-dimensional topology is defined by a sequence of n (sparse) topology matrices, T(n)
    T(n) defines the n-elements in terms of an oriented closed boundary of n-1 elements

    """
    # FIXME: add matrix chain part of constructor here

    @cached_property
    def regions_per_vertex(self):
        """Slow but readable code for counting how many connected regions each vertex has,
        connected to a tiny n-ball centered on the vertex; or in its direct neighborhood

        Returns
        -------
        ndarray, [n_vertices], index_dtype
        """
        # convert to column format for fast slicing from the right
        TnN = self.matrix(-1).tocsc()
        # iterate over the N-elements of each 0-element
        T0N = self.accumulated_operators_N()[0].T.tocoo()
        I0, IN = T0N.row, T0N.col
        regions = self.chain(n=0, fill=0, dtype=index_dtype)
        for i0, iN in zip(*npi.group_by(I0, IN)):
            # n-1-n incidence matrix of n-element incident to this 0-element
            subset = TnN[:, iN]
            # n-element adjacency matrix of the n-elements incident to this 0-element
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

    def label_connections(self):
        """Label all n-elements with a unique integer,
        denoting the connected region they are part of

        Returns
        -------
        n_components : int
        labels : ndarray, [n_elements[-1]], int
        """
        TnN = self.matrix(-1)
        graph = TnN.T * TnN     # graph where N-elements are nodes and n-elements are edges
        n_components, labels = scipy.sparse.csgraph.connected_components(graph)
        return n_components, labels

    @cached_property
    def is_connected(self):
        """Returns true of the mesh consists of a single connected component

        Returns
        -------
        bool
            Whether the segment consists of a single connected component
        """
        n, c = self.label_connections()
        return n == 1

    def check_manifold(self):
        """Raises if a violation of manifoldness is found."""
        if not np.all(self.regions_per_vertex == 1):
            raise ManifoldException

    @cached_property
    def is_manifold(self):
        """True if the topology is manifold."""
        try:
            self.check_manifold()
            return True
        except ManifoldException:
            return False

    def chain(self, n, fill=0, dtype=sign_dtype):
        """Construct an n-chain"""
        c = np.empty(self.n_elements[n], dtype=dtype)
        if isinstance(fill, np.ndarray):
            c.fill(0)
            c[fill] = 1
        else:
            c.fill(fill)
        return c

    def range(self, n):
        """Construct an n-chain filled with an integer range"""
        return np.arange(self.n_elements[n], dtype=index_dtype)

    def form(self, n, fill=0, dtype=np.float32):
        """Construct an n-form"""
        c = np.empty(self.n_elements[n], dtype=dtype)
        c.fill(fill)
        return c

    def relative_parity(self):
        """Try to find the relative parity of all N-elements

        Returns
        -------
        parity : ndarray, [n-N_elements], bool
            N-chain denoting the relative parity of each N-element

        Raises
        ------
        ValueError
            if the topology is not orientable
        """
        if self.n_dim == 0:
            return self.chain(n=-1)

        inc = self.matrix(-1)

        # filter out the relevant interior n-1-elements
        # FIXME: can use selector matrix here? nope; introduces cyclic dependency; needs boundary, which checks orientation
        interior = inc.getnnz(axis=1) == 2
        inc = inc[interior]
        # translate connectivity between n-elements and n-1 elements
        # into clauses constraining the relative orientation of n-elements
        clauses = ((inc.indices + 1) * inc.data).reshape(-1, 2)
        if not len(clauses):
            # FIXME: this is needed for loose n-elements;
            # but what about loose disjoint n-element plus other connection component?
            # would prob fail; need to handle remapping back from n-elements lost in interior filtering
            return self.chain(-1)
        orientation = pycosat.solve(clauses.tolist() + (-clauses).tolist())
        if orientation == 'UNSAT':
            raise ValueError('Topology is a non-orientable manifold.')

        # convert to parity description
        from pycomplex.topology.util import orientation_to_parity
        return orientation_to_parity(orientation)

    @cached_property
    def is_oriented(self):
        """Return true if the topology is oriented"""
        return npi.all_equal(self.relative_parity())

    def check_chain(self):
        """Some basic sanity checks on a topology object
        check that the chain complex satisfies its defining properties
        """
        for i in range(self.n_dim - 2):
            a, b = self.matrices[i], self.matrices[i+1]
            if not (a * b).nnz == 0:
                raise ValueError('chain {i} does not match'.format(i=i+1))

    def accumulated_operators_0(self):
        """Accumulated topology matrices from 0 to n

        Returns
        -------
        list of sparse matrix, [n_0-elements, n-n_elements]
            n-th element of the list maps 0-elements to n-elements

        Notes
        -----
        Primal could override this implementation with a more efficient implementation based on elements/corners arrays

        """
        A = list(accumulate([np.abs(m) for m in self.matrices], func=operator.mul))
        return [scipy.sparse.identity(A[0].shape[0], dtype=sign_dtype)] + A

    def accumulated_operators_N(self):
        """Accumulated topology matrices from N to n

        Returns
        -------
        list of sparse matrix, [n_N-elements, n-n_elements]
            n-th element of the list maps n-elements to N-elements

        """
        A = list(accumulate([np.abs(m.T) for m in self.matrices[::-1]], func=operator.mul))
        A = [scipy.sparse.identity(A[0].shape[0], dtype=sign_dtype)] + A
        return A[::-1]

    @cached_property
    def averaging_operators_0(self):
        """Linear operators that average over all 0-elements of each n-element

        Returns
        -------
        list of sparse matrix, [n-n_elements, n_0-elements]
            n-th element of the list maps 0-elements to n-elements
            all columns in each row sum to one

        """
        return [pycomplex.sparse.normalize_l1(a.T, axis=1) for a in self.accumulated_operators_0()]

    @cached_property
    def averaging_operators_N(self):
        """Linear operators that average over all N-elements of each n-element

        Returns
        -------
        list of sparse matrix, [n-n_elements, n_N-elements]
            n-th element of the list maps N-elements to n-elements
            all columns in each row sum to one

        """
        return [pycomplex.sparse.normalize_l1(a.T, axis=1) for a in self.accumulated_operators_N()]

    @cached_property
    def degree(self):
        """Compute the degree of each n-element; or the number of adjacent N-elements

        Returns
        -------
        list of ndarray, length n_dim + 1
            n-th element of the list is an n-chain denoting the number of incident
            N-elements for each n-element
        """
        A = self.accumulated_operators_N()
        N_elements = self.chain(-1, fill=1)
        # ones_like constructs a summing operator; seek nnz per column
        return [pycomplex.sparse.ones_like(a.T) * N_elements for a in A]

    @cached_property
    def selector_interior(self):
        """Sparse matrices selecting interior portion of a chain"""
        raise NotImplementedError

    @cached_property
    def selector_boundary(self):
        """Sparse matrices selecting boundary portion of a chain"""
        raise NotImplementedError

    @cached_property
    def matrices(self):
        """

        Returns
        -------
        List[Sparse], len(ndim)
            Sequence of signed incidence matrices defining n-elements in terms of their indicence
            with n-1 boundary elements
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """Given that the topology matrices are really the thing of interest of our topology object,
        we make them easily accessible"""
        return self.matrices[item]
