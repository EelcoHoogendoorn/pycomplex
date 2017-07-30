
import numpy as np
import numpy_indexed as npi
import scipy
import scipy.sparse
import pycosat
from cached_property import cached_property

from pycomplex.topology import ManifoldException
from pycomplex.topology import sign_dtype, index_dtype


class BaseTopology(object):
    """An n-dimensional topology is defined by a sequence of n (sparse) topology matrices, T(n)
    T(n) defines the n-elements in terms of an oriented closed boundary of n-1 elements

    """
    # FIXME: add matrix chain part of constructor here

    def vertex_degree(self):
        """Compute the degree of each vertex; or the number of adjecent n-elements"""
        IN0 = self.incidence[-1, 0]
        _, count = npi.count(IN0.flatten())
        return count

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

    def label_connections(self):
        """Label all n-elements with int denoting the connected region they are part of

        Returns
        -------
        n_components : int
        labels : ndarray, [n_elements[-1], int
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

    @cached_property
    def is_manifold(self):
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
            return self.chain(-1)
        orientation = pycosat.solve(clauses.tolist() + (-clauses).tolist())
        if orientation == 'UNSAT':
            raise ValueError('Topology is a non-orientable manifold.')

        # reorient the faces according to the solution found
        orientation = np.array(orientation)[:, None] > 0
        return orientation

    @cached_property
    def is_oriented(self):
        """Return true if the topology is oriented"""
        return npi.all_equal(self.relative_orientation())

    def check_chain(self):
        """Some basic sanity checks on a topology matrix"""
        for i in range(self.n_dim - 2):
            a, b = self.matrices[i], self.matrices[i+1]
            if not (a * b).nnz() == 0:
                raise ValueError(f'chain [{i}, {i+1}] to [{i+1}, {i+2}] does not match')


