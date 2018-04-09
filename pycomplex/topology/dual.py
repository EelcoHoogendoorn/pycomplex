import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.topology.base import BaseTopology
import pycomplex.sparse


class ClosedDual(BaseTopology):
    """Dual to a closed primal. should be rather boring. no need for primal/dual blocked terms here"""

    def __init__(self, primal):
        # bind primal; all internals are lazily computed
        self.primal = primal
        if not primal.boundary is None:
            raise ValueError('Closed dual is not appropriate!')

    @cached_property
    def n_dim(self):
        return self.primal.n_dim

    @cached_property
    def n_elements(self):
        return self.primal.n_elements[::-1]

    @cached_property
    def matrices(self):
        return self.matrices_original

    @cached_property
    def matrices_2(self):
        return self.matrices

    @cached_property
    def matrices_original(self):
        return [T.T for T in self.primal.matrices[::-1]]

    @cached_property
    def selector_interior(self):
        """Mapping to interior of closed topology always is identity mapping"""
        def s(np):
            return scipy.sparse.eye(np)
        return [s(np) for np in self.primal.n_elements]

    @cached_property
    def selector_boundary(self):
        """Mapping to boundary of closed topology always gives vanishing chain"""
        def s(np):
            return pycomplex.sparse.sparse_zeros((0, np))
        return [s(np) for np in self.primal.n_elements]

    def __getitem__(self, item):
        """alias for matrix"""
        return self.matrices[item]


class Dual(BaseTopology):
    """Object to wrap dual topology logic

    In the internal of the manifold the dual topology is simply the primal topology transposed

    However, on the boundary, we need some extra attention, which this class encapsulates

    In short, we can think of the total dual as the dual of primal, plus the dual of the primal boundary
    """

    def __init__(self, primal):
        # bind primal; all internals are lazily computed
        self.primal = primal
        if primal.boundary is None:
            raise ValueError('Construct closed dual instead!')

    @cached_property
    def n_dim(self):
        return self.primal.n_dim

    @cached_property
    def p_elements(self):
        """number of Primal boundary elements"""
        boundary = self.primal.boundary
        return boundary.n_elements + [0]
    @cached_property
    def d_elements(self):
        """number of Dual boundary elements, per primal index"""
        boundary = self.primal.boundary
        return [0] + boundary.n_elements
    @cached_property
    def i_elements(self):
        """number of Internal elements of primal"""
        return [p - b for p, b in zip(self.primal.n_elements, self.p_elements)]

    @cached_property
    def n_elements(self):
        """

        Returns
        -------
        list of ints
            number of elements of each n-dim, primal and dual boundary elements inclusive, per dual index
        """
        return [p + b for p, b in zip(self.primal.n_elements, self.d_elements)][::-1]

    @cached_property
    def elements(self):
        """Compute all elements in terms of vertex indices? not sure there is a use case yet"""
        raise NotImplementedError

    @cached_property
    def matrices_original(self):
        """Topology matrices without any dual boundary topology attached"""
        return [T.T for T in self.primal.matrices[::-1]]

    @cached_property
    def matrices_2(self):
        """Construct dual topology matrices stripped of dual boundary topology
        This is the discrete derivative operator that we are usually interested in

        Returns
        -------
        array_like, [n_dim], sparse matrix
            n-th element describes incidence of dual n-elements to dual n+1 elements
        """
        # FIXME: come up with a descriptive name for this.
        M = self.matrices    # [D0D1 ... DnDN]
        S = self.selector_interior    # [P0DN ... PND0]
        return [m * s.T for m, s in zip(M, S[::-1][1:])]

    @cached_property
    def selector_interior(self):
        """Operators to select interior elements; or to strip boundary elements,
        or those that do not have a corresponding primal element

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping dual forms to primal subset
            first element of this list is square; maps dual n-forms to primal 0-forms, which are one-to-one
            list is indexed by primal form
        """
        def s(np, nd):
            return scipy.sparse.eye(np, nd)

        return [s(np, nd) for np, nd in zip(self.primal.n_elements, self.n_elements[::-1])]

    @cached_property
    def selector_boundary(self):
        """Operators to select interior elements; or to strip boundary elements,
        or those that do not have a corresponding primal element

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping dual forms to dual boundary subset
            first element of this list is trivial
            list is indexed by primal form
        """
        def s(np, nd):
            return scipy.sparse.eye(nd).tocsr()[np:, :]

        return [s(np, nd) for np, nd in zip(self.primal.n_elements, self.n_elements[::-1])]

    @cached_property
    def matrices(self):
        """Construct dual topology matrices, including the topology of the boundary, and its connection to the interior

        Returns
        -------
        list of dual topology matrices, len self.n_dim
            the chain complex defining the dual topology,
            where the n-th element of the list has shape [n_elements, n+1_elements]

        Notes
        -----
        This version attaches the dual boundary topology; the dual chain will thus be closed
        Note that this requires that both the primal and its boundary are oriented
        """
        assert self.primal.is_oriented
        boundary = self.primal.boundary

        if boundary is None:
            BT = [pycomplex.sparse.sparse_zeros((0, 0)) for _ in range(self.n_dim)]
        else:
            assert boundary.is_oriented
            BT = boundary.matrices
            BT = [pycomplex.sparse.sparse_zeros((0, BT[0].shape[0]))] + BT

        T = self.primal.matrices
        # this term effectively 'glues' the interior and boundary topology together
        Spb = self.primal.selector_boundary

        CBT = [scipy.sparse.bmat(
                [[T[d], Spb[d].T],
                 [None,  -BT[d]]]
            ) for d in range(self.n_dim)]
        return [T.T for T in CBT[::-1]]

    def __getitem__(self, item):
        """Given that the topology matrices are really the thing of interest of our dual object,
        we make them easily accessible"""
        return self.matrices[item]

    def transfer_matrices(self):
        """Construct dual transfer matrices

        Returns
        -------
        List[sparse]
            n-th sparse matrix relates fine and coarse primal n-elements

        Notes
        -----
        logic is similar to topology matrices; just copy the relevant block from the primal and concat
        """
        fine = self
        coarse = self.primal.parent.dual
        T = self.primal.transfer_matrices   # coarse to fine on the primal
        fine_p = fine.primal.boundary.parent_idx
        coarse_p = coarse.primal.boundary.parent_idx
        result = []
        for n, (t, bt, fp, cp) in enumerate(zip(T[::-1], T[:-1][::-1], fine_p[::-1], coarse_p[::-1])):
            b = bt[fp, :][:, cp] # select relevant part of boundary transfer
            blocks = [
                [t, None],
                [None, b]
            ]
            result.append(scipy.sparse.bmat(blocks))
        result.append(T[0])  # dual of primal vertices do not have anything added to them
        return result

    # def form(self, n):
    #     """allocate a dual n-form. This is a block-vector"""
    #     bn = self.boundary.n_elements
    #     i = self.primal.n_elements[n]
    #     i = i - p
    #     d = 0
    #     # FIXME
    #     return
