import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.topology.base import BaseTopology
from pycomplex.topology import sign_dtype, index_dtype
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
    def selector(self):
        """Mapping to interior of closed always is identity mapping"""
        def s(np):
            return scipy.sparse.eye(np)
        return [s(np) for np in self.primal.n_elements]

    @cached_property
    def selector_b(self):
        """Mapping to boundary of closed always gives vanishing vector"""
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
        # FIXME: come up with a descriptive name for this. call it matrices; and rename matrices to matrices_full?
        M = self.matrices    # [D0D1 ... DnDN]
        S = self.selector    # [P0DN ... PND0]
        return [m * s.T for m, s in zip(M, S[::-1][1:])]

    @cached_property
    def selector(self):
        """Operators to select interior elements; or to strip boundary elements,
        or those that do not have a corresponding primal element

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping dual forms to primal subset
            first element of this list is square; maps dual n-forms to primal 0-forms, which are one-to-one
            list is indexed by primal form
        """
        # FIXME: rename to interior_selector or somesuch? bit more descriptive
        def s(np, nd):
            return scipy.sparse.eye(np, nd)

        return [s(np, nd) for np, nd in zip(self.primal.n_elements, self.n_elements[::-1])]

    @cached_property
    def selector_b(self):
        """Operators to select interior elements; or to strip boundary elements,
        or those that do not have a corresponding primal element

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping dual forms to dual boundary subset
            first element of this list is trivial
            list is indexed by primal form
        """
        # FIXME: rename to boundary_selector or somesuch? bit more descriptive
        def s(np, nd):
            return scipy.sparse.eye(nd).tocsr()[np:, :]

        return [s(np, nd) for np, nd in zip(self.primal.n_elements, self.n_elements[::-1])]

    @cached_property
    def matrices(self):
        """Construct dual topology matrices, including the topology of the boundary, and its connection to the interior

        Returns
        -------
        list of dual topology matrices, len self.n_dim
            the chain complex defining the dual topology

        Notes
        -----
        This version attaches the dual boundary information; the dual chain will thus be closed
        Note that this requires that both the primal and its boundary are oriented
        """
        assert self.primal.is_oriented
        assert self.primal.boundary.is_oriented

        def dual_T(T, B, idx):
            """Compose dual topology matrix in presence of boundaries

            FIXME: make block structure persistent? would be more self-documenting to vectors
            also would be cleaner to split primal topology in interior/boundary blocks first
            alternatively; make more principled use of selection matrices

            """

            orientation = np.ones_like(idx)

            # FIXME: seems like we are just recreating primal selection matrix here again. due for a rewrite?
            # NOTE: one difference; pure I term is from boundary space of one type of element to interior space of another
            I = scipy.sparse.coo_matrix(
                (orientation,
                 (np.arange(len(idx)), idx)),
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
                    [I, B]      # it is far from obvious any application actually needs the B term...
                ]
            return (blocks)

        boundary = self.primal.boundary
        CBT = []
        T = [self.primal.matrix(i) for i in range(self.primal.n_dim)]
        S = self.primal.selector
        Sb = self.primal.selector_b

        if not boundary is None:
            BT = [boundary.matrix(i) for i in range(boundary.n_dim)]

        for d in range(len(T)):
            CBT.append(
                T[::-1][d].T if boundary is None else
                dual_T(
                    T[::-1][d].T,
                    BT[::-1][d].T if d < len(BT) else None,
                    # (S[::-1][d+1] * Sb[::-1][d+1].T).T
                    boundary.parent_idx[::-1][d]
                )
            )

        return [scipy.sparse.bmat(t) for t in CBT]

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
        # little check
        for r, c, f in zip(result, coarse.n_elements, fine.n_elements):
            assert (r.shape == r, c)
        return result

    # def form(self, n):
    #     """allocate a dual n-form. This is a block-vector"""
    #     bn = self.boundary.n_elements
    #     i = self.primal.n_elements[n]
    #     i = i - p
    #     d = 0
    #     # FIXME
    #     return
