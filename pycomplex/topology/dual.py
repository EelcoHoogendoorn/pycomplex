import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.topology.base import BaseTopology
from pycomplex.topology import sign_dtype, index_dtype


class ClosedDual(BaseTopology):
    """Dual to a closed primal. should be rather boring"""

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
        def s(np):
            return scipy.sparse.eye(np)
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
        return [T.T for T in self.primal.matrices[::-1]]

    @cached_property
    def matrices_2(self):
        """Construct dual topology matrices stripped of dual boundary topology
        This leaves us at liberty to construct custom boundary conditions

        Returns
        -------
        array_like, [n_dim], sparse matrix
        """
        M = self.matrices    # [D0D1 ... DnDN]
        S = self.selector    # [P0DN ... PND0]
        return [m * s.T for m, s in zip(M, S[::-1][1:])]

    # @cached_property
    # def matrices_3(self):
    #     """Construct dual topology matrices stripped of dual boundary topology
    #     This leaves us at liberty to construct custom boundary conditions
    #
    #     Returns
    #     -------
    #     array_like, [n_dim], sparse matrix
    #     """
    #     S = self.selector[::-1]
    #     return [l * m * r.T for l, m, r in zip(S[:-1], self.matrices, S[1:])]

    @cached_property
    def selector(self):
        """Operators to select primal form from dual form

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping dual forms to primal subset
            first element of this list is square; maps dual n-forms to primal 0-forms, which are one-to-one
        """

        def s(np, nd):
            return scipy.sparse.eye(np, nd)

        return [s(np, nd) for np, nd in zip(self.primal.n_elements, self.n_elements[::-1])]

    @cached_property
    def matrices(self):
        """Construct dual topology matrices

        Returns
        -------
        list of dual topology matrices, len self.n_dim
            the chain complex defining the dual topology

        Notes
        -----
        This version attaches the dual boundary information; the dual chain will thus be closed
        Note that this requires that both the primal and its boundary are oriented
        """
        def dual_T(T, B, idx):
            """Compose dual topology matrix in presence of boundaries

            FIXME: make block structure persistent? would be more self-documenting to vectors
            also would be cleaner to split primal topology in interior/boundary blocks first

            """

            orientation = np.ones_like(idx)

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
                    [I, B]      # its far from obvious any application actually needs this term...
                ]
            return (blocks)

        boundary = self.primal.boundary
        CBT = []
        T = [self.primal.matrix(i) for i in range(self.primal.n_dim)]
        if not boundary is None:
            BT = [boundary.matrix(i) for i in range(boundary.n_dim)]

        for d in range(len(T)):
            CBT.append(
                T[::-1][d].T if boundary is None else
                dual_T(
                    T[::-1][d].T,
                    BT[::-1][d].T if d < len(BT) else None,
                    boundary.parent_idx[::-1][d]
                )
            )

        return [scipy.sparse.bmat(t) for t in CBT]

    def __getitem__(self, item):
        """Given that the topology matrices are really the thing of interest of our dual object,
        we make them easily accessible"""
        return self.matrices[item]

    def form(self, n):
        """allocate a dual n-form. This is a block-vector"""
        bn = self.boundary.n_elements
        i = self.primal.n_elements[n]
        i = i - p
        d = 0
        # FIXME
        return
