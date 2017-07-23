import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.topology.base import BaseTopology
from pycomplex.topology import sign_dtype, index_dtype


class ClosedDual(BaseTopology):
    """dual for closed primal. should be rather boring"""

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
        return [T.T for T in self.primal.matrices[::-1]]


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
        """number of Dual boundary elements"""
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
            number of elements of each n-dim, primal and dual boundary elements inclusive
        """
        return [p + b for p, b in zip(self.primal.n_elements, self.d_elements)][::-1]

    @cached_property
    def elements(self):
        """Compute all elements in terms of vertex indices? not sure there is a use case yet"""
        raise NotImplementedError

    @cached_property
    def matrices(self):
        """Construct dual topology matrices

        Returns
        -------
        array_like, [n_dim], sparse matrix
        """

        def close_topology(T, idx_p, idx_P):
            """Dual topology constructed by closing partially formed dual elements
            """
            # FIXME: orientation of the closing elements is still failing hard
            # D01 case is simple; add opposing sign.
            # for subsequent operators, only care that product zeros out. can we use this?

            # T.shape = [P, p], or [d, D]

            # z = T[idx_P, :][:, idx_p].tocoo()

            q = np.arange(len(idx_p), dtype=index_dtype)
            orientation = -np.ones_like(idx_p, dtype=sign_dtype)
            # orientation = T[idx_n][:, q]

            I = scipy.sparse.coo_matrix(
                (orientation,
                 (q, idx_p)),
                shape=(len(idx_p), T.shape[1])
            )

            blocks = [
                [T],
                [-I]
            ]
            return scipy.sparse.bmat(blocks)

        boundary = self.primal.boundary
        cap = self.primal.chain(-1, fill=1)
        T = self.primal.matrices
        p_idx = boundary.parent_idx

        # D01 =

        return [close_topology(t.T, b, b2) for t, b, b2 in zip(T, p_idx, p_idx[1:]+None)][::-1]

    @cached_property
    def selector(self):
        """Operators to select primal form from dual form"""

        def s(np, nd):
            return scipy.sparse.eye(np, nd)

        return [s(np, nd) for np, nd in zip(self.primal.n_elements, self.n_elements)]

    @cached_property
    def matrix(self):
        """Construct dual topology matrices

        Returns
        -------
        list of dual topology matrices

        Notes
        -----
        This version attaches the dual boundary information; it is not clear that we actually need this anywhere
        """
        def dual_T(T, B, idx):
            """Compose dual topology matrix in presence of boundaries

            FIXME: make block structure persistent? would be more self-documenting to vectors
            also would be cleaner to split primal topology in interior/boundary blocks first

            To what extent do we care about relations between dual boundary elements?
            only really care about the caps to close the dual; interrelations appear irrelevant
            """

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
                    [I, B]      # its far from obvious any application actually needs this term...
                ]
            return scipy.sparse.bmat(blocks)

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
        return CBT


    def __getitem__(self, item):
        """alias for matrix"""
        return self.matrix[item]

    def form(self, n):
        """allocate a dual n-form. This is a block-vector"""
        bn = self.boundary.n_elements
        i = self.primal.n_elements[n]
        i = i - p
        d = 0
        # FIXME
        return
