import numpy as np
import numpy_indexed as npi
import scipy.sparse
from cached_property import cached_property

from pycomplex.topology.base import BaseTopology


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

        def close_topology(T, idx):
            """Dual topology constructed by closing partially formed dual elements
            """

            # # FIXME: this is obviously nonsense; need to work out signs; grab T[idx] or somesuch
            orientation = -np.ones_like(idx)
            # T[idx]

            I = scipy.sparse.coo_matrix(
                (orientation,
                 (np.arange(len(idx)), idx )),
                shape=(len(idx), T.shape[1])
            )

            blocks = [
                [T],
                [-I]
            ]
            return scipy.sparse.bmat(blocks)

        boundary = self.primal.boundary
        T = self.primal.matrices

        return [close_topology(t.T, b) for t, b in zip(T, boundary.parent_idx)][::-1]

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
        dual topology is transpose of primal topology plus transpose of primal boundary topology
        boundary does not add any dual n-elements
        Every topology matrix needs an identity term added to link the new boundary elements with their
        adjacent dual internal topology
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

        2d example:
        D01 = T12.T + I + B01.T
        D12 = T01.T + I

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
                    [I, B * 0]      # its far from obvious any application actually needs this term...
                ]
            return scipy.sparse.bmat(blocks).T

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
