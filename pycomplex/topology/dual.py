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
        return [T.T for T in self.primal.matrices[::-1]]

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

        How to determine sign of dual boundary-interior connection?
        primal boundary elements contribute dual elements shifted to the left
        in 2d
        how a dual boundary edge sticks to the dual face seems hard
        how primal vertex sticks to edges appears not to inform us; two edges anyway
        dual boundary vertex to dual interior edge has a simple rule;
        just take open dual edge and ensure its closed
        dual faces are not closed since boundary of boundary results in two added vertices
        until i get a better idea; can guess positive orientation

        General structure:
         0i.0p.0d
        [d, 0, 0] 1i
        [d, d, I] 1p
        [0, 0, b] 1d

         1i.1p.1d
        [d, 0, 0] 2i
        [d, d, I] 2p
        [0, 0, b] 2d

        I-terms obey => b I == I b. find n-1 flips that neutralize n-flips
        alternatively, make sure sign plays no role
        that requires that both the interior and boundary are oriented
        can we adjust 'generate_boundary' such that this holds? think so...
        yes we can, except for P01; is a special case, just like D01

        Returns
        -------
        array_like, [n_dim], sparse matrix
        """

        def close_topology(T, idx_p, idx_P, i):
            """Dual topology constructed by closing partially formed dual elements
            """
            # FIXME: orientation of the closing elements is still failing hard
            # PNn/D01 case is simple; add opposing sign.
            # for subsequent operators, only care that product zeros out. can we use this?
            # and how important are subsequent operators, really?

            # T.shape = [P, p], or [d, D]
            if i == 1:
                q = T.sum(axis=0)  # sum over all dual vertices / primal faces per edge; shape [d_edges]
                orientation = np.asarray(q[0, idx_p]).flatten()
            else:
                orientation = -np.ones_like(idx_p, dtype=sign_dtype)

            q = np.arange(len(idx_p), dtype=index_dtype)
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

        P = self.primal.matrices
        p_idx = boundary.parent_idx

        if False:
            # attempted special case logic for 2d case
            P01, P12 = P
            D01 = P12.T # D01 has shape [d_vertices, d_edges]
            q = D01.sum(axis=0)  # sum over all dual vertices / primal faces per edge; shape [d_edges]
            orientation = np.asarray(q[0, p_idx[1]]).flatten()
            q = np.arange(len(p_idx[1]), dtype=index_dtype)
            # add a closing vertex for each dual edge
            I = scipy.sparse.coo_matrix((orientation * -1, (q, p_idx[1])), shape=(len(q), D01.shape[1]))
            D01 = scipy.sparse.bmat([[D01], [I]]) # add dual vertices to close the edges
            q = D01.sum(axis=0)     # sum over all dual vertices / primal faces per edge; shape [d_edges]
            assert np.all(q==0)     # check that all edges are indeed closed

            # hmm; if we want to be truly closed, we need the dual boundary element block in D01
            # what are the implications of this? is an interaction term between 0 and 2 forms truly sound?
            D02 = D01 * P01.T

            import matplotlib.pyplot as plt
            # plt.scatter(z.row, z.col, z.data)
            plt.imshow(D02.todense(), cmap='bwr')
            plt.show()


            D12 = P01.T

            # interior_edges = 1 - self.primal.chain(1, fill=p_idx[1])
            # q = P01[p_idx[0], :][:, interior_edges.astype(np.bool)]
            # orientation = np.asarray(q).flatten()

            q = np.arange(len(p_idx[0]), dtype=index_dtype)
            # add a closing egde for each dual face
            I = scipy.sparse.coo_matrix((orientation * -1, (q, p_idx[0])), shape=(len(q), D12.shape[1]))

            D12 = scipy.sparse.bmat([[D12], [I]]) # add dual edges to close the faces


            q = D12.sum(axis=0)     # sum over all dual vertices / primal faces per edge; shape [d_edges]

            q = np.arange(len(p_idx[0]), dtype=index_dtype)
            I = scipy.sparse.coo_matrix((orientation * -1, (q, p_idx[0])), shape=(len(q), D12.shape[1]))
            D12 = scipy.sparse.vstack([D12, I]) # add dual vertices to close the edges

            q = D12.sum(axis=0)     # sum over all edges per dual face; shape [d_faces]

            return [D01, D12]

        return [close_topology(t.T, b, b2, i)
                for i, (t, b, b2) in enumerate(zip(P, p_idx, p_idx[1:]+[None]))][::-1]

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
            only really care about the caps to close the dual; interrelations appear irrelevant as far as i can tell so far

            However, may play an important part in constructing the connection?
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

        for i in range(len(CBT)):
            # patch up connection signs
            pass

        return [scipy.sparse.bmat(t) for t in CBT]

    def __getitem__(self, item):
        """Given that the topology matrices are really the thing of interest of our dual object,
        we make them easily accessible"""
        return self.matrix[item]

    def form(self, n):
        """allocate a dual n-form. This is a block-vector"""
        bn = self.boundary.n_elements
        i = self.primal.n_elements[n]
        i = i - p
        d = 0
        # FIXME
        return
