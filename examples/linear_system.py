import numpy
import numpy as np
import scipy.sparse
from pycomplex.topology import sign_dtype, index_dtype

from cached_property import cached_property
from pycomplex import block
import pycomplex.sparse


class System(object):
    """Generalized linear system; like class below but with block function filtered out

    does it handle naming of things?
    should we make specific physics a subclass hereof?
    generally initialized with slice of cochain complex
    would like to know what dimension each form is; maybe that should be input?
    note that 4-th order potentials, or multiple loops around the cochain complex,
    cannot be sliced directly as such; may need alternative constructors. thats ok

    can we efficiently construct laplace-beltrami this way, through elimination
    of noncentral equation(s)?

    is the B matrix really required in this scope?
    seems to only come into play for eigensolver, really
    note that once we start talking about eliminating variables,
    this class is no longer the right abstraction;
    this class is very much about the direct connection to the cochain complex

    default setup is that system matrix takes dual from rhs and primal from lhs

    make this class encapsulate boundary condition manipulation
    relies on mutability atm; can we make immutable design?
    also should have interface for material parameter variations / metric-warps

    can use nested block structure; but start with plain block structure.
    just use this class to abstract away the accessing of the logically different subblocks
    that works; boundary setters are sure a lot cleaner

    how to handle rhs? feels a bit clunky to be tied to a single rhs
    should we assemble boundary conditions in a seperate datastructure,
    to be added on top of core equations later / at will?

    altenative to block datastructure is to let a class like this encapsulate all offsetting logic,
    the way selection matrices are now used to manage substructure of first order operators


    full cochain complex in 3d; var idx refer to primal index

    [I , 01, 0 , 0 ]
    [01, I , 12, 0 ]
    [0 , 12, I,  23]
    [0 , 0 , 23, I ]

    δpp block is denoted b; boundary topology
    δpd block is an identity term
    dip block is always zero; interior verts do not connect to boundary edges; and so on
    dpi block is never zero; interior edges do connect to boundary verts; and so on
    note that ddd block is always initialized at zero now; perhaps we should include it in setup,
    and have a seperate function to zero out dual equations and columns to set custom bcs?
    perhaps the full structure would be usefull in deriving mg smoother;
    not obvious what the bcs for that ought to be;
    although 'anything not involving the central form directly' may be a good bet?

    [[I, 0], [δ, 0, 0], [0, 0, 0], [0, 0]] [0i]   [0i]
    [[0, I], [δ, b, I], [0, 0, 0], [0, 0]] [0p]   [0p]

    [[d, d], [I, 0, 0], [δ, 0, 0], [0, 0]] [1i]   [1i]
    [[0, b], [0, I, 0], [δ, b, I], [0, 0]] [1p]   [1p]
    [[0, _], [0, 0, _], [0, 0, 0], [0, 0]] [1d]   [_]

    [[0, 0], [d, d, 0], [I, 0, 0], [δ, 0]] [2i]   [2i]
    [[0, 0], [0, b, 0], [0, I, 0], [I, I]] [2p] = [2p]
    [[0, 0], [0, _, 0], [0, 0, _], [0, 0]] [2d]   [_]

    [[0, 0], [0, 0, 0], [d, I, 0], [I, 0]] [3i]   [3i]
    [[0, 0], [0, 0, 0], [0, _, 0], [0, _]] [3d]   [_]

    can we think of a third order system that actually models something interesting?
    would it have interesting dynamics that may not relate to any model of physics?
    or is this mathematically precluded since chaining exterior derivatives trivialize?
    not a clue, but would be fun to play with.

    """
    def __init__(self, complex, A, B=None, L=None, R=None, rhs=None):
        """

        Parameters
        ----------
        A : sparse block matrix
            left matrix in A x = B y
        B : sparse block matrix, optional
            right matrix in A x = B y
        L : List[int]
            primal form associated with rows of the system
            or space of left-multiplication
        R : List[int]
            primal form associated with columns of the system
            or space of right-multiplication
        """
        self.complex = complex
        self.A = A
        self.B = B
        self.L = np.array(L)
        self.R = np.array(R)
        self.rhs = self.allocate_y() if rhs is None else rhs

    def copy(self, **kwargs):
        """Copy self with some constructor args overridden. Part of general functional logic"""
        import funcsigs
        args = funcsigs.signature(type(self)).parameters.keys()
        nkwargs = {}
        for a in args:
            if hasattr(self, a):
                nkwargs[a] = getattr(self, a)

        nkwargs.update(kwargs)
        c = type(self)(**nkwargs)
        c.parent = self
        return c

    @staticmethod
    def canonical(complex):
        """Set up the full cochain complex

        Parameters
        ----------
        complex : BaseComplex

        Returns
        -------
        System
            system of full cochain complex
            default boundaries are blank
            maps from dual to primal
            as a result the first order system is symmetric
            and we avoid needing to have a dual boundary metric

        Examples
        --------
        for a 3d complex, the full structure is thus, and scales according to this pattern with dimension

        [ * , *δ , 0  , 0  ]
        [ d*, *  , *δ , 0  ]
        [ 0 , d* , *  , *δ ]
        [ 0 , 0  , d* , *  ]

        note that the inclusion of hodges can make rows/cols kinda unbalanced
        primal hodges scales as l^k
        and corresponding dual as l^(n-k)
        with l a characteristic edge length
        so f.i. hodge mapping from dual to primal on 2d complex scales as
        0: -2
        1: 0
        2: +2

        on 3d:
        0: -3
        1: -1
        2: +1
        3: +3

        Is this an argument in favor of working in 'dimensionless midpoint space' by default?
        worked for eschereque
        if doing so, we should not expect refinement to influence the scaling of our equations at all?
        every extterior derivative would be wrapped left and right by metric.
        multiply with k-metric, exterior derivative, and then divide by k+1 metric;
        so every operator scales as 1/l
        may also be more elegant in an mg-transfer scenario?


        """
        # make sure we are not engaging in nonsense here
        complex.topology.check_chain()
        complex.topology.dual.check_chain()

        T = complex.topology.dual.matrices_2[::-1]
        N = complex.n_dim + 1

        NE = complex.topology.dual.n_elements[::-1]
        A = block.SparseBlockMatrix(
            [[pycomplex.sparse.sparse_zeros((NE[i], NE[j])) for j in range(N)] for i in range(N)])
        S = complex.topology.dual.selector
        # FIXME: make these available as properties on the System?
        PD = [scipy.sparse.diags(pd) for pd in complex.hodge_PD]
        # these terms are almost universally required
        for i, t in enumerate(T):
            A.block[i, i + 1] = S[i].T * PD[i] * t.T
            A.block[i + 1, i] = S[i+1].T * S[i+1] * t * PD[i] * S[i]    # selector zeros out the boundary conditions

        # put hodges on diag by default; easier to zero out than to fill in
        for i in range(N):
            A.block[i, i] = S[i].T * PD[i] * S[i]

        LR = np.arange(N, dtype=index_dtype)
        return System(complex, A=A, B=None, L=LR, R=LR)

    @staticmethod
    def canonical_midpoint(complex):
        """Set up the full cochain complex, in midpoint-space

        Parameters
        ----------
        complex : BaseComplex

        Returns
        -------
        System
            system of full cochain complex
            default boundaries are blank
            maps from midpoint to midpoint
            as a consequence, equations are naturally balanced
            however, the first order system will not be symmetrical;
            needs to be solved by elimination or normal equations

            and we need a dual boundary metric, unlike the dual-to-primal formulation

        Examples
        --------
        for a 3d complex, the full structure is thus, and scales according to this pattern with dimension

        [ I , δ , 0 , 0 ]
        [ d,  I , δ , 0 ]
        [ 0 , d , I , δ ]
        [ 0 , 0 , d , I ]
        """
        # FIXME: should this be a seperate or subclass? probably many  methods in the curret system calss are tied to assumptions in the canonical method
        # make sure we are not engaging in nonsense here
        complex.topology.check_chain()
        complex.topology.dual.check_chain()

        # these are matrices with dual boundary term dropped; use full and leave this as seperate steo?
        T = complex.topology.dual.matrices_2[::-1]
        N = complex.n_dim + 1

        NE = complex.topology.dual.n_elements[::-1]
        A = block.SparseBlockMatrix(
            [[pycomplex.sparse.sparse_zeros((NE[i], NE[j])) for j in range(N)] for i in range(N)])
        S = complex.topology.dual.selector
        # FIXME: make these available as properties on the System?
        Mp, Md = complex.metric
        Md = Md[::-1]
        # FIXME: boundary metric needs to be available to complete dual metric operator
        # FIXME: current dual boundary metric for regular complex seems broken; zeros on edges?
        Mb = complex.boundary.metric
        D = scipy.sparse.diags
        for i, t in enumerate(T):
            A.block[i, i + 1] = S[i].T * (D(1/Md[i]) * t.T * D(Md[i+1])) * S[i+1]
            A.block[i + 1, i] = S[i+1].T * S[i+1] * (D(1/Mp[i+1]) * t * D(Mp[i])) * S[i]    # selector zeros out the boundary conditions

        # put identity on diag by default; easier to zero out than to fill in
        for i in range(N):
            A.block[i, i] = S[i].T * scipy.sparse.eye(NE[i]) * S[i]

        LR = np.arange(N, dtype=index_dtype)
        return System(complex, A=A, B=None, L=LR, R=LR)

    def __getitem__(self, item):
        """Slice a subsystem of full cochain complex"""
        return System(
            self.complex,
            self.A.__getitem__(item).copy(),
            L=self.L[item[0]],
            R=self.R[item[1]],
        )

    def set_dia_boundary(self, i, j, d):
        """Set a boundary on a 'diagonal' term of the cochain complex
        This influences a dual boundary variable

        Parameters
        ----------
        i : int
            index of row/equation
        j : int
            index of column/variable
        d : array_like, boundary chain
            one value for each boundary element

        Side effects
        ------------
        modifies the A matrix
        """
        assert self.L[i] == self.R[j]   # this implies diagonal of the cochain complex

        S = self.complex.topology.dual.selector_b[self.L[i]]
        self.A.block[i, j] = self.A.block[i, j] + scipy.sparse.diags(S.T * d)

    def set_off_boundary(self, i, j, o):
        """Set a boundary on a 'off-diagonal' term of the cochain complex
        This influences a primal boundary variable

        Parameters
        ----------
        i : int
            index of row/equation
        j : int
            index of column/variable
        o : array_like, boundary chain
            one value for each boundary element

        Side effects
        ------------
        modifies the A matrix
        """
        assert self.L[i] == self.R[j] + 1   # this implies entry below the diagonal, or primal exterior derivative

        Srd = self.complex.topology.dual.selector[self.R[j]]
        Srp = self.complex.topology.selector_b[self.R[j]]
        Sld = self.complex.topology.dual.selector_b[self.L[i]]

        self.A.block[i, j] = self.A.block[i, j] + Sld.T * scipy.sparse.diags(o) * Srp * Srd

    def set_rhs_boundary(self, i, r):
        """Set a boundary term on the rhs

        Parameters
        ----------
        i : int
            index of row/equation
        r : array_like, boundary chain
            one value for each boundary element

        Side effects
        ------------
        modifies the rhs vector
        """
        S = self.complex.topology.dual.selector_b[self.L[i]]
        self.rhs.block[i] = self.rhs.block[i] + S.T * r

    def set_rhs(self, i, r):
        """Set a source term on the rhs

        Parameters
        ----------
        i : int
            index of row/equation
        r : array_like, primal chain
            one value for each primal element

        Side effects
        ------------
        modifies the rhs vector
        """
        S = self.complex.topology.dual.selector[self.L[i]]
        self.rhs.block[i] = self.rhs.block[i] + S * r

    def set_sum_boundary(self, i, j, s, row, sum):
        assert self.L[i] == self.R[j]   # this implies diagonal of the cochain complex
        # FIXME: is there also some S matrix magic for this?
        cols = np.flatnonzero(s).astype(index_dtype)
        rows = np.ones_like(cols) * row
        data = s[cols]
        b = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(len(s),)*2)

        S = self.complex.topology.dual.selector_b[self.L[i]]
        self.A.block[i, j] = self.A.block[i, j] + S.T * b * S

        q = np.zeros_like(s)
        q[row] = sum
        self.set_rhs_boundary(i, q)

    def plot(self, dense=True, order=None):
        """Visualize the structure of the linear system

        Parameters
        ----------
        dense : bool
            dense looks better; but only feasible for small systems
        """
        import matplotlib.pyplot as plt
        S = self.A.merge()
        if not dense:
            S = S.tocoo()
            plt.scatter(S.col, S.row, c=S.data, cmap='seismic', vmin=-4, vmax=+4)
            plt.gca().invert_yaxis()
        else:
            S = S.todense()
            if order is not None:
                S = S[order, :][:, order]
            cmap = 'bwr'#''seismic'
            plt.imshow(S[::-1], cmap=cmap, vmin=-1, vmax=+1, origin='upper', extent=(0, S.shape[1], 0, S.shape[0]))
            # # plot block boundaries
            # for l in np.cumsum([0] + self.rows):
            #     plt.plot([0, S.shape[1]], [l, l], c='k')
            # for l in np.cumsum([0] + self.cols):
            #     plt.plot([l, l], [0, S.shape[0]], c='k')
            plt.gca().invert_yaxis()

        plt.axis('equal')
        plt.show()


    # FIXME:  all the below are solver-related nonsense. should that be in a seperate class?
    # perhaps this class should just be a matrix assembly helper
    def allocate_x(self):
        N = self.complex.topology.n_dim
        b = np.zeros(len(self.R), np.object)
        for i, k in enumerate(self.R):
            b[i] = self.complex.topology.dual.form(n=N-k)
        return block.DenseBlockArray(b)

    def allocate_y(self):
        N = self.complex.topology.n_dim
        b = np.zeros(len(self.L), np.object)
        for i, k in enumerate(self.L):
            b[i] = self.complex.topology.dual.form(n=N-k)
        return block.DenseBlockArray(b)

    def eliminate(self, rows, cols):
        """Symbolically eliminate the rows specified

        Parameters
        ----------
        rows : List[int]
            rows to be eliminated
        cols : List[int]
            cols to be eliminated
            is it ever different from rows?
            yes it is! current implementation only considers diag elimination but there are other types
            but maybe that should be a seperate method

        Returns
        -------
        System
            equivalent system with variable eliminations applied

        Examples
        --------
        what elimination looks like for a streamfunction:

        [0, δ] [phi ] = [r]
        [d, I] [flux] = [0]

        [phi ] = [I] [phi]
        [flux] = [d]

        [I, d] from the left as well? or just plain selection of retained equations?

        or for vector laplace:
        [I, δ, 0] [r] = [0]
        [d, 0, δ] [f] = [0]
        [0, d, I] [p] = [0]

        [r] = [δ]
        [f] = [I] [f]
        [p] = [d]

        or for stokes:
        [I, δ, 0] [r] = [0]
        [d, 0, δ] [f] = [0]
        [0, d, 0] [p] = [0]

        [r] = [δ, 0]
        [f] = [I, 0] [f]
        [p] = [0, I]

        seek to handle arbitrary laplace-type equations this way
        but what about incomplete diagonals then?
        like for instance a normal-flux constraint
        can rewrite those dual pressure vars as terms of laplace of flux... this is the goal in the end; to rewrite as flux
        alternatively, view like this: either we have a nonzero diag, in which case row can be eliminated,
        or we have a simple off-diag constraint; can also be used to eliminate?
        current laplace implementation simply assumed diag constrained to zero everywhere
        if normal flux is prescribed, we can zero out the corresponding column and move it to rhs

        do we have a general theory of elimination? each to eliminate a group of vars,
        if we have a set of equations that are easily inverted wrt the appearance of that group of vars
        like an I block, it should be possible.
        note that eliminating P via rewrie of middle eq also requires elimination of normal flux col
        this should be easy tho
        altogether still quite involved tho
        can work in two passes; retain pd in first pass

        [r]  = [δ, 0]
        [f]  = [I, 0] [f]
        [pi] = [d, 0] [pd]
        [pd] = [0, I]

        then we retain a sys of the form:

        [L, L  , 0  ] [f]
        [L, Lpp, Ipd] [f]
        [0, Idp, 0  ] [pd]

        so
        [f]    [I]
        [f]  = [I] [f]
        [pd]   [L]

        this transform from the right will trivialize the middle eq and drop the third column
        after this we only need to reorg eqs to be done;
        just hoist the Idp term into trivialized middle eq, and we are done!

        not any easier to implement tho; can we do this as a single transform?
        can implement a selector for those dual P with zero diag; expression is then something like:

        pd = S I.I δ I.I d f, where the left I.I derives from the Ipd term

        can also perform left-elimination first
        [[L, L, 0], [δ, 0]] [vi]   [fi]
        [[L, L, 0], [I, I]] [vp] = [fp]
        [[0, 0, _], [0, 0]] [vd]   [_]

        [[d, I, 0], [I, 0]] [Pi]   [si] source/sink
        [[0, _, 0], [0, _]] [Pd]   [_]

        can have two selectors; P with and without diagonal term
        can we process both at the same time? kinda hard since they are coupled in I<->I term
        note that Pi can be eliminated just as easy as Pd if all Pd is known;
        can eliminate Pd first, after which Pi I term can be inverted;
        wait; it isnt really an I term; only touches primal boundary, does not cover interior

        right I term is a 'true' I term though, even though spread over rows. but every dual vertex
        has a primal edge/face associated with it in 2d/3d respc.

        remember: goal is to set up generalized laplacian, with arbitrary bcs and arbitrary degree
        still unclear what the correct laplacian and bcs are to derive a multigrid smoother including boundary terms
        note that we dont have the eliminate equations to perform smoothing on our laplacian

        """
        # FIXME: check that each eq to be eliminated has full diag, and no dependence on other elim vars
        rows_retained = np.delete(np.arange(len(self.L)), rows)
        cols_retained = np.delete(np.arange(len(self.R)), cols)

        # setup elimination transformation matrix
        # FIXME: current implementation ignores / assumes zero RHS!
        n_unknowns = len(self.R)
        n_retained = len(cols_retained)
        elim = np.zeros((n_unknowns, n_retained), np.object)

        import pycomplex.sparse
        def get_diag(i, j):
            diag = self.A.block[i, j]
            diag = diag.todia()
            diag = pycomplex.sparse.inv_diag(-diag)
            assert np.all(np.isfinite(diag.data))
            return diag

        for i, sr in enumerate(self.R): # loop over columns of A and rows of elim
            if i in rows:
                # eliminated row; invert relationship
                for j, nr in enumerate(cols_retained):
                    b = self.A.block[i, nr]
                    if b.nnz > 0:
                        elim[i, j] = get_diag(i, i) * b  # identity is retained
                    else:
                        elim[i, j] = pycomplex.sparse.sparse_zeros(b.shape)
            else:
                # this is retained; set identity row
                for j, nr in enumerate(cols_retained):
                    b = self.A.block[i, nr]
                    shape = b.shape
                    if shape[0] == shape[1]:
                        elim[i, j] = scipy.sparse.identity(shape[0])
                    else:
                        elim[i, j] = pycomplex.sparse.sparse_zeros(shape)

        elim = block.SparseBlockMatrix(elim)
        return self.copy(
            A=(self.A * elim)[rows_retained, :],
            rhs=self.rhs[rows_retained],
            L=self.L[rows_retained],
            R=self.R[cols_retained],
        )

    def normal(self):
        """Form normal equations

        Returns
        -------
        System
            normal equations belonging to self
        """
        AT = self.A.transpose()
        return self.copy(
            A=AT * self.A,
            rhs=AT * self.rhs,
            R=self.R,
            L=self.R,   # NOTE: this is the crux of forming normal equations
        )

    def balance(self, reg=0.0):
        """Divide each row by l1 norm through left-premultiplication"""
        bal = 1.0 / (self.A.norm_l1(axis=1) + reg)
        bal = block.SparseBlockMatrix.as_diagonal(bal)
        return self.copy(
            A=bal * self.A,
            rhs=bal * self.rhs,
        )

    def solve_minres(self, tol=1e-12, M=None):
        A = self.A.merge()
        rhs = self.rhs.merge()
        x = scipy.sparse.linalg.minres(A, rhs, tol=tol, M=M)[0]
        r = A * x - rhs
        return self.allocate_x().split(x), self.rhs.split(r)


# def sparse_diag(diag):
#     s = len(diag)
#     i = np.arange(s, dtype=index_dtype)
#     return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))
#
# def sparse_zeros(shape):
#     q = np.zeros(0, dtype=sign_dtype)
#     return scipy.sparse.coo_matrix((q, (q, q)), shape=shape)
#
#
# class BlockSystem(object):
#     """Blocked linear system;
#
#     system * unknowns = knowns
#
#     Todo
#     ----
#     Add boundary condition terms; that would amount to a hierarchical nesting of these block classes
#
#     make seperate classes for dense block vectors?
#
#     need principled approach to doing things like elimination of equations, or preconditioning.
#     left/right multiply by block system, while retaining reference to parent?
#     if we split unknowns in unknowns to be kept and to be eliminated,
#     this is a valid split if all eliminated unknowns appear only on the diagonal,
#     and have only retained unknowns off-diagonal
#     in case of typical laplace-beltrami;
#     [I, a, 0] [a]   [0]
#     [c, 0, b] [I] = [ca + bd]
#     [0, d, I] [d]   [0]
#     unknowns which satisfy this condition will indeed trivialize after that transformation
#     cant do this for something like stokes; could do normal equations but get three laplacians
#     can we do a hybrid of these two and get two laplacians for stokes?
#     [I, a, 0] [a, 0]   [0]
#     [c, 0, b] [I, 0] = [ca + bd, b]
#     [0, d, 0] [0, I]   [d, 0]
#
#     [I, 0] [L, b] = [ca + bd, b]
#     [d, I] [d, 0]   [dbd, db]
#
#     Notes
#     -----
#     Make this class focuss on the blocking-aspects
#     seperate the solving and related concerns
#     Generalized linear equation class should be able to use these blocks as a drop-in replacement
#     for normal sparse operations
#
#     """
#
#     def __init__(self, equations, knowns, unknowns):
#
#         self.shape = len(knowns), len(unknowns)
#         self.equations = np.asarray(equations, dtype=np.object)
#         self.knowns = knowns
#         self.unknowns = unknowns
#         if not self.equations.shape == self.shape:
#             raise ValueError
#
#         self.rows = [self.equations[i, 0].shape[0] for i in range(self.shape[0])]
#         self.cols = [self.equations[0, i].shape[1] for i in range(self.shape[1])]
#
#
#         # unknown_shape = [row[0].shape[0] for row in self.system]
#         # unknown_shape = [row[0].shape[1] for row in self.system.T]
#
#         # check that subblocks are consistent
#
#     def symmetrize(self):
#         """symmetrize systems with a structural symmetry, by rewriting boundary terms"""
#         raise NotImplementedError
#
#     def preconditioned_normal_equations(self):
#         # FIXME: logic here is broken; we need a factor for each equation if we want to contract it inside the normal equations
#         diag = self.normal_equations().diag()
#         diag = [scipy.sparse.diags(1 / d) for d in diag]
#         return self.normal_equations(diag=diag)
#
#     def normal_equations(self):
#         """Formulate normal equations by premultiplication of self with self.transpose
#
#         Returns
#         -------
#         BlockSystem
#         """
#         output_shape = self.equations.shape[1], self.equations.shape[1]
#         S = self.equations
#         equations = np.zeros(output_shape, dtype=np.object)
#         # knowns = np.zeros(len(self.unknowns), dtype=np.object)
#         knowns = [0] * len(self.unknowns)
#         for i in range(self.equations.shape[0]):
#             for j in range(self.equations.shape[1]):
#                 for k in range(self.equations.shape[1]):
#                     equations[j, k] = equations[j, k] + S[i, j].T * S[i, k]
#
#                 knowns[j] = knowns[j] + S[i, j].T * self.knowns[i]
#         return BlockSystem(equations=equations, knowns=knowns, unknowns=list(self.unknowns))
#
#     def concatenate(self):
#         """Concatenate blocks into single system"""
#         return scipy.sparse.bmat(self.equations), scipy.hstack(self.knowns)
#
#     def split(self, x, axis='cols'):
#         """Split concatted vector into blocks
#
#         Parameters
#         ----------
#         x : ndarray, [n_cols], float
#
#         Returns
#         -------
#         list of ndarray, float
#         """
#         splits = [0] + list(np.cumsum(self.cols if axis == 'cols' else self.rows))
#         return [x[s:e] for s, e in zip(splits[:-1], splits[1:])]
#
#     def print_equations(self):
#         for i in range(self.equations.shape[0]):
#             for j in range(self.equations.shape[1]):
#                 f = self.equations[i, j]
#                 shape = getattr(f, 'shape', None)
#                 # print(shape, end='')
#             print()
#
#     def plot(self, dense=True, order=None):
#         """Visualize the structure of the linear system
#
#         Parameters
#         ----------
#         dense : bool
#             dense looks better; but only feasible for small systems
#         """
#         import matplotlib.pyplot as plt
#         S, _ = self.concatenate()
#         if not dense:
#             s = S.tocoo()
#             plt.scatter(S.col, S.row, c=S.data, cmap='seismic', vmin=-4, vmax=+4)
#             plt.gca().invert_yaxis()
#         else:
#             S = S.todense()
#             if order is not None:
#                 S = S[order, :][:, order]
#             cmap = 'bwr'#''seismic'
#             plt.imshow(S[::-1], cmap=cmap, vmin=-1, vmax=+1, origin='upper', extent=(0, S.shape[1], 0, S.shape[0]))
#             # plot block boundaries
#             for l in np.cumsum([0] + self.rows):
#                 plt.plot([0, S.shape[1]], [l, l], c='k')
#             for l in np.cumsum([0] + self.cols):
#                 plt.plot([l, l], [0, S.shape[0]], c='k')
#             plt.gca().invert_yaxis()
#
#         plt.axis('equal')
#         plt.show()
#
#     def solve_direct(self):
#         equations, knowns = self.concatenate()
#         x = np.linalg.solve(equations.todense(), knowns)
#         r = equations * x - knowns
#         return self.split(x), r
#
#     def solve_minres(self, tol=1e-12, M=None):
#         equations, knowns = self.concatenate()
#         x = scipy.sparse.linalg.minres(equations, knowns, tol=tol, M=M)[0]
#         # x = scipy.sparse.linalg.minres(equations, knowns, x0=x, tol=1e-10)[0]
#         r = equations * x - knowns
#         return self.split(x), self.split(r, axis='rows')
#
#     def solve_minres_amg(self, tol=1e-12):
#         """Sole using minres and amg preconditioner"""
#         print('minres+amg solve')
#         from pyamg import smoothed_aggregation_solver, ruge_stuben_solver, rootnode_solver
#         equations, knowns = self.concatenate()
#         equations = equations.tocsr()
#
#         from time import clock
#         t = clock()
#         # options = []
#         # options.append(('symmetric', {'theta': 0.0}))
#         # options.append(('symmetric', {'theta': 0.25}))
#         # options.append(('evolution', {'epsilon': 4.0}))
#         # options.append(('algebraic_distance', {'theta': 1e-1, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
#         # options.append(('algebraic_distance', {'theta': 1e-2, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
#         # options.append(('algebraic_distance', {'theta': 1e-3, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
#         # options.append(('algebraic_distance', {'theta': 1e-4, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
#
#         M = smoothed_aggregation_solver(equations, smooth='jacobi').aspreconditioner()
#         # M = rootnode_solver(equations).aspreconditioner()
#         print('amg setup', clock() - t)
#         # x = M.solve(knowns, accel='minres', tol=tol)
#         # x = M.solve(knowns, x0=x, accel='minres', tol=tol)
#
#
#         x = scipy.sparse.linalg.minres(equations, knowns, tol=tol, M=M)[0]
#         r = equations * x - knowns
#         return self.split(x), self.split(r, axis='rows')
#
#     def solve_amg(self, tol=1e-12):
#         """Sole using amg"""
#         print('amg solve')
#         from pyamg import smoothed_aggregation_solver, ruge_stuben_solver, rootnode_solver
#         equations, knowns = self.concatenate()
#         equations = equations.tocsr()
#
#         from time import clock
#         t = clock()
#
#         # FIXME: add these
#         null = None
#
#         M = smoothed_aggregation_solver(equations, B=null, smooth='jacobi')
#         # M = rootnode_solver(equations).aspreconditioner()
#         print('amg setup', clock() - t)
#
#         x = M.solve(knowns, tol=tol)
#         r = equations * x - knowns
#         return self.split(x), self.split(r, axis='rows')
#
#
#     def solve_least_squares(self):
#         """Solve equations in a least-squares sense
#
#         Notes
#         -----
#         This is conceptually similar to minres on normal equations
#         """
#         equations, knowns = self.concatenate()
#         # x = scipy.sparse.linalg.lsqr(equations, knowns, atol=1e-16, btol=1e-16)[0]
#         x = scipy.sparse.linalg.lsqr(equations, knowns, atol=0, btol=0, conlim=0, damp=1)[0]
#         r = equations * x - knowns
#         return self.split(x), r
#
#     def diag(self):
#         """Get the diagonal of the block system
#
#         Returns
#         -------
#         list of array
#         """
#         return [self.equations[i, i].diagonal() for i in range(self.shape[0])]
#
#     def precondition(self):
#         """Diagonal preconditioner that maintains symmetry and maps the diagonal to unity
#
#         Returns
#         -------
#         BlockSystem
#             preconditioned
#         """
#         diag = [scipy.sparse.diags(1 / np.sqrt(d)) for d in self.diag()]
#         equations = [
#             [diag[i] * self.equations[i, j] * diag[j]
#                 for j in range(self.shape[1])]
#                     for i in range(self.shape[0])]
#         knowns = [diag[i] * self.knowns[i] for i in range(self.shape[0])]
#         return BlockSystem(equations=equations, knowns=knowns, unknowns=self.unknowns)
#
#     def norm_l1(self):
#         """Return the l1 norm of the block, summing over columns"""
#         def norm(A):
#             return np.array(np.abs(A).sum(axis=1)).flatten()
#         l1s = [
#             sum([norm(self.equations[i, j])
#                 for j in range(self.shape[1])])
#                     for i in range(self.shape[0])]
#         return l1s
#
#     def balance(self, reg=0):
#         """Divide each row by l1 norm by left-premultiplication"""
#         l1s = [scipy.sparse.diags(reg / (l + reg)) for l in self.norm_l1()]
#         equations = [
#             [l1s[i] * self.equations[i, j]
#                 for j in range(self.shape[1])]
#                     for i in range(self.shape[0])]
#         knowns = [l1s[i] * self.knowns[i] for i in range(self.shape[0])]
#         return BlockSystem(equations=equations, knowns=knowns, unknowns=self.unknowns)
#
#
# def d_matrix(chain, shape, O, rows=None):
#     """Dual boundary term"""
#     if rows is None:
#         rows = np.arange(len(chain), dtype=index_dtype)
#     else:
#         rows = np.ones(len(chain), dtype=index_dtype) * rows
#     cols = np.arange(len(chain), dtype=index_dtype) + O
#     return scipy.sparse.csr_matrix((
#         chain.astype(np.float),
#         (rows, cols)),
#         shape=shape
#     )
#
#
# def o_matrix(chain, cols, shape, rows=None):
#     """Primal boundary term"""
#     if rows is None:
#         rows = np.arange(len(chain), dtype=index_dtype)
#     else:
#         rows = np.ones(len(chain), dtype=index_dtype) * rows
#     return scipy.sparse.coo_matrix((
#         chain.astype(np.float),
#         (rows, cols)),
#         shape=shape
#     )