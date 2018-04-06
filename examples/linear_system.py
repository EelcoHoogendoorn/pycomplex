import numpy
import numpy as np
import scipy.sparse
from pycomplex.topology import sign_dtype, index_dtype

from cached_property import cached_property
from pycomplex import block
import pycomplex.sparse


class BaseSystem(object):
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

    def __getitem__(self, item):
        """Slice a subsystem of full cochain complex"""
        return self.copy(
            A=self.A.__getitem__(item).copy(),
            L=self.L[item[0]],
            R=self.R[item[1]],
        )

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
            lim = np.abs(S).max()/2
            plt.imshow(S[::-1], cmap=cmap, vmin=-lim, vmax=+lim, origin='upper', extent=(0, S.shape[1], 0, S.shape[0]))
            # # plot block boundaries
            # for l in np.cumsum([0] + self.rows):
            #     plt.plot([0, S.shape[1]], [l, l], c='k')
            # for l in np.cumsum([0] + self.cols):
            #     plt.plot([l, l], [0, S.shape[0]], c='k')
            plt.gca().invert_yaxis()

        plt.axis('equal')
        plt.show()

    # boundary condition related methods
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

        Sb = self.complex.topology.dual.selector_b[self.L[i]]
        self.A.block[i, j] = self.A.block[i, j] + scipy.sparse.diags(Sb.T * d)

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

    def solve_gmres(self, tol=1e-12, M=None):
        A = self.A.merge()
        rhs = self.rhs.merge()
        x = scipy.sparse.linalg.gmres(A, rhs, tol=tol, M=M)[0]
        r = A * x - rhs
        return self.allocate_x().split(x), self.rhs.split(r)


class System(BaseSystem):
    """Generalized linear system

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

        [-3, -3, +0, +0]
        [-3, -1, -1, +0]
        [+0, -1, +1, +1]
        [+0, +0, +1, +3]

        Is this an argument in favor of working in 'dimensionless midpoint space' by default?
        worked for eschereque
        if doing so, we should not expect refinement to influence the scaling of our equations at all?
        every extterior derivative would be wrapped left and right by metric.
        multiply with k-metric, exterior derivative, and then divide by k+1 metric;
        so every operator scales as 1/l
        may also be more elegant in an mg-transfer scenario?

        note that this problem resolves itself after variable elimination to form laplace
        otoh, absence of symmetry does not resolve itself in midpoint method, after elimination
        is there a standard scaling we can apply, that rebalances this type of system without affecting symmetry?
        left/right multiplication by l^-k factor seems like itd work

        [-3, -4, +0, +0]
        [-4, -3, -4, +0]
        [+0, -4, -3, -4]
        [+0, +0, -4, -3]

        """
        # make sure we are not engaging in nonsense here
        complex.topology.check_chain()
        complex.topology.dual.check_chain()
        complex.topology.boundary.check_chain()

        Tp = complex.topology.matrices
        Td = complex.topology.dual.matrices_2[::-1]
        N = complex.n_dim + 1

        NE = complex.topology.dual.n_elements[::-1]
        A = block.SparseBlockMatrix(
            [[pycomplex.sparse.sparse_zeros((NE[i], NE[j])) for j in range(N)] for i in range(N)])
        S = complex.topology.dual.selector
        # FIXME: make these available as properties on the System? need them more often
        PD = [scipy.sparse.diags(pd) for pd in complex.hodge_PD]
        # these terms are almost universally required
        for i, (tp, td) in enumerate(zip(Tp, Td)):
            A.block[i, i + 1] = S[i].T * PD[i] * td.T
            A.block[i + 1, i] = S[i+1].T * tp.T * PD[i] * S[i]

        # put hodges on diag by default; easier to zero out than to fill in
        for i in range(N):
            A.block[i, i] = S[i].T * PD[i] * S[i]

        LR = np.arange(N, dtype=index_dtype)
        return System(complex, A=A, B=None, L=LR, R=LR)

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

    def block_balance(self, l=0, k=0):
        """Perform block based scaling A=DAD, where D is a diagonal scaling matrix
        intended to give each block comparable weight

        Parameters
        ----------
        l : float, optional
            scale factor; or typical edge length compared to unit scale
            if none is given, the average primal edge length is used
        k : int, optional
            primal k-form that experiences identity transform

        Returns
        -------
        System
            rebalanced system
            retains symmetry, but unknowns are transformed
        """
        if l == 0:
            l = self.complex.primal_metric[1].mean()    # average primal edge length
        k = self.complex.topology.n_dim + 1 + k if k < 0 else k
        scale = l ** -(np.arange(self.complex.topology.n_dim + 1) - k)

        D = block.DenseBlockArray(scale)
        L = D[self.L]
        R = D[self.R]
        return self.copy(
            A=block.SparseBlockMatrix((L[:, None] * R[None, :] * self.A).block),
            rhs=L * self.rhs
        )

    def laplace(self, k):
        """Form laplace-beltrami operator from full system"""
        laplace = self[k-1:k+2, k-1:k+2]
        # FIXME: this fails for scalar laplacian already
        laplace.A.block[1, 1] *= 0

        # FIXME: flexible way to configure bcs? set up so it works for scalar laplace too
        # default boundaries should be those that leave the central k-form untouched
        q = laplace.L[2]
        z = self.complex.topology.dual.selector_b[q].nnz    # FIXME: there should be a cleaner way to compute this
        laplace.set_dia_boundary(2, 2, np.ones(z))
        q = laplace.L[1]
        z = self.complex.topology.dual.selector_b[q].nnz
        laplace.set_off_boundary(1, 0, np.ones(z))

        laplace = laplace.block_balance()

        return laplace


class SystemMid(BaseSystem):

    """Linear system, where both variables and equations are formulated in the space between primal and dual

    This means the resulting linear system is not entirely symmetrical,
    even though it comes close to being so and every operator scales as 1 / l

    """

    @staticmethod
    def canonical(complex):
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

        Notes
        -----
        as a consequence of midpoint logic, equations are naturally balanced
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
        # FIXME: should this be a seperate or subclass? probably many methods in the current system class are tied to assumptions in the canonical method
        # make sure we are not engaging in nonsense here
        complex.topology.check_chain()
        complex.topology.dual.check_chain()

        # these are matrices with dual boundary term dropped; use full and leave this as seperate steo?
        Tp = complex.topology.matrices
        Td = complex.topology.dual.matrices_2[::-1]
        N = complex.n_dim + 1

        NE = complex.topology.dual.n_elements[::-1]
        A = block.SparseBlockMatrix(
            [[pycomplex.sparse.sparse_zeros((NE[i], NE[j])) for j in range(N)] for i in range(N)])
        S = complex.topology.dual.selector

        Mp= complex.primal_metric
        Md = complex.dual_metric_closed[::-1]

        D = scipy.sparse.diags
        for i, (tp, td) in enumerate(zip(Tp, Td)):
            A.block[i, i + 1] = (D(1/Md[i]) * S[i].T * td.T * D(Md[i+1]))
            A.block[i + 1, i] = S[i+1].T * (D(1/Mp[i+1]) * tp.T * D(Mp[i])) * S[i]    # selector zeros out the boundary conditions

        # put identity on diag by default; easier to zero out than to fill in
        for i, ne in enumerate(complex.topology.n_elements):
            A.block[i, i] = S[i].T * scipy.sparse.eye(ne) * S[i]

        LR = np.arange(N, dtype=index_dtype)
        return System(complex, A=A, B=None, L=LR, R=LR)
