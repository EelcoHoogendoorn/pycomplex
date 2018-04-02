import numpy
import numpy as np
import scipy.sparse
from pycomplex.topology import sign_dtype, index_dtype

from cached_property import cached_property
from pycomplex import block


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
        """
        # make sure we are not engaging in nonsense here
        complex.topology.check_chain()
        complex.topology.dual.check_chain()

        T = complex.topology.dual.matrices_2[::-1]
        N = complex.n_dim + 1

        NE = complex.topology.dual.n_elements[::-1]
        A = block.SparseBlockMatrix(
            [[sparse_zeros((NE[i], NE[j])) for j in range(N)] for i in range(N)])
        S = complex.topology.dual.selector
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

        def d_matrix(chain, shape, offset, rows=None):
            """Dual boundary term"""
            if rows is None:
                rows = np.arange(len(chain), dtype=index_dtype) + offset
            else:
                rows = np.ones(len(chain), dtype=index_dtype) * rows + offset
            cols = np.arange(len(chain), dtype=index_dtype) + offset
            return scipy.sparse.csr_matrix((
                chain.astype(np.float),
                (rows, cols)),
                shape=shape
            )
        self.A.block[i, j] = self.A.block[i, j] + d_matrix(
            d, self.A.block[i, j].shape, self.complex.topology.n_elements[self.L[i]])

    def set_off_boundary(self, i, j, o):
        """Set a boundary on a 'off-diagonal' term of the cochain complex

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
        # FIXME: need to get hodge involved?

        assert self.L[i] == self.R[j] + 1   # this implies entry below the diagonal

        def o_matrix(chain, cols, shape, offset, rows=None):
            """Primal boundary term"""
            if rows is None:
                rows = np.arange(len(chain), dtype=index_dtype) + offset
            else:
                rows = np.ones(len(chain), dtype=index_dtype) * rows + offset
            return scipy.sparse.coo_matrix((
                chain.astype(np.float),
                (rows, cols)),
                shape=shape
            )

        # idx that primal boundary connects to
        idx = self.complex.topology.boundary.parent_idx[self.R[j]]

        self.A.block[i, j] = self.A.block[i, j] + o_matrix(
            o, idx, self.A.block[i, j].shape, self.complex.topology.n_elements[self.L[i]])

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
        # boundary is on last rows of the block; shift down by all primal elements
        # FIXME: we should expand the concept of the selector matrices to handle things like addressing the dual boundary
        offset = self.complex.topology.n_elements[self.L[i]]
        self.rhs.block[i] = self.rhs.block[i] + np.concatenate([np.zeros(offset), r])

    def set_rhs(self, i, r):
        S = self.complex.topology.dual.selector[self.L[i]]
        self.rhs.block[i] = self.rhs.block[i] + S * r


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

    def eliminate(self, rows):
        """eliminate rows from the block with nonzero diagonal

        Parameters
        ----------
        rows : List[int]
            rows to be eliminated

        Returns
        -------
        System
            equivalent system with rows eliminated
        """

    def normal(self):
        """Form normal equations

        Returns
        -------
        System
            normal equations belonging to self
        """
        AT = self.A.transpose()
        return System(
            complex=self.complex,
            A=AT * self.A,
            rhs=AT * self.rhs,
            R=self.R,
            L=self.R,   # NOTE: this is the crux of forming normal equations
        )

    def balance(self, reg=0.0):
        """Divide each row by l1 norm through left-premultiplication"""
        bal = 1.0 / (self.A.norm_l1(axis=1) + reg)
        bal = block.SparseBlockMatrix.as_diagonal(bal)
        return System(
            complex=self.complex,
            A=bal * self.A,
            rhs=bal * self.rhs,
            L=self.L,
            R=self.R
        )

    def solve_minres(self, tol=1e-12, M=None):
        A = self.A.merge()
        rhs = self.rhs.merge()
        x = scipy.sparse.linalg.minres(A, rhs, tol=tol, M=M)[0]
        r = A * x - rhs
        return self.allocate_x().split(x), self.rhs.split(r)


def sparse_diag(diag):
    s = len(diag)
    i = np.arange(s, dtype=index_dtype)
    return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))

def sparse_zeros(shape):
    q = np.zeros(0, dtype=sign_dtype)
    return scipy.sparse.coo_matrix((q, (q, q)), shape=shape)


class BlockSystem(object):
    """Blocked linear system;

    system * unknowns = knowns

    Todo
    ----
    Add boundary condition terms; that would amount to a hierarchical nesting of these block classes

    make seperate classes for dense block vectors?

    need principled approach to doing things like elimination of equations, or preconditioning.
    left/right multiply by block system, while retaining reference to parent?
    if we split unknowns in unknowns to be kept and to be eliminated,
    this is a valid split if all eliminated unknowns appear only on the diagonal,
    and have only retained unknowns off-diagonal
    in case of typical laplace-beltrami;
    [I, a, 0] [a]   [0]
    [c, 0, b] [I] = [ca + bd]
    [0, d, I] [d]   [0]
    unknowns which satisfy this condition will indeed trivialize after that transformation
    cant do this for something like stokes; could do normal equations but get three laplacians
    can we do a hybrid of these two and get two laplacians for stokes?
    [I, a, 0] [a, 0]   [0]
    [c, 0, b] [I, 0] = [ca + bd, b]
    [0, d, 0] [0, I]   [d, 0]

    [I, 0] [L, b] = [ca + bd, b]
    [d, I] [d, 0]   [dbd, db]

    Notes
    -----
    Make this class focuss on the blocking-aspects
    seperate the solving and related concerns
    Generalized linear equation class should be able to use these blocks as a drop-in replacement
    for normal sparse operations

    """

    def __init__(self, equations, knowns, unknowns):

        self.shape = len(knowns), len(unknowns)
        self.equations = np.asarray(equations, dtype=np.object)
        self.knowns = knowns
        self.unknowns = unknowns
        if not self.equations.shape == self.shape:
            raise ValueError

        self.rows = [self.equations[i, 0].shape[0] for i in range(self.shape[0])]
        self.cols = [self.equations[0, i].shape[1] for i in range(self.shape[1])]


        # unknown_shape = [row[0].shape[0] for row in self.system]
        # unknown_shape = [row[0].shape[1] for row in self.system.T]

        # check that subblocks are consistent

    def symmetrize(self):
        """symmetrize systems with a structural symmetry, by rewriting boundary terms"""
        raise NotImplementedError

    def preconditioned_normal_equations(self):
        # FIXME: logic here is broken; we need a factor for each equation if we want to contract it inside the normal equations
        diag = self.normal_equations().diag()
        diag = [scipy.sparse.diags(1 / d) for d in diag]
        return self.normal_equations(diag=diag)

    def normal_equations(self):
        """Formulate normal equations by premultiplication of self with self.transpose

        Returns
        -------
        BlockSystem
        """
        output_shape = self.equations.shape[1], self.equations.shape[1]
        S = self.equations
        equations = np.zeros(output_shape, dtype=np.object)
        # knowns = np.zeros(len(self.unknowns), dtype=np.object)
        knowns = [0] * len(self.unknowns)
        for i in range(self.equations.shape[0]):
            for j in range(self.equations.shape[1]):
                for k in range(self.equations.shape[1]):
                    equations[j, k] = equations[j, k] + S[i, j].T * S[i, k]

                knowns[j] = knowns[j] + S[i, j].T * self.knowns[i]
        return BlockSystem(equations=equations, knowns=knowns, unknowns=list(self.unknowns))

    def concatenate(self):
        """Concatenate blocks into single system"""
        return scipy.sparse.bmat(self.equations), scipy.hstack(self.knowns)

    def split(self, x, axis='cols'):
        """Split concatted vector into blocks

        Parameters
        ----------
        x : ndarray, [n_cols], float

        Returns
        -------
        list of ndarray, float
        """
        splits = [0] + list(np.cumsum(self.cols if axis == 'cols' else self.rows))
        return [x[s:e] for s, e in zip(splits[:-1], splits[1:])]

    def print_equations(self):
        for i in range(self.equations.shape[0]):
            for j in range(self.equations.shape[1]):
                f = self.equations[i, j]
                shape = getattr(f, 'shape', None)
                # print(shape, end='')
            print()

    def plot(self, dense=True, order=None):
        """Visualize the structure of the linear system

        Parameters
        ----------
        dense : bool
            dense looks better; but only feasible for small systems
        """
        import matplotlib.pyplot as plt
        S, _ = self.concatenate()
        if not dense:
            s = S.tocoo()
            plt.scatter(S.col, S.row, c=S.data, cmap='seismic', vmin=-4, vmax=+4)
            plt.gca().invert_yaxis()
        else:
            S = S.todense()
            if order is not None:
                S = S[order, :][:, order]
            cmap = 'bwr'#''seismic'
            plt.imshow(S[::-1], cmap=cmap, vmin=-1, vmax=+1, origin='upper', extent=(0, S.shape[1], 0, S.shape[0]))
            # plot block boundaries
            for l in np.cumsum([0] + self.rows):
                plt.plot([0, S.shape[1]], [l, l], c='k')
            for l in np.cumsum([0] + self.cols):
                plt.plot([l, l], [0, S.shape[0]], c='k')
            plt.gca().invert_yaxis()

        plt.axis('equal')
        plt.show()

    def solve_direct(self):
        equations, knowns = self.concatenate()
        x = np.linalg.solve(equations.todense(), knowns)
        r = equations * x - knowns
        return self.split(x), r

    def solve_minres(self, tol=1e-12, M=None):
        equations, knowns = self.concatenate()
        x = scipy.sparse.linalg.minres(equations, knowns, tol=tol, M=M)[0]
        # x = scipy.sparse.linalg.minres(equations, knowns, x0=x, tol=1e-10)[0]
        r = equations * x - knowns
        return self.split(x), self.split(r, axis='rows')

    def solve_minres_amg(self, tol=1e-12):
        """Sole using minres and amg preconditioner"""
        print('minres+amg solve')
        from pyamg import smoothed_aggregation_solver, ruge_stuben_solver, rootnode_solver
        equations, knowns = self.concatenate()
        equations = equations.tocsr()

        from time import clock
        t = clock()
        # options = []
        # options.append(('symmetric', {'theta': 0.0}))
        # options.append(('symmetric', {'theta': 0.25}))
        # options.append(('evolution', {'epsilon': 4.0}))
        # options.append(('algebraic_distance', {'theta': 1e-1, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
        # options.append(('algebraic_distance', {'theta': 1e-2, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
        # options.append(('algebraic_distance', {'theta': 1e-3, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))
        # options.append(('algebraic_distance', {'theta': 1e-4, 'p': np.inf, 'R': 10, 'alpha': 0.5, 'k': 20}))

        M = smoothed_aggregation_solver(equations, smooth='jacobi').aspreconditioner()
        # M = rootnode_solver(equations).aspreconditioner()
        print('amg setup', clock() - t)
        # x = M.solve(knowns, accel='minres', tol=tol)
        # x = M.solve(knowns, x0=x, accel='minres', tol=tol)


        x = scipy.sparse.linalg.minres(equations, knowns, tol=tol, M=M)[0]
        r = equations * x - knowns
        return self.split(x), self.split(r, axis='rows')

    def solve_amg(self, tol=1e-12):
        """Sole using amg"""
        print('amg solve')
        from pyamg import smoothed_aggregation_solver, ruge_stuben_solver, rootnode_solver
        equations, knowns = self.concatenate()
        equations = equations.tocsr()

        from time import clock
        t = clock()

        # FIXME: add these
        null = None

        M = smoothed_aggregation_solver(equations, B=null, smooth='jacobi')
        # M = rootnode_solver(equations).aspreconditioner()
        print('amg setup', clock() - t)

        x = M.solve(knowns, tol=tol)
        r = equations * x - knowns
        return self.split(x), self.split(r, axis='rows')


    def solve_least_squares(self):
        """Solve equations in a least-squares sense

        Notes
        -----
        This is conceptually similar to minres on normal equations
        """
        equations, knowns = self.concatenate()
        # x = scipy.sparse.linalg.lsqr(equations, knowns, atol=1e-16, btol=1e-16)[0]
        x = scipy.sparse.linalg.lsqr(equations, knowns, atol=0, btol=0, conlim=0, damp=1)[0]
        r = equations * x - knowns
        return self.split(x), r

    def diag(self):
        """Get the diagonal of the block system

        Returns
        -------
        list of array
        """
        return [self.equations[i, i].diagonal() for i in range(self.shape[0])]

    def precondition(self):
        """Diagonal preconditioner that maintains symmetry and maps the diagonal to unity

        Returns
        -------
        BlockSystem
            preconditioned
        """
        diag = [scipy.sparse.diags(1 / np.sqrt(d)) for d in self.diag()]
        equations = [
            [diag[i] * self.equations[i, j] * diag[j]
                for j in range(self.shape[1])]
                    for i in range(self.shape[0])]
        knowns = [diag[i] * self.knowns[i] for i in range(self.shape[0])]
        return BlockSystem(equations=equations, knowns=knowns, unknowns=self.unknowns)

    def norm_l1(self):
        """Return the l1 norm of the block, summing over columns"""
        def norm(A):
            return np.array(np.abs(A).sum(axis=1)).flatten()
        l1s = [
            sum([norm(self.equations[i, j])
                for j in range(self.shape[1])])
                    for i in range(self.shape[0])]
        return l1s

    def balance(self, reg=0):
        """Divide each row by l1 norm by left-premultiplication"""
        l1s = [scipy.sparse.diags(reg / (l + reg)) for l in self.norm_l1()]
        equations = [
            [l1s[i] * self.equations[i, j]
                for j in range(self.shape[1])]
                    for i in range(self.shape[0])]
        knowns = [l1s[i] * self.knowns[i] for i in range(self.shape[0])]
        return BlockSystem(equations=equations, knowns=knowns, unknowns=self.unknowns)


def d_matrix(chain, shape, O, rows=None):
    """Dual boundary term"""
    if rows is None:
        rows = np.arange(len(chain), dtype=index_dtype)
    else:
        rows = np.ones(len(chain), dtype=index_dtype) * rows
    cols = np.arange(len(chain), dtype=index_dtype) + O
    return scipy.sparse.csr_matrix((
        chain.astype(np.float),
        (rows, cols)),
        shape=shape
    )


def o_matrix(chain, cols, shape, rows=None):
    """Primal boundary term"""
    if rows is None:
        rows = np.arange(len(chain), dtype=index_dtype)
    else:
        rows = np.ones(len(chain), dtype=index_dtype) * rows
    return scipy.sparse.coo_matrix((
        chain.astype(np.float),
        (rows, cols)),
        shape=shape
    )