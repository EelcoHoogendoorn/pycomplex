import numpy
import numpy as np
import scipy.sparse


def sparse_diag(diag):
    s = len(diag)
    i = np.arange(s)
    return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))

def sparse_zeros(shape):
    q = np.zeros(0)
    return scipy.sparse.coo_matrix((q, (q, q)), shape=shape)


class BlockSystem(object):
    """Blocked linear system;

    system * unknowns = knowns

    Todo
    ----
    Add boundary condition terms

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

    def preconditioned_normal_equations(self):
        # FIXME: logic here is broken; we need a factor for each equation if we want to contract it inside the normal equations
        diag = self.normal_equations().diag()
        diag = [scipy.sparse.diags(1 / d) for d in diag]
        return self.normal_equations(diag=diag)

    def normal_equations(self):
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

    def split(self, x):
        """Split concatted vector into blocks

        Parameters
        ----------
        x : ndarray, [n_cols], float

        """
        splits = [0] + list(np.cumsum(self.cols))
        return [x[s:e] for s, e in zip(splits[:-1], splits[1:])]

    def print(self):
        for i in range(self.equations.shape[0]):
            for j in range(self.equations.shape[1]):
                f = self.equations[i, j]
                shape = getattr(f, 'shape', None)
                print(shape, end='')
            print()

    def plot(self, dense=True):
        import matplotlib.pyplot as plt
        S, _ = self.concatenate()
        if not dense:
            s = S.tocoo()
            plt.scatter(S.col, S.row, c=S.data, cmap='seismic', vmin=-4, vmax=+4)
            plt.gca().invert_yaxis()
        else:
            S = S.todense()
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

    def solve_minres(self, tol=1e-12):
        equations, knowns = self.concatenate()
        x = scipy.sparse.linalg.minres(equations, knowns, tol=tol)[0]
        # x = scipy.sparse.linalg.minres(equations, knowns, x0=x, tol=1e-10)[0]
        r = equations * x - knowns
        return self.split(x), r

    def solve_least_squares(self):
        equations, knowns = self.concatenate()
        # x = scipy.sparse.linalg.lsqr(equations, knowns, atol=1e-16, btol=1e-16)[0]
        x = scipy.sparse.linalg.lsqr(equations, knowns, atol=0, btol=0, conlim=0, damp=1)[0]
        r = equations * x - knowns
        return self.split(x), r

    def diag(self):
        return [self.equations[i, i].diagonal() for i in range(self.shape[0])]

    def precondition(self):
        """Diagonal preconditioner that maintains symmetry and maps the diagonal to unity"""
        diag = [scipy.sparse.diags(1 / np.sqrt(d)) for d in self.diag()]
        equations = [
            [diag[i] * self.equations[i, j] * diag[j]
                for j in range(self.shape[1])]
                    for i in range(self.shape[0])]
        knowns = [diag[i] * self.knowns[i] for i in range(self.shape[0])]
        return BlockSystem(equations=equations, knowns=knowns, unknowns=self.unknowns)


def d_matrix(idx, shape, O):
    return scipy.sparse.csr_matrix((
        idx.astype(np.float),
        (np.arange(len(idx)), np.arange(len(idx)) + O)),
        shape=shape
    )


def o_matrix(v, col, shape):
    return scipy.sparse.coo_matrix((
        v.astype(np.float),
        (np.arange(len(v)), col)),
        shape=shape
    )