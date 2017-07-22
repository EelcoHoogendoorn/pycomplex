
import numpy as np
import scipy.sparse


def sparse_diag(diag):
    s = len(diag)
    i = np.arange(s)
    return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))


class BlockSystem(object):
    """Blocked linear system"""

    def __init__(self, system, rhs, unknowns):

        self.system = np.asarray(system, dtype=np.object)
        unknown_shape = [row[0].shape[0] for row in self.system]
        unknown_shape = [row[0].shape[1] for row in self.system.T]

        # check that subblocks are consistent
        self.rhs = rhs
        self.unknowns = unknowns

    @property
    def shape(self):
        return self.system.shape

    def normal_equations(self):
        output_shape = self.system.shape[1], self.system.shape[1]
        S = self.system
        output = np.zeros(output_shape, dtype=np.object)
        for i in range(self.system.shape[0]):
            for j in range(self.system.shape[1]):
                for k in range(self.system.shape[1]):
                    output[i, k] += S[i, j] * S[j, k]
        return output

    def concatenate(self):
        """Concatenate blocks into single system"""
        raise NotImplementedError

    def split(self, x):
        """Split concatted vector into blocks

        Parameters
        ----------
        x : ndarray, [n_cols], float

        """
        raise NotImplementedError
