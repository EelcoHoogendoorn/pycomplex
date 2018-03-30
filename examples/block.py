"""Blocked linear algebra

This is usefull for managing structure in our DEC equations;
briding the gap between the abstract linear algebra perspective and more structured DEC operators

Note that it is important that these blocked matrices behave much the same, or have the same interface,
as their unblocked couterparts

Should be able to use this in low level topology classes as well;
maintain seperation interior/primal/dual variables, and so on

Would like these objects to work with at least a single level of nesting;
a blocked cochain complex having i/p/d substructure, for instance

should blocked arrays have a uniform dtype? think not

Note that the block itself is always dense in this particular use case
this is as opposed to scipy.sparse.bsr_matrix, which gets its sparsity from the top level structure
"""

import numpy as np
import scipy.sparse

from pycomplex.sparse import sparse_zeros
from cached_property import cached_property


def pairs(iterable):
    """Yield pairs of an iterator"""
    it = iter(iterable)
    x = next(it)
    while True:
        p = x
        x = next(it)
        yield p, x


class BlockArray(object):
    """ndim-blocked array object"""
    def __init__(self, block):
        self.block = np.asarray(block, dtype=np.object)
        # assert self.compatible

    @cached_property
    def ndim(self):
        return self.block.ndim

    @cached_property
    def compatible(self):
        """Check that each block is consistent with the overall shape"""
        self.shape
        return

    @cached_property
    def shape(self):
        """shape object

        one list for each axis
        """
        if self.nested:
            # FIXME
            raise NotImplementedError
        return [[self.block.take(i, axis=n).shape[n] for i in range(self.block.shape[n])] for n in range(self.ndim)]

    @cached_property
    def nested(self):
        """True if any of the elements of self.block is itself a blocked matrix"""
        return any(isinstance(b, BlockMatrix) for b in self.block.flatten())

    def split(self, merged):
        """split a merged array according to the shape of self

        Parameters
        ----------
        merged : array

        Returns
        -------
        type(self)
        """
        raise NotImplementedError

    def merge(self):
        """Merge self into a non blocked format"""
        raise NotImplementedError

    def apply(self, func):
        """Copy of self, with func applied to all blocks"""
        block = [func(b) for b in self.block.flatten()]
        return type(self)(np.asarray(block, np.object).reshape(self.block.shape))

    def transpose(self):
        """Return transpose of self"""

        def t(x):
            try:
                return x.transpose()
            except:
                return np.transpose(x)

        block = [t(b) for b in self.block.T.flatten()]
        return type(self)(np.asarray(block, np.object).reshape(self.block.T.shape))

    def __getitem__(self, slc):
        """Slicing simply passes on to the block"""
        return type(self)(self.block.__getitem__(slc))


class BlockMatrix(BlockArray):
    """Blocked linear algebra

    Notes
    -----
    In this context, everything is always a 2d matrix, like the sparse classes themselves
    """

    @cached_property
    def rows(self):
        return self.block.shape[0]
    @cached_property
    def cols(self):
        return self.block.shape[1]

    @cached_property
    def square(self):
        """Return true if matrix is square"""
        r, c = np.array(self.shape).T
        return np.alltrue(r.T == c)

    def diagonal(self):
        """Get the diagonal of the block

        Returns
        -------
        DenseBlockArray
        """
        assert self.square
        return [self.block[i, i].diagonal() for i in range(self.rows)]


class SparseBlockMatrix(BlockMatrix):
    """This wraps a block of scipy.sparse matrices"""

    def __mul__(self, other):
        """Compute self[i,j] * other[j,k] -> output[i,k]"""
        assert self.cols == other.rows
        output_shape = self.rows, other.cols
        output = np.zeros(output_shape, dtype=np.object)
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.cols):
                    output[i, k] = output[i, k] + self.block[i, j] * other.block[j, k]

        return type(other)(output)

    def split(self, x):
        """"""
        # Figure out best way to slice arbitrary sparse formats
        raise NotImplementedError

    def merge(self):
        if self.nested:
            raise NotImplementedError
        return scipy.sparse.bmat(self.block)


class DenseBlockArray(BlockArray):
    """Dense array with block structure"""
    def split(self, x):
        """Split non-blocked array x according to the pattern of self"""
        # splits along all axes
        splits = [[0] + list(np.cumsum(axis)) for axis in self.shape]
        def slices(coords):
            return tuple([slice(s[c], s[c+1]) for c, s in zip(coords, splits)])

        coords = np.indices(self.block.shape)
        blocks = [x[slices(c)] for c in coords.reshape(self.ndim, -1).T]
        blocks = np.asarray(blocks).reshape(self.block.shape)
        if self.nested:
            # split each subblock
            # FIXME
            raise NotImplementedError
        return DenseBlockArray(blocks)

    # FIXME: rename to concatenate?
    def merge(self):
        if self.nested:
            block = self.apply(lambda b: b.merge()).block
        else:
            block = self.block
        return np.block(block.tolist())

    def norm(self):
        pass


def test_sparse():
    S = SparseBlockMatrix([
        [sparse_zeros((4, 8)), sparse_zeros((4, 3))],
        [sparse_zeros((3, 8)), scipy.sparse.diags([1, 1, 1])],
    ])
    N = S.transpose() * S
    # print(N.square)
    print(N.merge().todense())
    print(S[:, 1:].merge().todense())


def test_vec():
    v = DenseBlockArray([
        np.ones(4), np.arange(6)
    ])
    print(v)
    m = v.transpose().merge()
    print(m)
    q = v.split(m)
    print(q)


if __name__ == '__main__':
    test_sparse()
    test_vec()

