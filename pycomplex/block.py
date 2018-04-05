"""Blocked arrays and linear algebra

This is usefull for managing structure in our DEC equations;
briding the gap between the abstract linear algebra perspective and more structured DEC operators

Note that it is important that these blocked matrices behave much the same, or have the same interface,
as their unblocked counterparts

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

def cochain(complex):
    """construct full cochain complex"""
    return SparseBlockMatrix(complex.topology.matrices)


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
        # FIXME: we wish to convertnested lists but not merge in leaf level arrays. how to enforce this?
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
        # FIXME: can we also do a form of nested slicing?
        return type(self)(self.block.__getitem__(slc))

    def broadcasting_operator(self, other, op):
        try:
            A = self.block
        except:
            A = self
        try:
            B = other.block
        except:
            B = other

        A, B = np.broadcast_arrays(A, B)
        block = [op(a, b) for a, b in zip(A.flatten(), B.flatten())]
        block = np.asarray(block, np.object).reshape(A.shape)
        return type(other)(block)

    def __add__(self, other):
        """Add two block matrices"""
        return self.broadcasting_operator(other, lambda a, b: a + b)

    def __mul__(self, other):
        """pointwise multiply two block matrices"""
        return self.broadcasting_operator(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """pointwise multiply two block matrices"""
        return self.broadcasting_operator(other, lambda a, b: a / b)
    def __rtruediv__(self, other):
        return self.broadcasting_operator(other, lambda a, b: b / a)


    @staticmethod
    def einsum(operands, formula):
        """Generic multilinear product over blocks

        Notes
        -----
        swapping indices should recurse down the block level
        if we cant use this function to transpose correctly, it is of little use
        """
        import itertools
        import functools

        # parse formula
        left, right = formula.split('->')
        spec = left.split(',')
        axes_labels = list(set(formula) - set(',->'))

        # characterize axes
        axis_size = {}
        for o, o_axes in zip(operands, spec):
            for a, s in zip(o_axes, o.block.shape):
                if a in axis_size:
                    assert axis_size[a] == s
                else:
                    axis_size[a] = s

        # allocate output
        output_shape = [axis_size[a] for a in right]
        output = np.zeros(output_shape, dtype=np.object)

        # loop over all axes of the product
        for idx in itertools.product(*[range(axis_size[a]) for a in axes_labels]):
            # this is a operation per subblock. need to select the subblock, and then swivel the axes
            # could recurse into einsum for dense blocks?
            # need to add similar logic for sparse then too
            try:
                output = output + np.einsum(operands, formula)
            except:
                output = output + functools.reduce(operands, lambda x, y: x * y)

        return type(operands[-1])(output)

    def copy(self):
        return type(self)(self.block.copy())


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
        # assert self.square
        return DenseBlockArray([self.block[i, i].diagonal() for i in range(self.rows)])

    @staticmethod
    def as_diagonal(diag):
        assert diag.ndim == 1
        block = np.zeros((diag.block.shape[0],)*2, dtype=np.object)
        for i, a in enumerate(diag.block):
            for j, b in enumerate(diag.block):
                if i == j:
                    block[i, i] = scipy.sparse.diags(b)
                else:
                    block[i, j] = sparse_zeros((len(a), len(b)))
        return SparseBlockMatrix(block)

    def norm_l1(self, axis):
        """Return the l1 norm of the block, summing over the given axis"""
        # FIXME: generalize to nd and other norms?
        if axis == 0:
            return self.transpose().norm_l1(axis=1)
        def norm(A):
            return np.array(np.abs(A).sum(axis=axis)).flatten()
        l1s = [
            sum([norm(self.block[i, j])
                for j in range(self.block.shape[1])])
                    for i in range(self.block.shape[0])]
        return DenseBlockArray(l1s)

    def identity_like(self):
        return


class SparseBlockMatrix(BlockMatrix):
    """This wraps a block of scipy.sparse matrices"""

    def __mul__(self, other):
        """Compute self[i,j] * other[j,...] -> output[i,...]
        Contract self and other along their inner dimension

        Parameters
        ----------
        other : BlockArray

        Returns
        -------
        type(other)
        """
        assert self.block.shape[1] == other.block.shape[0]
        output_shape = self.rows, np.prod(other.block.shape[1:], dtype=np.int)
        other_block = other.block.reshape(other.block.shape[0], output_shape[1])
        output = np.zeros(output_shape, dtype=np.object)
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(output_shape[1]):
                    output[i, k] = output[i, k] + self.block[i, j] * other_block[j, k]

        return type(other)(output.reshape(self.rows, *other.block.shape[1:]))

    def split(self, x):
        """"""
        # Figure out best way to slice arbitrary sparse formats
        raise NotImplementedError

    def merge(self):
        if self.nested:
            raise NotImplementedError
        return scipy.sparse.bmat(self.block)

    @staticmethod
    def diags(diag):
        """Turn dense block vector into diagonal block structure"""
        assert diag.ndim == 1
        output_shape = diag.block.shape * 2
        output = np.zeros(output_shape, dtype=np.object)
        for i, b in enumerate(diag.block):
            output[i, i] = scipy.sparse.diags(b)
        return SparseBlockMatrix(output)



class DenseBlockArray(BlockArray):
    """Dense array with block structure

    usefull for n-forms
    """

    def split(self, x):
        """Split non-blocked array x according to the pattern of self"""
        # splits along all axes
        splits = [[0] + list(np.cumsum(axis)) for axis in self.shape]
        def slices(coords):
            return tuple([slice(s[c], s[c+1]) for c, s in zip(coords, splits)])

        coords = np.indices(self.block.shape)
        blocks = [x[slices(c)] for c in coords.reshape(self.ndim, -1).T]
        blockss = np.empty(len(blocks), np.object)
        blockss[...] = blocks
        blocks = blockss.reshape(self.block.shape)
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
    # N = S.einsum((S, S), 'ji,jk->ik')
    N = S.transpose() * S
    # print(N.square)
    print(N.merge().todense())
    print(S[:, 1:].merge().todense())
    v = DenseBlockArray([np.arange(8), np.arange(3)])
    v = DenseBlockArray([[np.arange(16).reshape(8,2)], [np.arange(6).reshape(3, 2)]])
    q = S * v
    print(q.merge())

    print(S.norm_l1(axis=1).merge())
    print()



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
    # test_vec()