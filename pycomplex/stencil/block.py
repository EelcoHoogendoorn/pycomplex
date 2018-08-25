from cached_property import cached_property

import numpy as np


class BlockArray(object):
    """Blocked linear operators"""
    """ndim-blocked array object"""

    def __init__(self, block):
        self.block = np.asarray(block, dtype=np.object)
        # assert self.is_compatible

    @cached_property
    def ndim(self):
        return self.block.ndim

    @cached_property
    def is_compatible(self):
        """Check that each block is consistent with the overall shape"""
        self.shape
        return True

    @cached_property
    def shape(self):
        """shape object

        one list for each axis
        """
        return [[self.block.take(i, axis=n).shape[n] for i in range(self.block.shape[n])] for n in range(self.ndim)]

    def apply(self, func):
        """Copy of self, with func applied to all blocks"""
        block = [func(b) for b in self.block.flatten()]
        return type(self)(np.asarray(block, np.object).reshape(self.block.shape))

    def transpose(self):
        """Return transpose of self"""

        def t(x):
            try:
                return x.T
            except:
                return np.transpose(x)

        block = [t(b) for b in self.block.T.flatten()]
        return type(self)(np.asarray(block, np.object).reshape(self.block.T.shape))

    def __getitem__(self, slc):
        """Slicing simply passes on to the block"""
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
        return type(self)(block)

    def __add__(self, other):
        """Add two block operators"""
        return self.broadcasting_operator(other, lambda a, b: a + b)

    def __mul__(self, other):
        """pointwise multiply two block operators"""
        return self.broadcasting_operator(other, lambda a, b: a * b)

    def copy(self):
        return type(self)(self.block.copy())


class BlockOperator(BlockArray):
    """Blocked linear algebra

    Notes
    -----
    In this context, everything is always a 2d operator, like the operators themselves
    """

    @cached_property
    def rows(self):
        return self.block.shape[0]

    @cached_property
    def cols(self):
        return self.block.shape[1]

    @staticmethod
    def zeros(L, R):
        from pycomplex.stencil.operator import ZeroOperator
        return BlockOperator(
            [[ZeroOperator((l, r))
                for r in R]
                    for l in L]
        )


    @cached_property
    def is_square(self):
        """Return true if matrix is square"""
        r, c = np.array(self.shape).T
        return np.alltrue(r.T == c)

    def diagonal(self):
        """Get the diagonal of the block

        Returns
        -------
        DenseBlockArray
        """
        assert self.is_square
        raise NotImplementedError

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

        # FIXME: this only works for mat-mat multiply; as such does not genrealize to mat-vec
        L = [self.block[i, 0].shape[0] for i in range(self.block.shape[0])]
        R = [other.block[0, i].shape[1] for i in range(other.block.shape[1])]
        output = BlockOperator.zeros(L, R).block
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(output_shape[1]):
                    output[i, k] = output[i, k] + self.block[i, j] * other_block[j, k]

        return type(other)(output.reshape(self.rows, *other.block.shape[1:]))
