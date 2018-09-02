from cached_property import cached_property

import numpy as np

from pycomplex.stencil.operator import ZeroOperator


class BlockArray(object):
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

    def __sub__(self, other):
        """Sub two block operators"""
        return self.broadcasting_operator(other, lambda a, b: a - b)

    def __mul__(self, other):
        """pointwise multiply two block operators"""
        return self.broadcasting_operator(other, lambda a, b: a * b)

    def sum(self):
        return sum(b.sum() for b in self.block)

    def abs(self):
        return self.apply(np.abs)

    def invert(self):
        return self.apply(lambda x: 1 / x)

    def simplify(self):
        return self.apply(lambda x: x.simplify())

    def copy(self):
        return type(self)(self.block.copy())

    def __getslice__(self, slc):
        """Slicing simply passes on to the block"""
        return type(self)(self.block.__getitem__(slc))

    def __getitem__(self, item):
        return self.block[item]

    def __setitem__(self, key, value):
        try:
            if value == 0:
                from pycomplex.stencil.operator import ZeroOperator
                self.block[key] = ZeroOperator(shape=self.block[key].shape)
                return
        except:
            pass
        try:
            if value == 1:
                from pycomplex.stencil.operator import IdentityOperator
                assert self.block[key].shape[0] == self.block[key].shape[1]
                self.block[key] = IdentityOperator(shape=self.block[key].shape[0])
                return
        except:
            pass
        try:
            from pycomplex.stencil.operator import StencilOperator
            if isinstance(value, StencilOperator):
                assert self.block[key].shape == value.shape
                self.block[key] = value
                return
        except:
            pass
        try:
            if isinstance(value, np.ndarray):
                assert self.block[key].shape == value.shape
                self.block[key] = value
                return
        except:
            pass
        raise NotImplementedError

    # def to_dense(self):
    #     return np.block([r for r in self.block])


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

    @staticmethod
    def identity(L):
        eye = BlockOperator.zeros(L, L)
        for i, l in enumerate(L):
            eye[i, i] = 1
        return eye

    @cached_property
    def is_square(self):
        """Return true if matrix is square"""
        r, c = np.array(self.shape).T
        return np.alltrue(r.T == c)

    def to_dense(self):
        # dense = self.apply(lambda x: x.to_dense())    # something broken with apply
        return np.block([[e.to_dense() for e in r] for r in self.block])

    def diagonal(self):
        """Get the diagonal of the block

        Returns
        -------
        DenseBlockArray

        Notes
        -----
        This brute-forces the stencil diagonal using some checkerboard-type evaluation
        Perhaps there may be a more efficient method of reaching the same result?
        This requires two evaluations for each form-component,
        where each form is evaluated typically over a laplacian-like operator,
        so about a sixfold multiplier relative to a single forward smoother evaluation,
        which is about the cost of a single V-cycle; so manageble.

        Proven to work for for a full complex with random hodges

        """
        # FIXME: should this not be moved elsewhere? seems quite problem-specific;
        # and needs to know about seperate components
        from pycomplex.stencil.util import checkerboard

        def pattern(shape, c, sign):
            pattern = np.zeros(shape)
            pattern[c] = checkerboard(shape[1:]) if sign else 1 - checkerboard(shape[1:])
            return pattern

        def do(db, shape, c):
            p, a = pattern(shape, c, 0), pattern(shape, c, 1)
            return db(p) * p + db(a) * a
        # assert self.is_square
        def block(i):
            db = self.block[i, i]
            shape = db.shape[1]
            return sum([do(db, shape, i) for i in range(shape[0])])

        return BlockArray([block(i) for i in range(self.rows)])

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
        if not isinstance(other, BlockOperator):
            return self.vector_mul(other)
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
                    # FIXME: cleaner to make operator implement a fused multiply-add
                    if not isinstance(self.block[i, j], ZeroOperator):
                        output[i, k] = output[i, k] + self.block[i, j] * other_block[j, k]

        return type(other)(output.reshape(self.rows, *other.block.shape[1:]))

    def vector_mul(self, other: BlockArray):
        L = [self.block[i, 0].shape[0] for i in range(self.block.shape[0])]

        output = [np.zeros(b) for b in L]
        for i in range(self.rows):
            for j in range(self.cols):
                if not isinstance(self.block[i, j], ZeroOperator):
                    output[i] = output[i] + self.block[i, j] * other.block[j]

        return type(other)(output)

    def __repr__(self):
        return '\n'.join('\t'.join([repr(e) for e in row]) for row in self.block)
