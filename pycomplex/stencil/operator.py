from typing import Tuple
import numpy as np


class StencilOperator(object):
    """Transposable linear operator for use in stencil based operations

    Notes
    -----
    Nothing terribly stencil specific here
    """
    def __init__(self, left: callable, right: callable, shape: Tuple):
        self.left = left
        self.right = right
        self.shape = shape

    @property
    def transpose(self):
        return type(self)(
            right=self.left,
            left=self.right,
            shape=(self.shape[1], self.shape[0])
        )
    @property
    def T(self):
        return self.transpose

    @property
    def inverse(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        assert args[0].shape == self.shape[1]
        ret = self.right(*args, **kwargs)
        assert ret.shape == self.shape[0]
        return ret

    def __add__(self, other):
        return CombinedOperator([self, other]).simplify()

    def __mul__(self, other):
        if isinstance(other, StencilOperator):
            return ComposedOperator([self, other]).simplify()
        else:
            return self(other)

    def simplify(self):
        return self

    def to_dense(self):
        """Transform a matrix-free operator into a dense operator by brute force evaluation"""
        shape = self.shape[1]
        def canonical(i):
            r = np.zeros(shape)
            r[tuple(i)] = 1
            return r

        idx = np.indices(shape)
        r = [self(canonical(c)) for c in idx.reshape(len(shape), -1).T]
        return np.array(r).reshape(len(r), -1).T

    def __repr__(self):
        return 'f'


class ZeroOperator(StencilOperator):
    """Operator that maps all input to zero"""
    def __init__(self, shape):
        super(ZeroOperator, self).__init__(
            lambda x: np.zeros(shape[1]),
            lambda x: np.zeros(shape[0]),
            shape
        )

    @property
    def transpose(self):
        return ZeroOperator(
            shape=(self.shape[1], self.shape[0])
        )

    def __repr__(self):
        return '0'


class DerivativeOperator(StencilOperator):
    """Operator which produces closed forms;
     that is an operator which applied to itself equals zero.

    Only used as a tag to simplify expressions
    """
    @property
    def transpose(self):
        return DualDerivativeOperator(
            right=self.left,
            left=self.right,
            shape=(self.shape[1], self.shape[0])
        )

    def __repr__(self):
        return 'd'


class DualDerivativeOperator(StencilOperator):
    @property
    def transpose(self):
        return DerivativeOperator(
            right=self.left,
            left=self.right,
            shape=(self.shape[1], self.shape[0])
        )

    def __repr__(self):
        return 'δ'


class SymmetricOperator(StencilOperator):
    """left equals right; transpose returns self"""
    def __init__(self, op: callable, shape):
        self.left = op
        self.right = op
        self.shape = shape, shape

    @property
    def transpose(self):
        return self


class DiagonalOperator(SymmetricOperator):
    """Operator that makes only pointwise changes"""
    def __init__(self, diagonal: np.ndarray, shape):
        self.diagonal = diagonal
        self.shape = shape, shape
        op = lambda x: x * self.diagonal
        self.right = op
        self.left = op

    @property
    def inverse(self):
        with np.errstate(divide='raise'):
            return type(self)(
                1. / self.diagonal, self.shape[0]
            )

    def simplify(self):
        if np.allclose(self.diagonal, 0):
            return ZeroOperator(self.shape)
        if np.allclose(self.diagonal, 1):
            return IdentityOperator(self.shape[0])
        return self

    def __repr__(self):
        return "\\"


class HodgeOperator(DiagonalOperator):
    def __repr__(self):
        return r"*"


class IdentityOperator(DiagonalOperator):
    def __init__(self, shape):
        self.shape = shape, shape
        self.right = lambda x: x
        self.left = lambda x: x

    @property
    def inverse(self):
        return self

    @property
    def diagonal(self):
        return 1

    def __repr__(self):
        return 'I'


class ComposedOperator(StencilOperator):
    """Chains the action of a sequence of operators"""
    def __init__(self, args):
        for op in args:
            assert isinstance(op, StencilOperator)
        self.operators = args
        for l, r in zip(self.operators[:-1], self.operators[1:]):
            assert l.shape[1] == r.shape[0]

        self.shape = self.operators[0].shape[0], self.operators[-1].shape[-1]

    def is_zero(self):
        """Returns true if the sequence of operators can be deduced to be zero"""
        if any(isinstance(op, ZeroOperator) for op in self.operators):
            return True
        if any(isinstance(l, DerivativeOperator) and isinstance(r, DerivativeOperator)
               for l, r in zip(self.operators[:-1], self.operators[1:])):
            return True
        if any(isinstance(l, DualDerivativeOperator) and isinstance(r, DualDerivativeOperator)
               for l, r in zip(self.operators[:-1], self.operators[1:])):
            return True
        return False

    def simplify(self):
        """Implement some simplification rules, like combining diagonal operators"""
        if self.is_zero():
            return ZeroOperator(self.shape)
        # associativity
        for i in range(len(self.operators)):
            if isinstance(self.operators[i], ComposedOperator):
                ops = self.operators[:i] + self.operators[i].operators + self.operators[i+1:]
                return ComposedOperator(ops).simplify()
        # drop identity terms
        if any(isinstance(op, IdentityOperator) for op in self.operators):
            return ComposedOperator([op for op in self.operators if not isinstance(op, IdentityOperator)]).simplify()
        # combine adjacent diagonals
        for i in range(len(self.operators) - 2):
            l, r = self.operators[i: i+2]
            if isinstance(l, DiagonalOperator) and isinstance(r, DiagonalOperator):
                d = DiagonalOperator(diagonal=l.diagonal * r.diagonal, shape=l.shape[0]).simplify()
                return ComposedOperator(self.operators[:i] + [d] + self.operators[i+2:]).simplify()
        if len(self.operators) == 1:
            return self.operators[0].simplify()
        return self

    @property
    def transpose(self):
        return ComposedOperator([o.transpose for o in self.operators[::-1]])

    def __call__(self, x):
        assert x.shape == self.shape[1]
        for op in self.operators[::-1]:
            x = op(x)
        assert x.shape == self.shape[0]
        return x

    def __repr__(self):
        terms = ''.join(repr(o) for o in self.operators)
        return f'({terms})'



class CombinedOperator(StencilOperator):
    """Add the action of a set op operators"""

    def __init__(self, args):
        # FIXME: add possibility for sign flags to each operator term?
        self.operators = args

        self.shape = self.operators[0].shape

        for op in self.operators:
            assert isinstance(op, StencilOperator)
            assert op.shape == self.shape

        self.right = lambda x: sum(op(x) for op in self.operators)
        self.left = lambda x: sum(op.transpose(x) for op in self.operators)

    def simplify(self):
        # drop zero terms
        if any(isinstance(op, ZeroOperator) for op in self.operators):
            operators = [op for op in self.operators if not isinstance(op, ZeroOperator)]
            if len(operators):
                return CombinedOperator(operators).simplify()
            else:
                return ZeroOperator(self.shape)

        # associativity
        for i in range(len(self.operators)):
            if isinstance(self.operators[i], CombinedOperator):
                ops = self.operators[:i] + self.operators[i].operators + self.operators[i+1:]
                return CombinedOperator(ops).simplify()

        if len(self.operators) == 1:
            return self.operators[0].simplify()
        # FIXME: add check for summing repeated diagonal operators? never really happens does it?
        return self

    @property
    def transpose(self):
        return CombinedOperator([o.transpose for o in self.operators])

    def __repr__(self):
        terms = '+'.join(repr(o) for o in self.operators)
        return f'({terms})'
