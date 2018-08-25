from typing import Tuple
import numpy as np


class StencilOperator(object):
    """Transposable linear operator for use in stencil based operations"""
    def __init__(self, left: callable, right: callable, shape: Tuple):
        self.left = left
        self.right = right
        self.shape = shape

    @property
    def operators(self):
        return [self]

    @property
    def transpose(self):
        return StencilOperator(
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
        # FIXME: if we make this a dedicated object it might enable compute graph optimisations
        # FIXME: alternatively, implement some simple optimizations here; fold diagonal operators, and so on
        assert isinstance(other, StencilOperator)
        assert self.shape == other.shape

        if isinstance(self, ZeroOperator):
            return other
        if isinstance(other, ZeroOperator):
            return self

        return StencilOperator(
            right=lambda x: self(x) + other(x),
            left=lambda x: self.transpose(x) + other.transpose(x),
            shape=self.shape,
        )

    def __mul__(self, other):
        assert isinstance(other, StencilOperator)
        return ComposedOperator(self.operators + other.operators).simplify()


class ZeroOperator(StencilOperator):
    """Operator that maps all input to zero"""
    def __init__(self, shape):
        zero = lambda x: x * 0
        super(ZeroOperator, self).__init__(zero, zero, shape)

    @property
    def inverse(self):
        raise NotImplementedError


class ClosedOperator(StencilOperator):
    """Operator which produces closed forms;
     that is an operator which applied to itself equals zero.

    """
    # FIXME: Name is wrong, but nilpotent with n is 2 makes for such a poor class name.
    # maybe just call it derivativeOperator?


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
        self.right = lambda x: x * self.diagonal
        self.left = lambda x: x / self.diagonal

    @property
    def inverse(self):
        return DiagonalOperator(
            1. / self.diagonal, self.shape[0]
        )


class ComposedOperator(StencilOperator):
    """Chains a sequence of operators"""
    def __init__(self, *args):
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
        if any(isinstance(l, ClosedOperator) and isinstance(r, ClosedOperator)
               for l, r in zip(self.operators[:-1], self.operators[1:])):
            return True
        return False

    def simplify(self):
        """Implement some simplification rules, like combining diagonal operators"""
        if self.is_zero():
            return ZeroOperator(self.shape)
        return self

    @property
    def transpose(self):
        return ComposedOperator(*[o.transpose for o in self.operators[::-1]])

    def __call__(self, x):
        assert x.shape == self.shape[1]
        for op in self.operators[::-1]:
            x = op(x)
        assert x.shape == self.shape[0]
        return x
