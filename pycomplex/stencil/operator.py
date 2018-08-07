from typing import Tuple
import numpy as np


class StencilOperator(object):
    """Transposable linear operator for use in stencil based operations"""
    def __init__(self, left: callable, right: callable, shape: Tuple):
        self.left = left
        self.right = right
        self.shape = shape

    @property
    def transpose(self):
        return StencilOperator(
            right=self.left,
            left=self.right,
            shape=(self.shape[1], self.shape[0])
        )

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
        assert isinstance(other, type(StencilOperator))
        assert self.shape == other.shape
        return StencilOperator(
            right=lambda x: self(x) + other(x),
            left=lambda x: self.transpose(x) + other.transpose(x),
            shape=self.shape,
        )

    def __mul__(self, other):
        assert isinstance(other, type(StencilOperator))
        return ComposedOperator(self, other)


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
    def __init__(self, *args):
        self.operators = args
        for l, r in zip(self.operators[:-1], self.operators[1:]):
            assert l.shape[1] == r.shape[0]

        self.shape = self.operators[0].shape[0], self.operators[-1].shape[-1]

    @property
    def transpose(self):
        return ComposedOperator(*[o.transpose for o in self.operators[::-1]])

    def __call__(self, x):
        assert x.shape == self.shape[1]
        for op in self.operators[::-1]:
            x = op(x)
        assert x.shape == self.shape[0]
        return x
