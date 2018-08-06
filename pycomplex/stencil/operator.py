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

    def __call__(self, *args, **kwargs):
        assert args[0].shape == self.shape[1]
        ret = self.right(*args, **kwargs)
        assert ret.shape == self.shape[0]
        return ret

    def __mul__(self, other):
        assert isinstance(other, type(self))
        assert self.shape[1] == other.shape[0]
        # construct composed operator functions
        def left(*args, **kwargs):
            self.left
        return StencilOperator(
            left=self.left
        )


class SymmetricOperator(object):
    """left equals right; transpose returns self"""
    def __init__(self, op: callable, shape: Tuple):
        self.left = op
        self.right = op
        self.shape = shape, shape

    @property
    def transpose(self):
        return self

    def __call__(self, *args, **kwargs):
        assert args[0].shape == self.shape[1]
        ret = self.right(*args, **kwargs)
        assert ret.shape == self.shape[0]
        return ret


class DiagonalOperator(SymmetricOperator):
    def __init__(self, diagonal: np.ndarray, shape: Tuple):
        self.diagonal = diagonal
        self.shape = shape, shape

    @property
    def inverse(self):
        return DiagonalOperator(
            1. / self.diagonal, self.shape[0]
        )

    def __call__(self, x):
        return self.diagonal * x


class InvertableOperator(object):
    """Use for hodge?"""
    pass


class ComposedOperator(object):
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
