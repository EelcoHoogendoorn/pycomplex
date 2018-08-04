from typing import Tuple


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