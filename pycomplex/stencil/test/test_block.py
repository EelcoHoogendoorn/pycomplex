
import numpy as np
from pycomplex.stencil.block import BlockArray


def test_dense():

    B = BlockArray([np.eye(2), np.eye(2) * 2], ndim=1)
    D = B.to_dense()
    print(D)
    BB = B.from_dense(D)
    print(BB.to_dense())