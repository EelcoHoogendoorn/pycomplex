

from pycomplex.stencil.util import generate, pascal, smoother, checkerboard, checkerboard_2


def test_generate():
    ndim = 3
    symbols, terms, axes, parities = generate(ndim)
    print(symbols)
    assert len(symbols) == ndim + 1
    for i, s in enumerate(symbols):
        assert pascal(ndim, i) == len(s)


def test_smoother():
    for n in range(3):
        assert smoother(n).sum() == 1


def test_checkerboard():
    print(checkerboard((4, 6)))

    for i in range(4):
        print(checkerboard_2((4, 6), i))