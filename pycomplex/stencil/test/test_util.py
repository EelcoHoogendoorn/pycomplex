

from pycomplex.stencil.util import generate, pascal, smoother


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