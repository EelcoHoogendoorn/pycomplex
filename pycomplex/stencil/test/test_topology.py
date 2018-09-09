import numpy as np
import pytest
from numpy import testing as npt

from pycomplex.stencil.topology import StencilTopology


@pytest.mark.parametrize('i', [1, 2, 3, 4])
def test_duality(i):
    """Test numerically that matching primal and dual operators obey a transpose relationship"""
    topology = StencilTopology(tuple(np.arange(i, dtype=np.int) * 2 + 2))

    for n in range(topology.n_dim):
        p = topology.primal[n].to_dense()
        d = topology.dual[n].to_dense()

        npt.assert_allclose(p, d.T)


@pytest.mark.parametrize('i', [2, 3, 4])
def test_chain(i):
    """Test that primal and dual chain complexes are closed"""
    topology = StencilTopology(tuple(np.arange(i, dtype=np.int) * 2 + 2))
    for n in range(topology.n_dim - 1):
        f0 = topology.form(n)
        f0[...] = np.random.normal(size=f0.shape)
        f1 = topology.primal[n](f0)
        f2 = topology.primal[n+1](f1)
        npt.assert_allclose(f2, 0, atol=1e-5)

    for n in range(topology.n_dim, 1, -1):
        f0 = topology.form(n)
        f0[...] = np.random.normal(size=f0.shape)
        f1 = topology.dual[n-1](f0)
        f2 = topology.dual[n-2](f1)
        npt.assert_allclose(f2, 0, atol=1e-5)


@pytest.mark.parametrize('i', [ 2])
def test_averaging(i):
    print('i', i)
    topology = StencilTopology(tuple(np.arange(i, dtype=np.int) * 2 + 2))
    for n in range(topology.n_dim + 1):
        print('n', n)
        f0 = topology.form(0)
        f0[...] = np.random.normal(size=f0.shape)
        fn = topology.averaging_operators_0[n] * f0
        print(f0)
        print(fn)