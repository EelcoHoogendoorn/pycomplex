import numpy as np
import numpy_indexed as npi
from fastcache import clru_cache


def pascal(n, k):
    """Evaluate pascals triangle"""
    if k in (0, n):
        return 1
    return pascal(n - 1, k - 1) + pascal(n - 1, k)


def totuple(a):
    """Cast nested iterable to tuple"""
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


@clru_cache()
def generate(ndim):
    """Generate symbols and their derivative relations of anticommutative exterior algebra

    Parameters
    ----------
    ndim : int
        dimension of the space

    Returns
    -------
    symbols : tuple
        describes the interpretation of each component of each form
    terms : tuple
        refers to symbols in the lower forms that derives forms are built from
    axis : tuple
        axis to differentiate any given term to
    parities : tuple
        sign term
    """
    symbols = [(tuple(),)]  # 0-form is represented by empty symbol
    parities = []
    terms = []
    axes = []

    for n in range(ndim):
        # NOTE: interesting that we are reusing this code here...
        from pycomplex.topology.simplicial import permutation_map
        par, perm = permutation_map(n)
        p = symbols[n]
        # for new generation of symbols as all previous generations derived wrt all directions
        s = [q + (i,) for i in range(ndim) for q in p if i not in q]
        s = np.array(s)
        ss = np.sort(s, axis=1)
        arg = np.argsort(s, axis=1).astype(par.dtype)
        parity = par[npi.indices(perm, arg)]    # parity of each term relative to sorted order

        idx = npi.as_index(ss)
        u, up = npi.group_by(idx, parity)
        # last dir added to s is the current diff direction. earlier columns are id of terms to diff to reach s
        gs = npi.group_by(idx).split(s)
        gid, gd = gs[..., :-1], gs[..., -1]

        symbols.append(totuple(u))
        parities.append(totuple(up))
        axes.append(totuple(gd))
        terms.append(totuple(gid))

    return tuple(symbols), tuple(terms), tuple(axes), tuple(parities)


def binning(arr, steps):
    """inverse of tile"""
    shape = [(a // b, b) for a, b in zip(arr.shape, steps)]
    shape = [c for p in shape for c in p]
    return arr.reshape(shape).mean(axis=tuple(np.arange(len(steps), dtype=np.int) * 2 + 1))


def unbinning(arr, output, axes):
    """scale a factor two in all directions, but fill some with zeros"""


def smoother(ndim):
    """Note: this a seperable kernel. """
    if ndim == 1:
        return np.array([1, 2, 1]) / 4
    return smoother(1) * smoother(ndim-1)[..., None]