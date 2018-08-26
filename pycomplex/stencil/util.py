import numpy as np
import numpy_indexed as npi
from fastcache import clru_cache
# NOTE: interesting that we are reusing this code here...
from pycomplex.topology.simplicial import permutation_map


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
    """Generate symbols and their derivative relations describing anticommutative exterior algebra

    Parameters
    ----------
    ndim : int
        dimension of the space

    Returns
    -------
    symbols : tuple
        describes the interpretation of each component of each form
        each form is described by a set of components
        and each component is described by a set of integers
        where the integers denote the directions in which these variables have an extent
        for instance, in a 3d space, (1, 2) describes an area element dy*dz
    terms : tuple
        indices referring to components in the parent forms that derived forms are built from
    axis : tuple
        axis to differentiate any given parent component to
    parities : tuple
        sign term of above mentioned differentiation

    Notes
    -----
    This function may be called often and is nontrivial to compute,
    hence the immutable return types and caching
    """
    symbols = [(tuple(),)]  # 0-form is represented by empty symbol
    parities = []
    terms = []
    axes = []

    for n in range(ndim):
        # new generation of terms consists of all previous symbols derived wrt all directions not yet derived to
        t = [s + (d,) for d in range(ndim) for s in symbols[n] if d not in s]

        # compute parity of each term
        par, perm = permutation_map(n)
        parity = par[npi.indices(perm, np.argsort(t, axis=1).astype(perm.dtype))]

        # group terms by identical output components
        idx = npi.as_index(np.sort(t, axis=1))
        gp = npi.group_by(idx).split(parity)
        gt = npi.group_by(idx).split(t)
        gi = [[symbols[n].index(i) for i in t] for t in totuple(gt[..., :-1])]

        symbols.append(totuple(idx.unique))
        parities.append(totuple(gp))
        axes.append(totuple(gt[..., -1]))
        terms.append(totuple(gi))

    return tuple(symbols), tuple(terms), tuple(axes), tuple(parities)


def binning(arr, steps):
    """inverse of tile"""
    shape = [(a // b, b) for a, b in zip(arr.shape, steps)]
    shape = [c for p in shape for c in p]
    return arr.reshape(shape).mean(axis=tuple(np.arange(len(steps), dtype=np.int) * 2 + 1))


def smoother(ndim):
    """Note: this a seperable kernel. """
    if ndim == 1:
        return np.array([1, 2, 1]) / 4
    return smoother(1) * smoother(ndim-1)[..., None]


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def checkerboard_2(shape, offset=0):
    return np.einsum('i, i...->...', 2**np.arange(len(shape)), np.indices(shape) % 2) == offset
