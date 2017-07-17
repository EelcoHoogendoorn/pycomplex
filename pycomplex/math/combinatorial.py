import numpy as np


def combinations(L, n):
    """Generate combinations from a sequence of elements.
    Returns a generator object that iterates over all
    combinations of n elements from a sequence L.

    Parameters
    ----------
    L : list-like
        Sequence of elements
    n : integer
        Number of elements to choose at a time

    Returns
    -------
    A generator object

    Examples
    --------
    >>> combinations([1, 2, 3, 4], 2)
    <generator object at 0x7f970dcffa28>
    >>> list(combinations([1, 2, 3, 4], 2))
    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    >>> list(combinations([1, 2, 3, 4], 1))
    [[1], [2], [3], [4]]
    >>> list(combinations([1, 2, 3, 4], 0))
    [[]]
    >>> list(combinations([1, 2, 3, 4], 5))
    [[]]
    >>> list(combinations(['a', 'b', 'c'], 2))
    [['a', 'b'], ['a', 'c'], ['b', 'c']]

    Notes
    -----
    The elements remain in the same order as in L
    """
    if n == 0 or n > len(L):
        yield 0, []
    else:
        for i in range(len(L) - n + 1):
            for p, t in combinations(L[i + 1:], n - 1):
                yield (p+i)%2, [L[i]] + t


def permutations(L):
    """Generate permutations from a sequence of elements.
    Returns a generator object that iterates over all
    permutations of elements in a sequence L.

    Parameters
    ----------
    L : list-like
        Sequence of elements

    Returns
    -------
    A generator object yielding tuples of parity and permutations

    Examples
    --------
    >>> permutations([1, 2, 3])
    <generator object at 0x7f970dcffe60>
    >>> list(permutations([1, 2, 3]))
    [[1, 2, 3], [2, 1, 3], [2, 3, 1], [1, 3, 2], [3, 1, 2], [3, 2, 1]]
    >>> list(permutations([1]))
    [[1]]
    >>> list(permutations([1, 2]))
    [[1, 2], [2, 1]]
    >>> list(permutations(['a', 'b', 'c']))
    [['a', 'b', 'c'], ['b', 'a', 'c'], ['b', 'c', 'a'], ['a', 'c', 'b'], ['c', 'a', 'b'], ['c', 'b', 'a']]
    """
    if len(L) == 1:
        yield 0, [L[0]]
    elif len(L) >= 2:
        (h, t) = (L[0:1], L[1:])
        for par, p in permutations(t):
            for i in range(len(p) + 1):
                yield (par + i) % 2, p[:i] + h + p[i:]


def permutations_map(n):
    l = list(np.arange(n))
    from pycomplex.math.combinatorial import permutations
    par, perm = zip(*permutations(l))
    par = np.asarray(par, dtype=np.int8)
    perm = np.asarray(perm, dtype=np.int8)
    return par, perm
