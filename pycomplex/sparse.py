"""
custom sparse type, specialized for DEC-like applications

input and output ranges are explicitly stored

could generalize this into an ndsparse type; thatd be awesome
"""

import numpy as np
import numpy_indexed as npi
import scipy
from cached_property import cached_property


def ones_like(a):
    a = a.copy()
    a.data = np.ones_like(a.data)
    return a


def normalize_l1(A, axis=1):
    """Return A scaled such that the absolute sum over `axis` equals 1"""
    D = scipy.sparse.diags(1. / np.abs(np.array(A.sum(axis=axis))).flatten())
    if axis == 0:
        return A * D
    elif axis == 1:
        return D * A


# class Diagonal(object):
#     def __init__(self, diagonal):
#         self.diagonal = diagonal
# class SignedSparse(object):
#     """Sparse matrix having only +-1 entries"""
#     def __init__(self, positive, negative):
#         self.positive = positive
#         self.negative = negative


class Axis(object):
    @cached_property
    def is_unique(self):
        """no duplicate items"""


def multi_indices(this, that):
    """Find all possible combinations of indices i,j in this and that,
    such that this[i]==that[j]
    The set of (i, j) pairs is maximal and unique

    Should the items in both this and that be unique, npi.indices is a more efficient alternative
    with identical results

    Note that this function only has vectorized efficiency if the number of 'combinatorial patterns' is small
    relative to the overall size of the problem.

    Parameters
    ----------
    this : indexable object
        items to match with items in that
    that : indexable object
        items to match with items in this

    Returns
    -------
    i: ndarray, [n_pairs], int
        indices into this
    j: ndarray, [n_pairs], int
        indices into that

    """
    i_this = npi.as_index(this)
    i_that = npi.as_index(that)

    # FIXME: both counts need to be versus the same range
    _, c_this = npi.count(i_this)
    _, c_that = npi.count(i_that)
    ok = _

    # total number of entries in the broadcasting pattern
    nnz = (c_this * c_that).sum()

    cg = np.vstack((c_this, c_that)).T
    shapes, participants = npi.group_by(cg, ok)

    out_this = []
    out_that = []
    for shape, participant in zip(shapes, participants):
        this_idx, that_idx = np.indices(shape, np.int)

        if True:
            ti = npi.contains(participant, i_this)
            q = npi.group_by(this[ti]).split_array_as_array(np.flatnonzero(ti))
            f = q[:, this_idx.reshape(1, -1)]
            out_this.append(f.flatten())

            ti = npi.contains(participant, i_that)
            q = npi.group_by(that[ti]).split_array_as_array(np.flatnonzero(ti))
            f = q[:, that_idx.reshape(1, -1)]
            out_that.append(f.flatten())

        if False:
            for p in participant:
                a, = np.where(this==p)
                o = a[this_idx.flatten()]
                out_this.append(o)
                b, = np.where(that==p)
                o = b[that_idx.flatten()]
                out_that.append(o)

    out_this = np.concatenate(tuple(out_this))
    out_that = np.concatenate(tuple(out_that))

    if not len(out_this) == nnz:
        raise ValueError

    if not np.array_equiv(this[out_this], that[out_that]):
        # add a little sanity check; this should be true by construction
        raise ValueError

    return out_this, out_that


class Sparse(object):
    """Sparse matrix with explicit ranges based on npi operations
    """

    def __init__(self, axes, data):
        self.data = np.asarray(data)
        # FIXME: make axis object richer? cache index, and some other props like uniqueness
        self.axes = tuple([np.asarray(a) for a in axes])
        for a in self.axes:
            if not a.shape == (len(self.data),):
                raise ValueError

    @staticmethod
    def fromdense(arr):
        idx = np.indices(arr.shape, np.int)
        axes = [i.flatten() for i in idx]
        return Sparse(axes, arr[axes])

    @property
    def nnz(self):
        return len(self.data)

    @property
    def ndim(self):
        return len(self.axes)

    def concatenate(self, other):
        return Sparse(
            axes=[np.concatenate((s, o)) for s, o in zip(self.axes, other.axes)],
            data=np.concatenate((self.data, other.data)),
        )

    @cached_property
    def index(self):
        return npi.as_index((self.axes))

    def __add__(self, other):
        s = self.concatenate(other)
        axes, data = npi.group_by(s.index).sum(s.data)
        return Sparse(axes, data)

    def __mul__(self, other):
        if isinstance(other, Sparse):
            # FIXME: make self and other canonical first
            s = self.concatenate(other)
            # select from s only dual items
            axes, data = npi.group_by(s.index).prod(s.data)
            filter = s.index.count == 2
            return Sparse([a[filter] for a in axes], data[filter])
        elif isinstance(other, (float, int)):
            return Sparse(self.axes, self.data * other)

    def astype(self, dtype):
        return Sparse(self.axes, self.data.astype(dtype))

    def todense(self):
        return npi.Table(*self.axes).mean(self.data)

    def dot(self, other):
        if isinstance(other, Sparse):
            col_idx = npi.as_index(self.axes[-1])
            row_idx = npi.as_index(other.axes[0])

            # if not np.array_equiv(col_idx.unique, row_idx.unique):
            #     raise Exception('incompatible ranges for inner product')

            # determine broadcasting pattern from axes to be contracted
            si, oi = multi_indices(self.axes[-1], other.axes[0])


            free = []
            if self.ndim == 2:
                free.append(self.axes[0][si])
            if other.ndim == 2:
                free.append(other.axes[-1][oi])

            # sum over j
            g = npi.group_by(tuple(free))
            k, v = g.sum(self.data[si] * other.data[oi])
            return Sparse(k, v)
        else:
            raise NotImplementedError

    @property
    def T(self):
        return Sparse(self.axes[::-1], self.data)

    def canonical(self):
        """sum doubles, drop zeros"""

    def filter(self, condition):
        mask = condition(self.data)
        return Sparse([a[mask] for a in self.axes], self.data[mask])

