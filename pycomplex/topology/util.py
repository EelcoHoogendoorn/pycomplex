import numpy as np

from pycomplex.topology import sign_dtype, index_dtype


def parity_to_orientation(parity):
    """Convert parity to orientation

    Parameters
    ----------
    parity : ndarray, [...], bool

    Returns
    -------
    orientation : ndarray, [...], sign_dtype
    """
    return ((np.asarray(parity) * 2) - 1).astype(sign_dtype)


def orientation_to_parity(parity):
    """Convert orientation to parity

    Parameters
    ----------
    orientation : ndarray, [...], sign_dtype

    Returns
    -------
    parity : ndarray, [...], bool
    """
    return np.asarray(parity) < 0


def indices(shape, dtype=index_dtype):
    """mem-efficient version of np.indices

    Parameters
    ----------
    shape : tuple[Int]
    dtype : np.dtype, optional
        numpy compatible integer dtype

    Returns
    -------
    grid : List[ndarray[shape], dtype]
        len(grid)==len(shape)
        The result is functionally identical to np.indices,
        but using stride tricks to save memory
    """
    n_dim = len(shape)
    idx = [np.arange(s, dtype=dtype) for s in shape]
    for q, (i, s) in enumerate(zip(idx, shape)):
        strides = [0] * n_dim
        strides[q] = i.strides[0]
        idx[q] = np.ndarray(buffer=i.data, shape=shape, strides=strides, dtype=i.dtype)
    return idx


def sort_and_argsort(arr, axis):
    """Potentially faster method to sort and argsort; hasnt been profiled though

    Parameters
    ----------
    arr : ndarray, [shape]
    axis : int
        axis to sort along

    Returns
    -------
    sorted : ndarray, [shape]
    argsort : ndarray, [shape], int
        indices along axis of arr
    """
    argsort = np.argsort(arr, axis=axis)
    I = indices(arr.shape, index_dtype)
    I[axis] = argsort
    return arr[I], argsort


def relative_permutations(self, other):
    """Combine two permutations of indices to get relative permutation

    Parameters
    ----------
    self : ndarray, [n, m], int
    other : ndarray, [n, m], int

    Returns
    -------
    relative : ndarray, [n, m], int
    """
    assert self.shape == other.shape
    assert self.ndim == 2
    I = np.indices(self.shape)
    relative = np.empty_like(self)
    relative[I[0], other] = self
    return relative