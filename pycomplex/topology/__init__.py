"""Discrete topology module

"""
import numpy as np
import numpy_indexed as npi
import scipy

# dtypes enforced for indices referring to elements;
# 16 bits is too few for many applications, but 32 should suffice for almost all
# these types are used globally throughout the package; changing them here should change them everywhere
index_dtype = np.int32
sign_dtype = np.int8


class ManifoldException(Exception):
    pass


def transfer_matrix(r, c, shape):
    """Construct a transfer matrix
    Essentially a simple wrapper around the sparse constructor

    Parameters
    ----------
    r : ndarray
        row indices
    c : ndarray
        column indices
    shape : tuple[int]
        shape of the resulting matrix

    Returns
    -------
    scipy.sparse.csr_matrix of shape `shape`
    """
    # FIXME: do we ever need orientation information here? not at the moment
    return scipy.sparse.csr_matrix((
        np.ones_like(r, dtype=sign_dtype),
        (r, c)),
        shape=shape
    )


def topology_matrix(elements, orientations):
    """Construct a topology matrix from a set of topological elements
    described in terms of their boundary element indices, and their relative orientation

    Parameters
    ----------
    elements : ndarray, [n_elements, ...], index_dtype
    orientations : ndarray, [n_elements, ...], sign_dtype

    Returns
    -------
    sparse matrix of shape [n_boundary_elements, n_elements]
    """
    n_bounds = elements.size // len(elements)
    idx = np.arange(elements.size, dtype=index_dtype) // n_bounds
    return scipy.sparse.csr_matrix((
        orientations.flatten().astype(sign_dtype),
        (elements.flatten().astype(index_dtype), idx)))


def sparse_to_elements(T):
    """Split a sparse topology matrix to array form,
    assuming an equal number of nonzeros per row,
    or incident elements per element

    Parameters
    ----------
    T : scipy.sparse, [n_elements, n_incident_elements]
        topology matrix describing incidence between elements

    Returns
    -------
    elements : ndarray, [n_elements, n_incident_elements_per_n_element], index_dtype
        element array

    Raises
    ------
    ValueError
        If not all elements can be described in terms of the same number of incident elements
    """
    T = T.tocoo()
    r, c = T.row, T.col
    q = npi.group_by(r).split_array_as_array(c)
    return q.astype(index_dtype)


def selection_matrix(s):
    """Construct a sparse selection matrix, that picks out elements
    according to the nonzeros of `s`

    Parameters
    ----------
    s : ndarray, [n], bool
        indicates variables to be selected

    Returns
    -------
    sparse, [nnz(s), len(s)]
        sparse selector matrix according to nonzeros of s
    """
    cols = np.flatnonzero(s).astype(index_dtype)
    rows = np.arange(len(cols), dtype=index_dtype)
    data = np.ones(len(cols), dtype=sign_dtype)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(rows), len(s)))


def element_indices(this, that):
    """Look up indices of elements described in `that` in `this`

    Parameters
    ----------
    this : ndarray, [n_elements, ...], int
        vertex indices describing unique n-elements
    that : ndarray, [n_test_elements, ...], int
        vertex indices describing n-elements to look up in 'this'

    Returns
    -------
    i : ndarray, [n_test_elements, ...], int
        element indices into 'this', in range [0, n_elements)

    """
    n_elements = len(this)
    n_vertices = this.size // n_elements
    # FIXME: make sorting optional?
    i = npi.indices(
        np.sort(this.reshape(n_elements, n_vertices), axis=1),
        np.sort(that.reshape(-1, n_vertices), axis=1),
    ).astype(index_dtype)
    return i.reshape(that.shape[:that.ndim - (this.ndim - 1)])
