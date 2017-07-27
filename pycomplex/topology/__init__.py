"""Discrete topology module


Add support for subdomains? can they be made sense of without circumcenter in cell?
shouldnt matter wrt topology, in any case

try and find an elegant method of supporting boundary topology

Make sure that easy conversion between all representations are available
Matrix form, and index form

how to represent a simplex? would like to be able to map dual boundary back and stuff
right now each collection of simplices is implicitly numbered as an arange,
but having an int label would be cleaner

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


def transfer_matrix(idx, t, shape):
    """Construct a transfer matrix

    Parameters
    ----------
    t

    Returns
    -------
    sparse matrix of shape [sub_elements, parent_elements]
    """
    # FIXME: this could use orientation information too
    return scipy.sparse.csr_matrix((
        np.ones_like(idx, dtype=sign_dtype),
        (idx, t)),
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
    n_dim = elements.size // len(elements)
    idx = np.arange(elements.size, dtype=index_dtype) // n_dim
    return scipy.sparse.csr_matrix((
        orientations.flatten().astype(sign_dtype),
        (elements.flatten().astype(index_dtype), idx)))


def sparse_to_elements(T):
    """Split a sparse topology matrix to array form, assuming equal entries"""
    T = T.tocoo()
    r, c = T.row, T.col
    q = npi.group_by(r).split_array_as_array(c)
    return q


def generate_boundary_indices(this, that):
    """map boundary in terms of vertices to their unique indices

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
    # FIXME: would we like an optional sorting over n_vertices here?
    i = npi.indices(
        np.sort(this.reshape(n_elements, n_vertices), axis=1),
        np.sort(that.reshape(-1, n_vertices), axis=1),
    ).astype(index_dtype)
    return i.reshape(that.shape[:that.ndim - (this.ndim - 1)])