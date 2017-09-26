import numpy
import numpy as np


def dot(a, b):
    """Compute the dot products over the last axes of two arrays of vectors

    Parameters
    ----------
    a : ndarray, [..., n], float
        array of vectors
    b : ndarray, [..., n], float
        array of vectors

    Returns
    -------
    ndarray, [...], float
        dot products over the last axes of a and b
    """
    return np.einsum('...i,...i->...', a, b)


def normalized(array_of_vectors, axis=-1, ord=2, ignore_zeros=False, return_norm=False):
    """Return a copy of a, normalized along axis with the ord-norm

    Parameters
    ----------
    array_of_vectors : ndarray, [..., n_dim]
        input data to be normalized
    axis : int, optional
        axis to perform normalization along
    ord : see np.linalg.norm
        the normalization order. default 2 gives the euclidean norm
    ignore_zeros : bool
        if true, elements with norm zero are left zero
    return_norm : bool
        if true, the norms are returned

    Returns
    -------
    normalized : ndarray, [..., n_dim]
        normalized values; same shape as input array
    norms : ndarray, [...], float, optional
        the norms of the input vectors

    See Also
    --------
    np.linalg.norm, for the precise spec of the ord parameter
    """
    array_of_vectors = np.asarray(array_of_vectors)
    norm = np.linalg.norm(array_of_vectors, ord, axis)
    norm = np.expand_dims(norm, axis)
    zeros = norm == 0
    if ignore_zeros:
        norm[zeros] = 1
    normed = array_of_vectors / norm
    if ignore_zeros:
        norm[zeros] = 0
    if return_norm:
        return normed, np.take(norm, 0, axis)
    else:
        return normed


def orthonormalize(axes, demirror=True):
    """Orthonormalize the input matrix

    Parameters
    ----------
    axes : ndarray, [n, m], float
        input axes to orthonormalize

    Returns
    -------
    ndarray, [n, m], float
        orthonormal matrix which most closely resembles axes

    Examples
    --------
    >>> orthonormalize([[0, -2], [2, 0]])
    [[0, -1], [1, 0]]

    """
    axes = np.asarray(axes)
    u, _, v = np.linalg.svd(axes)
    if demirror:
        det = np.linalg.det(axes)
        v[0] *= np.sign(det)
    return u.dot(np.eye(*axes.shape)).dot(v)


def power(m, p):
    """Raise an square matrix to a (fractional) power

    Parameters
    ----------
    m : ndarray, [..., n, n], float
    p : ndarray, [...], float

    Returns
    -------
    ndarray, [..., n, n], float
        input matrices raised to the power p

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Functional_calculus

    """
    p = np.expand_dims(np.asarray(p), -1)
    w, v = np.linalg.eig(m)
    return np.einsum('...ij,...j,...jk->...ik', v, w ** p, np.linalg.inv(v)).real


def rotation_from_plane(u, v):
    """Construct a rotation matrix from u and v, with u and v two orthonormal vectors spanning a plane"""
    n = len(u)
    R = [[0, 1], [-1, 0]]
    p = np.asarray([u, v])
    return np.eye(n) - np.outer(u, u) - np.outer(v, v) + np.dot(p.T, np.dot(R, p))


def angle_from_rotation(R):
    """Extract rotation angle in radians from n-dimensional rotation matrix"""
    n, m = R.shape[-2:]
    assert n == m
    return np.arccos(1 - (n - np.einsum('...ii->...', R)) / 2)


def adjoint(A):
    """Compute 3x3 adjoint matrices

    Parameters
    ----------
    A : ndarray, [..., 3, 3]
        (array of) 3 x 3 matrices

    Returns
    -------
    AI : ndarray, [..., 3, 3]
        adjoints of (array of) 3 x 3 matrices

    """
    AI = np.empty_like(A)
    for i in range(3):
        AI[..., i, :] = np.cross(A[..., i-2, :], A[..., i-1, :])
    return AI


def null(A):
    """Vectorized nullspace algorithm, for 3x3 rank-2 matrix
    simply cross each pair of vectors, and take the average

    Parameters
    ----------
    A : ndarray, [..., 3, 3]
        (array of) 3 x 3 matrices

    Returns
    -------
    null : ndarray, [..., 3]
        (array of) null space vectors

    """
    return adjoint(A).sum(axis=-2)


def inverse_transpose(A):
    """Efficiently compute the inverse-transpose of 3x3 matrices

    Parameters
    ----------
    A : ndarray, [..., 3, 3]
        (array of) 3 x 3 matrices

    Returns
    -------
    I : ndarray, [..., 3, 3]
        inverse-transpose of (array of) 3 x 3 matrices

    """
    I = adjoint(A)
    det = dot(I, A).mean(axis=-1)
    return I / det[...,None,None]


def inverse(A):
    """Inverse of 3x3 matrices

    Parameters
    ----------
    A : ndarray, [..., 3, 3]
        (array of) 3 x 3 matrices

    Returns
    -------
    I : ndarray, [..., 3, 3]
        inverses of (array of) 3 x 3 matrices

    """
    return np.swapaxes( inverse_transpose(A), -1,-2)


def pinv(A, r=1e-9):
    """Pseudoinverse of a (set of) small matrices

    Parameters
    ----------
    A : ndarray, [..., n, n]
        (array of) matrices
    r : float, optional
        tolerance parameter

    Returns
    -------
    I : ndarray, [..., n, n]
        pseudo inverses

    """
    u, s, v = np.linalg.svd(A)
    inv = 1 / s
    inv[np.abs(s / s[..., :1]) < r] = 0
    # s[:, self.complex.topology.n_dim:] = 0
    return np.einsum('...ij,...j,...jk->...ki', u[..., :s.shape[-1]], inv, v)
