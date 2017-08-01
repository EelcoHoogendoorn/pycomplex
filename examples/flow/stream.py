"""Streamfunction solver"""

import scipy.sparse


def stream(complex, flux):
    """Reconstruct a streamfunction to visualize a 2d incompressible flowfield

    Parameters
    ----------
    complex : Complex
    flux : ndarray
        primal 1-flux on the complex

    Returns
    -------
    stream : ndarray
        primal 0-form on the complex
    """

    T01 = complex.topology.matrices[0]
    grad = T01.T
    div = T01
    D1P1 = scipy.sparse.diags(complex.hodge_DP[1])

    # construct our laplacian
    laplacian = div * D1P1 * grad
    vorticity = div * D1P1 * flux
    return scipy.sparse.linalg.minres(laplacian, vorticity, tol=1e-12)[0]
