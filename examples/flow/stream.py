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
    P1P0 = T01.T
    D2D1 = T01
    D1P1 = scipy.sparse.diags(complex.hodge_DP[1])

    # construct our laplacian
    laplacian = D2D1 * (D1P1 * P1P0)
    vorticity = D2D1 * (D1P1 * flux)
    phi = scipy.sparse.linalg.minres(laplacian, vorticity, tol=1e-12)[0]
    phi -= phi.min()
    return phi
