"""Streamfunction solver"""


def stream(complex, flux):
    """

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

