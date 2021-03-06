
# -*- coding: utf-8 -*-

"""Streamfunction solver"""

import scipy.sparse

from examples.linear_system import System


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


def setup_stream(complex):
    """Formulate 2d streamfunction system

    [0, δ] [phi ] = [r]
    [d, I] [flux] = [0]

    Parameters
    ----------
    complex : Complex
        a 2d complex

    Returns
    -------
    System
        representing a stream function
    """
    # FIXME: could write this as the normal equation of System.canonical(complex)[0, 1]!
    assert complex.topology.n_dim == 2  # only makes sense in 2d
    system = System.canonical(complex)[:2, :2]
    system.A.block[0, 0] *= 0
    # set all tangent fluxes to zero explicitly
    system.set_dia_boundary(1, 1, complex.topology.boundary.chain(n=1, fill=1))

    return system


def solve_stream(stream, flux_d1, eliminate=True):
    """

    Parameters
    ----------
    stream : System
        first order stream system
    flux_d1 : ndarray
        dual flux 1-form
    eliminate : bool
        if true, solve by elimination

    Returns
    -------
    phi : ndarray
        primal 0-form
    """
    # set the rhs to match the problem at hand
    S = stream.complex.topology.dual.selector_interior[1]
    # filter out tangent fluxes; must be zero
    vorticity = stream.A.block[0, 1] * S.T * S * flux_d1
    # FIXME: this mutable design is disgusting
    stream.rhs.block[0] = vorticity

    if eliminate:
        # solve through elimination
        # stream.plot()
        laplace = stream.eliminate([1], [1])    # eliminate the flux variables
        # laplace.plot()
        x, r = laplace.solve_minres()
    else:
        stream = stream.balance(1e-9)
        normal = stream.normal()
        # normal.plot()
        x, r = normal.solve_minres()
    return stream.complex.hodge_PD[0] * x.block[0]


def setup_potential(complex):
    """Formulate potential function system

    [I, δ] [flux] = [0]
    [d, 0] [P] = [d]

    Parameters
    ----------
    complex : Complex

    Returns
    -------
    System
        representing a potential function
    """
    system = System.canonical(complex)[-2:, -2:]
    system.A.block[1, 1] *= 0
    # set all tangent fluxes to zero explicitly
    # FIXME: this is nonsense. can potential be reconstructed from flux alone? maybe not..
    system.set_dia_boundary(0, 0, complex.topology.boundary.chain(n=-2, fill=1))

    return system


def solve_potential(potential, flux_d1, eliminate=True):
    """

    Parameters
    ----------
    potential : System
        first order stream system
    flux_d1 : ndarray
        dual flux 1-form
    eliminate : bool
        if true, solve by elimination

    Returns
    -------
    P : ndarray
        dual 0-form
    """
    # set the rhs to match the problem at hand
    S = potential.complex.topology.dual.selector_interior
    # filter out tangent fluxes; must be zero
    divergence = potential.A.block[1, 0] * flux_d1
    # FIXME: this mutable design is disgusting
    potential.rhs.block[1] = divergence

    if eliminate:
        # solve through elimination
        # stream.plot()
        laplace = potential.eliminate([0], [0])    # eliminate the flux variables
        # laplace.plot()
        P, r = laplace.solve_minres()
    else:
        potential = potential.balance(1e-9)
        normal = potential.normal()
        # normal.plot()
        P, r = normal.solve_minres()
    return S[-1] * P.block[-1]
