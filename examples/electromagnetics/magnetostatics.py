
# -*- coding: utf-8 -*-

"""Model a permanent magnet

Physics:
    div B = 0
    curl H = J
    B = mu H

Or in DEC:
    dB = 0      magnetic flux is divergence-free
    δH = J      magnetic flux is irrotational, where no current is present

With B a 1-form on a 2d manifold, or a 2-form on a 3d manifold


AMG works poorly here; worse than pure minres. for scalar laplace we get factor two gain, here factor two loss
note that the algebraic properties of the normal equations here are a pretty standard vector laplace-beltrami;
not clear why it should perform any worse than seismic simulation?
it doesnt; appears amg isnt as effective for vectorial fields generally?
seems like it; when adding anisotropy, amg becomes no faster; but less stable!
this is in contrast to seismic, where eigen decomposition becomes more stable with amg. what gives?


normal-equations to solve is essentially a vector-laplacian;
how is it different from elasto-statics? there, neither rotation nor compression is zero.
the fact that the solution are divergence and rotation free is a consequence of
left-multiplication of rhs?
normal-equation rhs is projected on the subspace of solenoidal and irrotational fields

note that we could make the mesh spacing variable to efficiently simulate open field at infinity

"""

import scipy.sparse
import matplotlib.pyplot as plt

from examples.linear_system import *
from pycomplex import synthetic


def setup_mesh(levels=6):
    # generate a mesh
    mesh = synthetic.n_cube(n_dim=2).as_22().as_regular()
    hierarchy = [mesh]
    # subdivide
    for i in range(levels):
        mesh = mesh.subdivide_cubical()
        hierarchy.append(mesh)
    return mesh, hierarchy


def setup_domain(mesh):
    """Construct domain

    Parameters
    ----------
    mesh : Complex

    Returns
    -------
    chains
        boundary and body description chains
    """

    # identify boundaries
    edge_position = mesh.boundary.primal_position[1]
    BPP = mesh.boundary.primal_position
    left_1 = (BPP[1][:, 0] == BPP[1][:, 0].min()).astype(sign_dtype)
    bottom_1 = (BPP[1][:, 1] == BPP[1][:, 1].min()).astype(sign_dtype)

    # right = (BPP[1][:, 0] == BPP[1][:, 0].max()).astype(sign_dtype)
    # construct closed part of the boundary
    all_1 = mesh.boundary.topology.chain(1, fill=1)
    closed_1 = (BPP[1][:, 1] != BPP[1][:, 1].min()).astype(sign_dtype)

    left_0  = (BPP[0][:, 0] == BPP[0][:, 0].min()).astype(sign_dtype)
    right_0 = (BPP[0][:, 0] == BPP[0][:, 0].max()).astype(sign_dtype)

    bottom_0 = (BPP[0][:, 1] == BPP[0][:, 1].min()).astype(sign_dtype)
    bottom_0 = bottom_0 * (1-left_0) * (1-right_0)

    # construct surface current; this is the source term
    PP = mesh.primal_position
    magnet_width = 0.25
    magnet_height = 0.25
    magnet_width = PP[0][:, 0][np.argmin(np.abs(PP[0][:, 0] - magnet_width))]
    # could write this as directional derivative of magnetization
    current_0 = (PP[0][:, 0] == magnet_width) * (PP[0][:, 1] < magnet_height)

    plate_width = 0.3
    plate_height = 0.1
    plate_2 = (PP[2][:, 0] < plate_width) * (PP[2][:, 1] > magnet_height) * (PP[2][:, 1] < magnet_height + plate_height)
    # primal 1-form
    plate_1 = mesh.topology.averaging_operators_N[1] * plate_2

    mu = plate_1 * 10000 + 1
    S = mesh.topology.dual.selector
    mu = S[-2].T * mu

    return dict(
        all_1=all_1,
        left_1=left_1,
        bottom_0=bottom_0,
        bottom_1=bottom_1,
        current_0=current_0,
        plate_1=plate_1,
        mu_1=mu,
    )


def setup_magnetostatics(complex, regions):
    """new style magnetostatics setup

    Parameters
    ----------
    complex : Complex
        complex with topological dimension > 2
    regions : dict[str, ndarray]
        dict of chains specifying regions of interest

    Returns
    -------
    system : System
        complete first order system describing magnetostatics problem
    """
    assert complex.topology.n_dim >= 2  # makes no sense in lower-dimensional space
    # unknown is a dual 1-form; two equations to close from from both sides
    system = System.canonical(complex)[-3:, :][:, [-2]]
    equations = dict(rotation=0, flux=1, divergence=2)  # should these names be tracked by system?
    variables = dict(flux=0)


    # add in mu; variable
    system.A.block[equations['divergence'], variables['flux']] *= scipy.sparse.diags(regions['mu_1'])
    system.A.block[equations['flux'], variables['flux']] *= 0   # flux is mostly unspecified

    # NOTE: boundary conditions begin here
    # antisymmetry on the bottom axis; set tangent flux to zero
    system.set_dia_boundary(equations['flux'], variables['flux'], regions['bottom_0'])

    # symmetry on the left axis; set normal flux to zero; also at 'infinity'
    # interesting; in this setup, could set normal flux on middle block just as easily
    system.set_off_boundary(equations['divergence'], variables['flux'], regions['all_1'] - regions['bottom_1'])

    # apply the source terms; current induces nonzero rotation
    system.set_rhs(equations['rotation'], regions['current_0'].astype(np.float))
    return system


if __name__ == '__main__':
    mesh, hierarchy = setup_mesh(levels=5)
    regions = setup_domain(mesh)
    # mesh.plot(plot_dual=False, plot_vertices=False)

    system = setup_magnetostatics(mesh, regions)

    system = system.block_balance(k=-2) # should not add much compared to row based balance here
    system = system.balance(1e-9)
    normal = system.normal()
    solution, res = normal.solve_minres()
    # normal.plot()

    flux = solution.merge()

    from examples.flow.stream import setup_stream, solve_stream
    phi = solve_stream(setup_stream(mesh), regions['mu_1'] * -flux)
    mesh.plot_primal_0_form(phi - phi.min(), cmap='jet', plot_contour=True, levels=29)
    plt.show()

