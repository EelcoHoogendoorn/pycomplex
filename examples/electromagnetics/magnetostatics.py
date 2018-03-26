
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

import numpy as np
import scipy.sparse

from pycomplex import synthetic
from pycomplex.topology import sign_dtype
from examples.linear_system import *


def make_mesh():
    """Construct domain

    Returns
    -------
    complex
        the complex to operate on
    chains
        boundary and body description chains
    """
    mesh = synthetic.n_cube(n_dim=2).as_22().as_regular()
    # subdivide
    for i in range(6):
        mesh = mesh.subdivide_cubical()

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
    current_0 = (PP[0][:, 0] == magnet_width) * (PP[0][:, 1] < magnet_height)

    plate_width = 0.3
    plate_height = 0.1
    plate_2 = (PP[2][:, 0] < plate_width) * (PP[2][:, 1] > magnet_height) * (PP[2][:, 1] < magnet_height + plate_height)
    plate_1 = mesh.topology.averaging_operators_N[1] * plate_2

    return mesh, all_1, left_1, bottom_0, bottom_1, current_0, plate_1


mesh, all_1, left_1, bottom_0, bottom_1, current_0, plate_1 = make_mesh()
# mesh.plot(plot_dual=False, plot_vertices=False)
mu = plate_1 * 10000 + 1


def magnetostatics(complex2):
    """Set up 2d magnetostatics

    Note that it does not actually involve any potentials
    And note the similarity, if not identicality, to potential flow problem
    """

    # grab the chain complex
    primal = complex2.topology
    boundary = primal.boundary
    assert boundary.is_oriented
    dual = primal.dual
    # make sure our boundary actually makes sense topologically
    primal.check_chain()
    dual.check_chain()

    P01, P12 = primal.matrices
    D01, D12 = dual.matrices_2

    P2P1 = P12.T
    D2D1 = D12.T

    P0, P1, P2 = primal.n_elements
    D0, D1, D2 = dual.n_elements
    B0, B1 = boundary.n_elements

    P0D2 = sparse_diag(complex2.hodge_PD[0])
    P1D1 = sparse_diag(complex2.hodge_PD[1])

    S = complex2.topology.dual.selector

    # compliance = scipy.sparse.diags(1/mu)
    rotation   = [P0D2 * D2D1]
    continuity = [P2P1 * P1D1 * scipy.sparse.diags(mu) * S[1]]   # dual boundary tangent fluxes are not involved in continuity

    # set up boundary equations
    rotation_bc   = [sparse_zeros((B0, d)) for d in [D1]]
    continuity_bc = [sparse_zeros((B1, d)) for d in [D1]]

    # antisymmetry on the bottom axis; set tangent flux to zero
    rotation_bc[0] = d_matrix(bottom_0, rotation_bc[0].shape, P1)

    # symmetry on the left axis; set normal flux to zero
    # the former only happens to work with minres
    # continuity_bc[0] = o_matrix(left_1, boundary.parent_idx[1], continuity_bc[0].shape)
    continuity_bc[0] = o_matrix(all_1 - bottom_1, boundary.parent_idx[1], continuity_bc[0].shape)

    equations = [
        rotation,
        rotation_bc,
        continuity,
        continuity_bc,
    ]

    flux = np.zeros(D1)   # one flux unknown for each dual edge
    unknowns = [flux]

    # set up right hand side
    # current = current          # rotation dependent on current
    current_bc = np.zeros(B0)
    source = np.zeros(P2)        # divergence is zero everywhere
    source_bc = np.zeros(B1)     # we only set zero fluxes in this example
    knowns = [
        current_0,
        current_bc,
        source,
        source_bc,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system


system = magnetostatics(mesh)
# system.plot()


# formulate normal equations and solve
normal = system.balance(1e-6).normal_equations()
# normal.precondition().plot()
from time import clock
solution, residual = normal.precondition().solve_amg(tol=1e-8)
t = clock()
print('starting solving')
solution, residual = normal.precondition().solve_minres(tol=1e-9)
print(residual)
print('solving time: ', clock() - t)
solution = [s / np.sqrt(d) for s, d in zip(solution, normal.diag())]
# solution, residual = normal.solve_direct()
flux, = solution


from examples.flow.stream import stream
primal_flux = mu * mesh.hodge_PD[1] * (mesh.topology.dual.selector[1] * flux)
phi = stream(mesh, primal_flux)

mesh.plot_primal_0_form(phi, cmap='jet', plot_contour=True, levels=39)

import matplotlib.pyplot as plt
plt.show()
