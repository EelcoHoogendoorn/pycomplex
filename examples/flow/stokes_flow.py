
# -*- coding: utf-8 -*-

"""
stokes lid driven cavity

u = 1 at the top

stokes in common second order form:
[L,   grad] [v] = [f]
[div, 0   ] [P]   [0]

making the following substitutions:
O = curl v
L = curl curl + grad div
we get stokes in first order form
[I,    curl, 0   ] [O]   [0]
[curl, 0,    grad] [v] = [f]
[0,    div,  0   ] [P]   [0]
this makes bc's easiest to see; each dual boundary element introduces a new unknown, breaking our symmetry

# we can split up each variable as originating in the interior, primal boundary, or dual boundary (i,p,d)
[[I, 0], [δ, 0, 0], [0, 0]] [ωi]   [mi]
[[0, I], [δ, b, I], [0, 0]] [ωp]   [mp]

[[d, d], [0, 0, 0], [δ, 0]] [vi]   [fi]
[[0, b], [0, 0, 0], [I, I]] [vp] = [fp]
[[0, _], [0, 0, _], [0, 0]] [vd]   [_]

[[0, 0], [d, I, 0], [0, 0]] [Pi]   [si] source/sink
[[0, 0], [0, _, 0], [0, _]] [Pd]   [_]

This implies a relation between [vp, Pd] and [ωp, vd] if se week to restore symmetry to the system

normalize bcs with potential infs on the diag
drop the infs by giving them prescribed values
we now have a symmetric well posed problem that we can feed to minres (P may still have a gauge)
this merely allows us to see a subset of boundary conditions that is provably consistent;
does not provably give us all possible consistent boundary conditions

However, rather than making the first order equation symmetrical, we can simply solve the normal equations.
This is at least conceptually even simpler

"""

from pycomplex.topology import sign_dtype
from examples.linear_system import *


def grid(shape=(32, 32)):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()


def concave():
    # take a 2x2 grid
    mesh = grid(shape=(2, 2))
    # discard a corner
    mesh = mesh.select_subset([1, 1, 1, 0])

    for i in range(4):  # subdiv 5 is already pushing our solvers...
        mesh = mesh.subdivide_cubical()
        # left = mesh.topology.transfer_matrices[1] * left
        # right = mesh.topology.transfer_matrices[1] * right

    # identify sides of the domain
    edge_position = mesh.boundary.primal_position[1]
    left  = (edge_position[:, 0] == edge_position[:, 0].min()).astype(sign_dtype)
    right = (edge_position[:, 0] == edge_position[:, 0].max()).astype(sign_dtype)
    # construct closed part of the boundary
    all = mesh.boundary.topology.chain(1, fill=1)
    closed = all - left - right

    return mesh, left, right, closed


mesh, inlet, outlet, closed = concave()
# mesh.plot()


def stokes_flow(complex2):
    """Formulate stokes flow over a 2-complex

    Note that this is incompressible time-invariant stokes flow
    Compressible time-variant stokes flow would use every 'slot'
    available on the 2d chain complex, and has the highest complexity,
    as measured by the number of terms in our equations.
    However, since it is mathematically more diagonally dominant,
    and its physics more local, it should be numerically easier to solve.

    """

    # grab the chain complex
    primal = complex2.topology
    boundary = primal.boundary
    dual = primal.dual
    # make sure our boundary actually makes sense topologically
    primal.check_chain()
    dual.check_chain()

    P01, P12 = primal.matrices
    D01, D12 = dual.matrices_2

    P2P1 = P12.T
    P1P0 = P01.T
    D2D1 = D12.T
    D1D0 = D01.T

    P0, P1, P2 = primal.n_elements
    D0, D1, D2 = dual.n_elements
    B0, B1 = boundary.n_elements

    P0D2 = sparse_diag(complex2.hodge_PD[0])
    P1D1 = sparse_diag(complex2.hodge_PD[1])

    # FIXME: autogen this given system shape
    P2D2_0 = sparse_zeros((P2, D2))
    P1D1_0 = sparse_zeros((P1, D1))
    P0D0_0 = sparse_zeros((P0, D0))
    P2D0_0 = sparse_zeros((P2, D0))

    S = complex2.topology.dual.selector

    # NOTE: could model divergence as a seperate 2-form; would that give more control over the numerical structure of the vector laplacian?
    vorticity  = [P0D2       , P0D2 * D2D1       , P0D0_0       ]
    momentum   = [P1P0 * P0D2, P1D1_0            , P1D1 * D1D0  ]
    continuity = [P2D2_0     , P2P1 * P1D1 * S[1], P2D0_0       ]

    # set up boundary equations
    continuity_bc = [sparse_zeros((B0, d)) for d in dual.n_elements[::-1]]
    momentum_bc = [sparse_zeros((B1, d)) for d in dual.n_elements[::-1]]
    # set normal flux
    continuity_bc[1] = o_matrix(closed, boundary.parent_idx[1], continuity_bc[1].shape)
    # set opening pressures
    continuity_bc[2] = d_matrix(inlet + outlet, continuity_bc[2].shape, P2)
    # set tangent flux
    momentum_bc[1] = d_matrix(inlet + outlet + closed, momentum_bc[1].shape, P1)
    # alternatively, set zero shear on boundaries
    # momentum_bc[0] = o_matrix(inlet + outlet + closed, boundary.parent_idx[0], momentum_bc[0].shape)

    equations = [
        vorticity,
        momentum,
        momentum_bc,
        continuity,
        continuity_bc,
    ]

    omega    = np.zeros(D2)
    velocity = np.zeros(D1)
    pressure = np.zeros(D0)
    unknowns = [
        omega,
        velocity,
        pressure
    ]

    vortex    = np.zeros(P0)
    force     = np.zeros(P1)
    force_bc  = np.zeros(B1)   # set tangent fluxes; all zero
    source    = np.zeros(P2)
    source_bc = np.zeros(B0) - inlet.astype(np.float) + outlet.astype(np.float)  # set opening pressures
    knowns = [
        vortex,
        force,
        force_bc,
        source,
        source_bc,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system

system = stokes_flow(mesh)


# system.print()
# system.plot()

# formulate normal equations and solve
# normal = system.preconditioned_normal_equations()
# normal.plot()

normal = system.normal_equations()
# normal.precondition().plot()
solution, residual = normal.precondition().solve_minres(tol=1e-16)
solution = [s / np.sqrt(d) for s, d in zip(solution, normal.diag())]

# solution, residual = system.solve_least_squares()
# solution, residual = normal.solve_minres()
# solution, residual = system.solve_direct()

vorticity, flux, pressure = solution
# normal.print()
# normal.plot()


# plot result
tris = mesh.subdivide_simplicial()
pressure = mesh.topology.dual.selector[2] * pressure
pressure = mesh.hodge_PD[2] * pressure
pressure = tris.topology.transfer_operators[2] * pressure
tris.as_2().plot_primal_2_form(pressure)

from examples.flow.stream import stream
primal_flux = mesh.hodge_PD[1] * (mesh.topology.dual.selector[1] * flux)
phi = stream(mesh, primal_flux)
phi = tris.topology.transfer_operators[0] * phi
tris.as_2().plot_primal_0_form(phi, cmap='jet', plot_contour=True)

vorticity = mesh.topology.dual.selector[0] * vorticity
vorticity = mesh.hodge_PD[0] * vorticity
vorticity = tris.topology.transfer_operators[0] * vorticity
limit = np.abs(vorticity).max()
tris.as_2().plot_primal_0_form(vorticity, cmap='seismic', vmin=-limit, vmax=limit, plot_contour=False, shading='gouraud')

import matplotlib.pyplot as plt
plt.show()