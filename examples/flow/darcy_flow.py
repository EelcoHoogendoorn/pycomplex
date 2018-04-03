
# -*- coding: utf-8 -*-

"""
darcy flow:
    flow is divergence free
    follows darcy's law; proportionality flux and the product of pressure gradient and permeability

[mu,  grad] [v] = [f]
[div, 0   ] [P]   [0]

if we model P as a primal-n-form, we get
[I, δ] [v] = [f]
[d, 0] [P]   [0]

[[I, 0],  [δ, 0]]  [vi]   [fi]
[[0, I],  [δ, I]]  [vp] = [fp]

[[d, d],  [0, 0]]  [Pi]   [0]
[[0, _],  [0, _]]  [Pd]   [0]

This can be solved easily using a normal equation method, leading to a second order system

Should we take the effort to make this symmetrical, including boundary conditions, minres could be applied directly.
The reduced condition number could be very valuable if variable mu renders the equations stiff

Alternatively, we may eliminate flux and solve for pressure alone.
this might help a lot with amg performance
"""

from time import clock

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

from examples.flow.stream import stream
from examples.linear_system import *


def grid(shape=(32, 32)):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()


def concave(levels=6):
    # take a 2x2 grid
    mesh = grid(shape=(2, 2))
    # discard a corner
    mesh = mesh.select_subset([1, 1, 1, 0])
    mesh = mesh.copy(vertices=mesh.vertices * 64)

    for i in range(levels):  # if you change this, change scaling below accordingly, to keep RD happy
        mesh = mesh.subdivide_cubical()

    return mesh


def setup_regions(complex):
    BPP = complex.boundary.primal_position
    left_1 = BPP[1][:, 0] == BPP[1][:, 0].min()
    right_1 = BPP[1][:, 0] == BPP[1][:, 0].max()

    # construct closed part of the boundary
    closed_1 = complex.boundary.topology.chain(1, fill=1)
    closed_1[np.nonzero(left_1)] = 0
    closed_1[np.nonzero(right_1)] = 0
    all_0 = complex.boundary.topology.chain(0, fill=1)

    return dict(
        left_1=left_1,
        right_1=right_1,
        closed_1=closed_1,
        all_0=all_0
    )


def generate_pattern(complex):
    """Use reaction-diffusion to set up an interesting permeability-pattern

    Parameters
    ----------
    complex

    Returns
    -------
    mu : ndarray
        dual 1-form

    Notes
    -----
    high min/max mu ratios make the equations very stiff, but gives the coolest looking results
    Solving the resulting equations may take a while; about a minute on my laptop
    Efficiently solving these equations is a known hard problem.
    """
    from examples.diffusion.reaction_diffusion import ReactionDiffusion
    rd = ReactionDiffusion(complex, key='labyrinth')
    rd.simulate(300)
    form = rd.state[0]
    if False:
        complex.plot_primal_0_form(form, plot_contour=False, cmap='jet')
        plt.show()
    # mu = form[complex.topology.elements[1]].mean(axis=1)
    mu = complex.topology.averaging_operators_0[1] * form   # map vertices to edges
    S = mesh.topology.dual.selector
    mu -= mu.min()
    mu /= mu.max()
    mu *= mu
    mu += .00001

    mu = S[-2].T * mu   # zero on dual boundary; is that ok?

    return mu


def setup_darcy_flow(complex, regions):
    """New style darcy flow setup"""
    assert complex.topology.n_dim >= 1
    system = System.canonical(complex)[-2:, -2:]
    equations = dict(momentum=0, continuity=1)
    variables = dict(flux=0, pressure=1)

    # FIXME: encapsulate this direct access with some modifier methods
    system.A.block[equations['continuity'], variables['pressure']] *= 0   # no compressibility
    # FIXME: implementation of mu not quite successfull yet?
    # system.A.block[equations['momentum'], variables['flux']] *= sparse_diag(1. / regions['mu'])
    system.A.block[equations['momentum'], variables['pressure']] = \
        sparse_diag(regions['mu']) * system.A.block[equations['momentum'], variables['pressure']]

    # prescribe tangent flux on entire boundary; all 0
    system.set_dia_boundary(equations['momentum'], variables['flux'], regions['all_0'])
    # set normal flux to zero
    system.set_off_boundary(equations['continuity'], variables['flux'], regions['closed_1'])
    # prescribe pressure on inlet and outlet
    inlet, outlet = regions['left_1'], regions['right_1']
    system.set_dia_boundary(equations['continuity'], variables['pressure'], inlet + outlet)
    system.set_rhs_boundary(equations['continuity'], outlet.astype(np.float) - inlet.astype(np.float))

    return system


if __name__ == '__main__':
    mesh = concave()
    regions = setup_regions(mesh)
    if True:
        mu = generate_pattern(mesh)
    else:
        mu = mesh.topology.chain(1, fill=1, dtype=np.float)
    regions['mu'] = mu

    # mesh.plot()
    system = setup_darcy_flow(mesh, regions)
    system = system.balance(1e-9)
    # system.plot()

    if True:
        # solve by elimination; seems to be working. even including mu terms we get a symmetric result
        system_up = system.eliminate([0], [0])
        solution, residual = system_up.solve_minres()
        pressure = solution.merge()
        # system_up.plot()

        pressure = mesh.topology.dual.selector[2] * pressure
        pressure = mesh.hodge_PD[2] * pressure
        mesh.plot_primal_2_form(pressure)

        plt.show()
    else:
        normal = system.normal()
        # normal.plot()
        solution, residual = normal.solve_minres()
        flux = solution[-2].merge()

        # visualize
        from examples.flow.stream import setup_stream, solve_stream
        phi = solve_stream(setup_stream(mesh), flux)
        mesh.plot_primal_0_form(phi - phi.min(), cmap='jet', plot_contour=True, levels=29)
        plt.show()
quit()



def darcy_flow(complex2, mu):
    """Formulate darcy flow over a 2-complex

    Parameters
    ----------
    mu : primal 1-form, describing local permeability
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
    # P1P0 = P01.T
    # D2D1 = D12.T
    D1D0 = D01.T

    P0, P1, P2 = primal.n_elements
    D0, D1, D2 = dual.n_elements
    B0, B1 = boundary.n_elements

    P1D1 = sparse_diag(complex2.hodge_PD[1])
    mu = sparse_diag(mu)


    P2D0_0 = sparse_zeros((P2, D0))


    momentum   = [P1D1       , mu * P1D1 * D1D0]      # darcy's law
    continuity = [P2P1 * P1D1, P2D0_0          ]
    # set up boundary equations
    continuity_bc = [sparse_zeros((B0, d)) for d in [P1, D0]]
    # set normal flux
    continuity_bc[0] = o_matrix(closed, boundary.parent_idx[1], continuity_bc[0].shape)
    # set opening pressures
    continuity_bc[1] = d_matrix(inlet + outlet, continuity_bc[1].shape, P2)


    equations = [
        momentum,
        continuity,
        continuity_bc,
    ]

    velocity = np.zeros(P1)     # no point in modelling tangent flux for darcy flow
    pressure = np.zeros(D2)
    unknowns = [
        velocity,
        pressure
    ]

    force  = np.zeros(P1)
    source = np.zeros(P2)
    source_bc = np.zeros(B0) - inlet.astype(np.float) + outlet.astype(np.float)  # set opening pressures
    knowns = [
        force,
        source,
        source_bc,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system


# toggle the use of anisotropic permeability
if True:
    # Use reaction-diffusion to set up an interesting permeability-pattern
    # high min/max mu ratios make the equations very stiff, but gives the coolest looking results
    # Solving the resulting equations may take a while; about a minute on my laptop
    # Efficiently solving these equations is a known hard problem.
    from examples.diffusion.reaction_diffusion import ReactionDiffusion
    rd = ReactionDiffusion(mesh, key='labyrinth')
    rd.simulate(300)
    form = rd.state[0]
    if True:
        tris = mesh.subdivide_simplicial()
        tris.as_2().plot_primal_0_form(tris.topology.transfer_operators[0] * form, plot_contour=False, cmap='jet')
        plt.show()
    mu = form[mesh.topology.elements[1]].mean(axis=1)   # map vertices to edges
    mu -= mu.min()
    mu /= mu.max()
    mu *= mu
    mu += .00001
else:
    mu = mesh.topology.chain(1, fill=1, dtype=np.float)


# formulate darcy flow equations
system = darcy_flow(mesh, mu)
if False:   # only set to true for small problem sizes
    system.plot()

# formulate normal equations and solve
normal = system.normal_equations()
t = clock()
print('starting solving')
# FIXME: solve_minres_amg only 20% faster than solve_minres
solution, residual = normal.precondition().solve_minres_amg(tol=1e-16)
print('solving time: ', clock() - t)
solution = [s / np.sqrt(d) for s, d in zip(solution, normal.diag())]
flux, pressure = solution

# plot result
tris = mesh.subdivide_simplicial()
pressure = mesh.topology.dual.selector[2] * pressure
pressure = mesh.hodge_PD[2] * pressure
pressure = tris.topology.transfer_operators[2] * pressure
tris.as_2().plot_primal_2_form(pressure)

primal_flux = mesh.hodge_PD[1] * flux
phi = stream(mesh, primal_flux)
phi = tris.topology.transfer_operators[0] * phi
tris.as_2().plot_primal_0_form(phi, cmap='jet', plot_contour=True, levels=50)

plt.show()
