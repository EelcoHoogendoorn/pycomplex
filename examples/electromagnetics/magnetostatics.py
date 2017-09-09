
# -*- coding: utf-8 -*-

"""Model a permanent magnet

Physics:
    div B = 0
    curl B = J

Or in DEC:
    dB = 0      magnetic flux is divergence-free
    δB = J      magnetic flux is irrotational, where no current is present

With B a 1-form on a 2d manifold, or a 2-form on a 3d manifold


"""

import numpy as np
import scipy.sparse

from pycomplex import synthetic
from pycomplex.topology import sign_dtype
from examples.linear_system import *


def make_mesh():
    mesh = synthetic.n_cube(2).as_22().as_regular()
    # subdivide
    for i in range(5):
        mesh = mesh.subdivide_cubical()

    # identify boundaries
    edge_position = mesh.boundary.primal_position[1]
    BPP = mesh.boundary.primal_position
    left  = (BPP[1][:, 0] == BPP[1][:, 0].min()).astype(sign_dtype)

    # right = (BPP[1][:, 0] == BPP[1][:, 0].max()).astype(sign_dtype)
    # construct closed part of the boundary
    all = mesh.boundary.topology.chain(1, fill=1)
    closed = (BPP[1][:, 1] != BPP[1][:, 1].min()).astype(sign_dtype)

    left_0  = (BPP[0][:, 0] == BPP[0][:, 0].min()).astype(sign_dtype)
    right_0  = (BPP[0][:, 0] == BPP[0][:, 0].max()).astype(sign_dtype)

    bottom_0  = (BPP[0][:, 1] == BPP[0][:, 1].min()).astype(sign_dtype)
    bottom_0 = bottom_0 * (1-left_0) * (1-right_0)

    # construct surface current
    PP = mesh.primal_position
    magnet_width = 0.25
    magnet_height = 0.05
    magnet_width = PP[0][:, 0][np.argmin(np.abs(PP[0][:, 0] - magnet_width))]
    current = (PP[0][:, 0] == magnet_width) * (PP[0][:, 1] < magnet_height)

    return mesh, all, left, bottom_0, current, closed


mesh, all, left, bottom, current, closed = make_mesh()
# mesh.plot(plot_dual=False, plot_vertices=False)


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

    rotation   = [P0D2 * D2D1       ]
    continuity = [P2P1 * P1D1 * S[1]]   # dual boundary tangent fluxes are not involved in continuity

    # set up boundary equations
    rotation_bc   = [sparse_zeros((B0, d)) for d in [D1]]
    continuity_bc = [sparse_zeros((B1, d)) for d in [D1]]

    # antisymmetry
    rotation_bc[0] = d_matrix(bottom, rotation_bc[0].shape, P1)

    # symmetry
    continuity_bc[0] = o_matrix(left, boundary.parent_idx[1], continuity_bc[0].shape)

    equations = [
        rotation,
        rotation_bc,
        continuity,
        continuity_bc,
    ]

    flux = np.zeros(D1)   # one flux unknown for each dual edge
    unknowns = [flux]

    # current = current          # rotation dependent on current
    current_bc = np.zeros(B0)
    source = np.zeros(P2)        # divergence is zero everywhere
    source_bc = np.zeros(B1)     # we only set zero fluxes in this example
    knowns = [
        current,
        current_bc,
        source,
        source_bc,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system


system = magnetostatics(mesh)
# system.plot()


# formulate normal equations and solve
normal = system.normal_equations()
# normal.precondition().plot()
from time import clock
t = clock()
print('starting solving')
solution, residual = normal.precondition().solve_minres(tol=1e-16)
print(residual)
print('solving time: ', clock() - t)
solution = [s / np.sqrt(d) for s, d in zip(solution, normal.diag())]
# solution, residual = normal.solve_direct()
flux, = solution

# plot result
tris = mesh.subdivide_simplicial()

# now we compute a streamfunction after all; just for visualization.
# no streamfunctions were harmed in the finding of the magnetic flux.
from examples.flow.stream import stream
primal_flux = mesh.hodge_PD[1] * (mesh.topology.dual.selector[1] * flux)
phi = stream(mesh, primal_flux)
phi = tris.topology.transfer_operators[0] * phi
tris.as_2().plot_primal_0_form(phi, cmap='jet', plot_contour=True, levels=49)

import matplotlib.pyplot as plt
plt.show()