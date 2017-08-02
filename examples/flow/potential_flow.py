"""
potential flow:
    divergence free
    rotation free

can be modelled as scalar potential:
    v = grad ϕ
or by streamfunction, or vector potential:
    v = curl ψ

vector potential fails to model a source/sink
scalar potential fails to model a confined vortex
so neither is great.. which one do we choose?
... neither, which will be the crux of the approach taken here!
perhaps we shouldnt be calling it a potential flow then anymore...
but havnt settled on a better name yet that does capture the essence

[[δ, 0, 0]]         [Oi]     curl; vorticity source term
[[δ, δ, I]]  [vi]   [Op]
             [vp]
[[d, d, 0]]  [vd]   [Si]    divergence; source/sink term
[[0, I, b]]         [bc]    implies normal flux constraint; ok. implied tangent flux change; kutta?

(what does it look like in 3d? should get another b term in there.. interesting for bcs)

equations and unknowns quite well matched.
not symmetric however. pre-mult with self-transpose would solve it in least-squares sense
interestingly, this results in a vector-laplacian-beltrami being formed
thats an interesting new perspective to me
how does this procedure of squaring the first order system generalize? and what does it imply for bc's?
any bc that is orthogonal to the existing basis is a valid one id say
solving in least square sense is also interesting in terms of underconstraining

implications for bcs? b terms gives a constraint on change in tangent velocity. kutta condition?
do we lose anything in terms of boundary conditions by not explicitly modelling the potentials?
i doubt it since they are rather lacking in physical meaning
consider potential flow in annulus; radial section of a vortex
would be easy with prescribed vector potential
we could add a constraint on circulation by adding an extra row describing a sum of tangent fluxes
or set a single tangent flux; should propagate to the rest of the boundary,
and we can afford to leave one normal flux constraint out, since it is implied by incompressibility

no justification from symmetry; but that is rather lacking here anyway somehow
which is a good thing; first order physical equations are inherently unsymmetrical.
we let symmetry come from least-squares procedure; this raises condition number,
but just to the common laplacian case

thinking more about bc's:
deleting one face from the boundary will remove one equation from divergence
deleting one face from the interior will remove one equation from divergence,
and adds three dual boundary variables.
also three normal fluxes which are now boundary fluxes
assuming these normal fluxes are each set to zero,
we have already added more constraints to the system than we have removed;
3 added, one removed
yet we also added three tangent variables.
this implies we can indeed add one additional constraint;
either pin one tangent, or specify the circulation, or sum over those three

it seems that foregoing the potential is also great for open boundary conditions
since we are working with velocity directly in a minres sense,
there is already a preferred solution in the face of indeterminism,
which reflects physical intuition; smallest velocity squared is the
least energetic / smoothest field

"""

import numpy as np
import scipy.sparse

from pycomplex.topology import sign_dtype
from examples.linear_system import *


def grid(shape):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()


def concave():
    # take a 3x3 grid
    mesh = grid(shape=(3, 3))
    # make a hole in it
    mask = np.ones((3, 3), dtype=np.int)
    mask[1, 1] = 0
    mesh = mesh.select_subset(mask.flatten())

    # subdivide
    for i in range(2):
        mesh = mesh.subdivide()

    # identify boundaries
    edge_position = mesh.boundary.primal_position[1]
    BPP = mesh.boundary.primal_position
    left  = (edge_position[:, 0] == edge_position[:, 0].min()).astype(sign_dtype)
    right = (edge_position[:, 0] == edge_position[:, 0].max()).astype(sign_dtype)
    # construct closed part of the boundary
    all = mesh.boundary.topology.chain(1, fill=1)
    closed = all - left - right

    # 0-element boundary
    interior = (np.linalg.norm(BPP[0], axis=1) < 1).astype(sign_dtype)

    all_0 = mesh.boundary.topology.chain(0, fill=1)
    top_right_0 = mesh.boundary.topology.chain(0, fill=0)
    top_right_0[np.argmin(np.linalg.norm(BPP[0]-[0.5,0.5], axis=1))] = 1

    exterior = all_0 - interior

    return mesh, all, left, right, closed, interior, exterior, top_right_0


mesh, all, inlet, outlet, closed, interior, exterior, top_right = concave()
mesh.plot(plot_dual=False, plot_vertices=False)


def potential_flow(complex2):
    """Set up potential flow system

    Note that it does not actually involve any potentials
    And note the similarity, if not identicality, to EM problems
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

    # impose a circulation around both boundaries
    rotation_bc[0] = \
        d_matrix(interior, rotation_bc[0].shape, P1, rows=0) + \
        d_matrix(exterior, rotation_bc[0].shape, P1, rows=1) + \
        d_matrix(top_right, rotation_bc[0].shape, P1) * 0   # for some reason, this does not work

        # activate condition on normal flux
    continuity_bc[0] = o_matrix(all, boundary.parent_idx[1], continuity_bc[0].shape)

    equations = [
        rotation,
        rotation_bc,
        continuity,
        continuity_bc,
    ]

    velocity = np.zeros(D1)   # one velocity unknown for each dual edge
    unknowns = [velocity]

    vortex = np.zeros(P0)       # generally irrotational
    vortex_bc = np.zeros(B0)
    vortex_bc[0] = 2e1 * 8
    vortex_bc[1] = 0
    source = np.zeros(P2)       # generally incompressible
    source_bc = np.zeros(B1) + inlet - outlet
    knowns = [
        vortex,
        vortex_bc,
        source,
        source_bc,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system


system = potential_flow(mesh)
system.plot()


# formulate normal equations and solve
normal = system.normal_equations()
# normal.precondition().plot()
from time import clock
t = clock()
print('starting solving')
solution, residual = normal.precondition().solve_minres(tol=1e-16)
# print(residual)
print('solving time: ', clock() - t)
solution = [s / np.sqrt(d) for s, d in zip(solution, normal.diag())]
# solution, residual = normal.solve_direct()
flux, = solution

# plot result
tris = mesh.to_simplicial()

# now we compute a streamfunction after all; just for visualization.
# no streamfunctions were harmed in the finding of the flowfield.
from examples.flow.stream import stream
primal_flux = mesh.hodge_PD[1] * (mesh.topology.dual.selector[1] * flux)
phi = stream(mesh, primal_flux)
phi = tris.topology.transfer_operators[0] * phi
tris.as_2().plot_primal_0_form(phi, cmap='jet', plot_contour=True, levels=49)

import matplotlib.pyplot as plt
plt.show()