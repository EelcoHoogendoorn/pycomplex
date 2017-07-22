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
[[I, 0], [d, 0, 0], [0, 0]] [Oi]   [0]
[[0, I], [d, d, I], [0, 0]] [Op]   [0]

[[δ, δ], [0, 0, 0], [d, 0]] [vi]   [fi]
[[0, δ], [0, 0, 0], [d, I]] [vp] = [fp]
[[0, I], [0, 0, J], [0, b]] [vd]   [fd]

[[0, 0], [δ, δ, 0], [0, 0]] [Pi]   [0]
[[0, 0], [0, I, b], [0, J]] [Pd]   [0]

we have a relation between [vp, Pd] and [Op, vd]
b term is quite interesting too; encodes a first order difference between vd or Pd;
constant tangent velocity or constant pressure along the boundary

normalize bcs with potential infs on the diag
drop the infs by giving them prescribed values
we now have a symmetric well posed problem that we can feed to minres (P may still have a gauge)

what would squaring the first order system imply for stokes?
seems like it would give a triplet of laplacians of each flavor, with some coupling terms on the off-diagonal blocks

this is of course what least-squares based minres does internally anyway!
absolves us from obsessing about symmetry, and gives more leeway in exploring boundary conditions!


"""
import numpy as np
import scipy.sparse
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

    for i in range(5):
        mesh = mesh.subdivide()
    return mesh

mesh = concave()
mesh.metric()


def debug_harmonics():
    from examples.spherical_harmonics import get_harmonics_0, get_harmonics_2
    v = get_harmonics_2(mesh)
    q = mesh.to_simplicial_transfer_2(v[:, -10])
    mesh.to_simplicial().as_2().plot_primal_2_form(q)

    v = get_harmonics_0(mesh)
    print(v[:, 5])
    q = mesh.to_simplicial_transfer_0(v[:, 7])
    mesh.to_simplicial().as_2().plot_primal_0_form(q)



def stokes_flow(complex2):

    # grab the chain complex
    P01 = complex2.topology.matrix(0, 1).T
    P12 = complex2.topology.matrix(1, 2).T
    D01, D12 = complex2.topology.dual.matrix

    # mass = complex2.P0D2
    # D1P1 = complex2.D1P1
    P1D1 = sparse_diag(complex.P1D1)
    P2D0 = sparse_diag(complex.P2D0)

    vorticity  = [1         , D01       , 0      ]
    momentum   = [P01 * P1D1, 0         , D12    ]
    continuity = [0         , P12 * P2D0, 0      ]

    system = [
        vorticity,
        momentum,
        continuity
    ]
    vortex = np.zeros(complex2.topology.n_elements[0])
    force  = np.zeros(complex2.topology.n_elements[1])
    source = np.zeros(complex2.topology.n_elements[2])
    rhs = [
        vortex,
        force,
        source,
    ]
    omega    = np.zeros(complex2.topology.dual.n_elements[0])
    velocity = np.zeros(complex2.topology.dual.n_elements[1])
    pressure = np.zeros(complex2.topology.dual.n_elements[2])
    unknowns = [
        omega,
        velocity,
        pressure
    ]

    return BlockSystem(system=system, rhs=rhs, unknowns=unknowns)
