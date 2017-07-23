"""
darcy flow:
    divergence free
    follows darcy's law; proportionality between gradient of pressure and flow velocity and permeability mu

[mu,  grad] [v] = [f]
[div, 0   ] [P]   [0]

if we model P as a primal-n-form, we get
[I, δ] [v] = [f]
[d, 0] [P]   [0]

[[I, 0, 0],  [δ, 0]]  [vi]   [fi]
[[0, I, 0],  [δ, I]]  [vp] = [fp]
[[0, 0, J],  [0, b]]  [vd]   [fd]

[[d, d, 0],  [0, 0]]  [Pi]   [0]
[[0, I, b],  [0, J]]  [Pd]   [0]

interesting question; should vd always be zero?
if b is zero, eq is quite simple

[[I, 0],  [δ, 0]]  [vi]   [fi]
[[0, I],  [δ, I]]  [vp] = [fp]

[[d, d],  [0, 0]]  [Pi]   [0]
[[0, _],  [0, _]]  [Pd]   [0]

either way, this lends itself perfectly to either solving as second order normal equation,
or directly using minres if we bother to make it symmetrical
indeed normal equations are overkill here, and solving in terms of the physical pressure potential
does not seem to impose any compromises

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

    for i in range(2):
        mesh = mesh.subdivide()
    return mesh


mesh = concave()
mesh.metric()


def darcy_flow(complex2):
    """Formulate darcy flow over a 2-complex"""

    # grab the chain complex
    P01, P12 = complex2.topology.matrices
    D01, D12 = complex2.topology.dual.matrices

    P2P1 = P12.T
    P1P0 = P01.T
    D2D1 = D12.T
    D1D0 = D01.T

    P1D1 = sparse_diag(complex2.P1D1)
    P2D0 = sparse_diag(complex2.P2D0)
    P0D2 = sparse_diag(complex2.P0D2)

    P0, P1, P2 = complex2.topology.n_elements
    D0, D1, D2 = complex2.topology.dual.n_elements

    P2D0_0 = sparse_zeros((P2, D0))

    S = complex2.topology.dual.selector

    momentum   = [P1D1       , P1D1 * D1D0]      # darcy's law
    continuity = [P2P1 * P1D1, P2D0_0     ]
    equations = [
        momentum,
        continuity
    ]

    velocity = np.zeros(P1)     # no point in modelling tangent flux for darcy flow
    pressure = np.zeros(D2)
    unknowns = [
        velocity,
        pressure
    ]

    force  = np.zeros(P1)
    source = np.zeros(P2)
    knowns = [
        force,
        source,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system


system = darcy_flow(mesh)

system.print()
# now add bc's; pressure difference over manifold, for instance
system.plot()

N = system.normal_equations()
N.print()
N.plot()