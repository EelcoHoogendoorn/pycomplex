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
from examples.linear_system import *



def grid(shape=(32, 32)):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()

def concave():
    # take a 2x2 grid
    mesh = grid(shape=(2, 2))
    # discard a corner
    mesh = mesh.select_subset([1, 1, 0, 1])

    # identify two boundaries
    edge_position = mesh.primal_position()[1]
    left = edge_position[:, 0] == edge_position[:, 0].min()
    right = edge_position[:, 0] == edge_position[:, 0].max()

    # subdivide
    for i in range(3):
        mesh = mesh.subdivide()
        left = mesh.topology.transfer_matrices[1] * left
        right = mesh.topology.transfer_matrices[1] * right
    return mesh, left, right

mesh, left, right = concave()
mesh.metric()


# BM = mesh.topology.dual.blocked_matrices
# print(BM)


def potential_flow(complex2):
    # grab all the operators we will be needing
    P1P0 = complex2.topology.matrix(0, 1).T
    P2P1 = complex2.topology.matrix(1, 2).T
    D1D0, D2D1 = complex2.topology.dual.matrix

    # mass = complex2.P0D2
    # D1P1 = complex2.D1P1
    P0D2 = sparse_diag(complex2.P0D2)
    P1D1 = sparse_diag(complex2.P1D1)

    rotation   = [P0D2 * D2D1]
    continuity = [P2P1 * P1D1]
    system = [
        rotation,
        continuity,
    ]

    vortex = np.zeros(complex2.topology.n_elements[0])  # generally irrotational
    source = np.zeros(complex2.topology.n_elements[2])  # generally incompressible
    rhs = [
        vortex,
        source,
    ]

    velocity = np.zeros(complex2.topology.dual.n_elements[1])   # one velocity unknown for each dual edge
    unknowns = [velocity]

    return BlockSystem(system=system, rhs=rhs, unknowns=unknowns)


potential_system = potential_flow(mesh)

N = potential_system.normal_equations()

mesh.plot()