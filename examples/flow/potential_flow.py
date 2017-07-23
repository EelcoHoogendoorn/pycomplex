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
    for i in range(2):
        mesh = mesh.subdivide()
        left = mesh.topology.transfer_matrices[1] * left
        right = mesh.topology.transfer_matrices[1] * right
    return mesh, left, right

mesh, left, right = concave()
mesh.metric()


def potential_flow(complex2):
    """Set up potential flow system

    Note that it does not actually involve any potentials
    And note the similarity, if not identicality, to EM problems
    """
    # grab all the operators we will be needing
    P01, P12 = complex2.topology.matrices
    D01, D12 = complex2.topology.dual.matrices

    P2P1 = P12.T
    P1P0 = P01.T
    D2D1 = D12.T
    D1D0 = D01.T

    P1D1 = sparse_diag(complex2.P1D1)
    P0D2 = sparse_diag(complex2.P0D2)

    P0, P1, P2 = complex2.topology.n_elements
    D0, D1, D2 = complex2.topology.dual.n_elements

    S = complex2.topology.dual.selector

    rotation   = [P0D2 * D2D1]
    continuity = [P2P1 * P1D1 * S[1]]   # dual boundary tangent fluxes are not involved in continuity
    equations = [
        rotation,
        continuity,
    ]

    velocity = np.zeros(D1)   # one velocity unknown for each dual edge
    unknowns = [velocity]

    vortex = np.zeros(P0)  # generally irrotational
    source = np.zeros(P2)  # generally incompressible
    knowns = [
        vortex,
        source,
    ]

    system = BlockSystem(equations=equations, knowns=knowns, unknowns=unknowns)
    return system


potential_system = potential_flow(mesh)
potential_system.plot()

S = mesh.topology.dual.selector
# all normal fluxes zero, except the ends
bc_rhs = mesh.topology.chain(1, fill=0)
bc_rhs[left] = 1
bc_rhs[right] = -1
bc_eq = [[S[1]]]



N = potential_system.normal_equations()
N.plot()
mesh.plot()