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



"""

class BlockSystem(object):
    """Blocked linear system"""

    def __init__(self, system, rhs, unknowns):

        self.system = np.asarray(system, dtype=np.object)
        unknown_shape = [row.shape[1] for row in self.system]

        # check that subblocks are consistent
        self.rhs = rhs
        self.unknowns = unknowns

    @property
    def shape(self):
        return self.system.shape

    def normal_equations(self):
        raise NotImplementedError

    def concatenate(self):
        """Concatenate blocks into single system"""
        raise NotImplementedError

    def split(self, x):
        """Split concatted vector into blocks

        Parameters
        ----------
        x : ndarray, [n_cols], float

        """
        raise NotImplementedError


import numpy as np
import scipy.sparse

def grid(shape=(32, 32)):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()

def concave():
    # take a 2x2 grid
    mesh = grid(shape=(2, 2))
    # discard a corner
    mesh = mesh.select_subset([1, 1, 1, 0])

    # identify two boundaries
    edge_position = mesh.primal_position()[1]
    left = edge_position[:, 0] == edge_position[:, 0].min()
    right = edge_position[:, 0] == edge_position[:, 0].max()

    # subdivide
    for i in range(5):
        mesh = mesh.subdivide()
        left = mesh.topology.transfer_matrices[1] * left
        right = mesh.topology.transfer_matrices[1] * right
    return mesh, left, right

mesh, left, right = concave()
mesh.metric()

print(np.flatnonzero(left))
print(np.flatnonzero(right))

def sparse_diag(diag):
    s = len(diag)
    i = np.arange(s)
    return scipy.sparse.csc_matrix((diag, (i, i)), shape=(s, s))


def potential_flow(complex2):
    # grab all the operators we will be needing
    T01 = complex2.topology.matrix(0, 1).T
    T12 = complex2.topology.matrix(1, 2).T
    D01, D12 = complex2.topology.dual()
    # grad = T01
    div = T01.T
    curl = T12
    mass = complex2.P0D2
    D1P1 = complex2.D1P1

    continuity = div
    momentum = curl * sparse_diag(D1P1)

    system = [
        [continuity],
        [momentum]
    ]
    source = np.zeros(complex2.topology.n_elements[2])
    vortex = np.zeros(complex2.topology.n_elements[0])
    rhs = [
        source,
        vortex,
    ]
    unknowns = np.zeros(complex2.topology.n_elements[1])
    return BlockSystem(system=system, rhs=rhs, unknowns=unknowns)


mesh.plot()