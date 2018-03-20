"""
Lame parameter based elasticity model

defining the unknowns as:
    [(r)otation, (d)isplacement, (c)ompression]
and given lame parameters [μ, λ]

We get the following system of equations:
[I,        curl, 0       ] [r]   [0]
[curl * μ, 0,    grad * λ] [d] = [f]
[0,        div,  I       ] [c]   [0]

including hodges, we can make the system symmetric like this:
[*/μ, *δ, 0  ] [r]   [0]
[d*,  0,  *δ ] [d] = [f]
[0,   d*, */λ] [c]   [0]


if not purely incompressible (λ is finite), it is easy to rewrite this as equation of displacement alone
infact, total equation is only minimally different from vector-laplacian
As a matter of fact, given mu=lambda, or a poisson material, `solving an elasticity problem` is to
`solving a poisson problem` as `diffusing the vector laplacian` is to 'diffusing the scalar laplacian'

We can split up each variable as originating in the interior, primal boundary, or dual boundary (i,p,d)

[[I, 0], [δ, 0, 0], [0, 0]] [ri]   [0]
[[0, I], [δ, b, I], [0, 0]] [rp]   [0]      (b term here is boundary.T01)

[[d, d], [0, 0, 0], [δ, 0]] [di]   [fi]
[[0, b], [0, 0, 0], [I, I]] [dp] = [fp]     # this line relates to shear on the boundary... is it enough?
[[0, _], [0, 0, _], [0, ?]] [dd]   [_]      (if we take the full dual matrix, boundary.T01 term reappears here)

[[0, 0], [d, I, 0], [I, 0]] [ci]   [0]
[[0, 0], [0, _, ?], [0, _]] [cd]   [_]

This implies a relation between [dp, cd] and [rp, dd] if we seek to restore symmetry to the system

for freely vibrating object, boundary conditions are such that forces (shear and pressure)
are zero on the boundary. This is different from a bc on rotation.
Freely vibrating string may have zero rotation bc, denoting symmetry,
but vibration mode of tuning fork in bending is different;
tangential velocity nor rotation zero at the end; infact we oscilate between those two conditions

For an euler beam, free end implies second and third derivative of displacement to be zero:
https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Example:_unsupported_(free-free)_beam
Not sure how to relate that to our current musings. These are shear and moment obviously,
but we do not model moments explicitly, so lack of moment constraint is implied over lack of forces
imposed over the thickness; and a single layer of elements will simply never be an accurate model of a beam,
which by definition has nontrivial behavior over its section

yeah; zeroing rotation is indeed just a symmetry constraint, as in the flow-case; which we dont want
nor do we seek to constrain displacement; so momentum_bc is simply left blank?
cant eliminate related unknown though; would again imply zero tangential boundary displacement
so does this leave the system underconstrained? sure would seem so.

does the b term of full dual topology help us any? gives us a delta in tangent displacement;
no obvious interpretation.

another option suggested is a proportionality between boundary rotation and boundary displacement
this does feel correct intuitively; but where would this constant come from?
matching constants to subtracting equations, this is equivalent to saying
integral of circulation of displacement over contours excluding the boundary are to be zero
nah; fucks with our nullspace; constant displacement no longer in nullspace


is the problem here that we dont model shear explicitly?
well our equation for fp is about shear at the boundary really; so we can consider this taken care off
or is it; this is about force-balance on boundary in normal direction only
force-balance in tangential direction is a fundamentally different matter

since we are talking about tangent space; we could fit an entire laplacian in the dual-displacement block!
cant be right either though; constant displacement field in embedding space,
does not give null when applied to this boundary laplacian

different perspective: perhaps we should view the free beam modes, with mix of rotation and displacement
as a superposition of rotation=0 and displacement=0 modes?

wait a sec; compression at the boundary is zero is complete nonsense! what about standing pressure wave in rod?
will have nonzero compression that is constant over cross-section, no?

so the only things we know we can do are prescribed normal displacement, tangent displacement,
isotropic pressure, or rotation; or mixes of these things

no pressure at the boundary can also just be given on the rhs as fp

is this a known hard problem? seems like...
https://library.seg.org/doi/abs/10.1190/1.1512752
https://academic.oup.com/gji/article/172/1/252/586648
http://geo.mff.cuni.cz/~io/con96/con96h.htm

vacuum method; tapering elastic parameters to zero at the boundary can create a free boundary?
this is infact very obvious; as the 'air method' explains. physical simulation is just carried on
outside the solid into a gas.
conversely, letting physical parameters go to infinity, or metric to zero,
will block any motion, as per fluid paper
there might be some pitfalls related to numerical stability though
if density and stiffness change at the same time, propagation speed should be constant

this approach to bcs leaves the domain bcs quite inconsequential;
mixed ones probably good for diagonal dominance
better yet; just zero the dual vars; make MG easier too



Would like to use geometric multigrid here;
this creates some concern about coarsening reducing the topological complexity of the problem;
closing a c-shape and thus rendering all coarser operations pretty meaningless
this might be addressed by constructing 'overlapping' coarse grids,
whereby we work from the fine grid, and create a coarse cube for each connected 2*ndim block of fine cubes,
such that the coarser grids need not represent a partition of unity
how do we exactly formalize this intuition? fine blocks must have overlap at some n-cube,
in order to create a shared n-cube on the coarser level

this makes it a lot more like AMG; except that we aggregate according to a semi-regular pattern
also, constructing the smoother algebraicly may have advantages at the material property interfaces
not entirely clear 'how much' air needs to be simulated around the object.
at least one whole voxel on fine level
zero dual boundary bcs means zero rotation and zero pressure
if we dont have some padding on coarser levels too this is unlikely to interpolate well
that said; displacement field should be constant over the soft boundary;
but if we drive tangent to zero closeby this is likely to have some effect
yet we do not want infill at fine levels from coarse
infact, same could be said of interior; only care for refinement near boundary,
since interior is smooth for all we care
how would we do the transfer between subset and supersets though?
note that unkknown density would be a zigzag across the hierarchy; expand to finer cubes,
then trim down to cubes close to the surface.
on the top level, we are solving a problem only at the boundary; so how is the interior accounted for?
in essence, because we (selectively) allow topological changes on the coarser level;
if these opposing surfaces are actually connected
note that we are relying on the properties of the solver to obtain the resulting behavior of interest here;
the actual linear system describing the fine cubes will not have the eigenvectors we are looking for
it is not all that obvious that our eigensolver of choice will be very happy about this
and if our choice of connectivity on the coarse grid defines rather than guides our fine solver,
being correct about it becomes all the more important
either way, should start with a baseline implementation
perhaps writing optimized finite difference operators is more important in the end
this would preclude AMG or smart coarsened geometric multigrid though, and also, memory is likely the dominant concern
100^3 grid equals 4mb for 4byte float. if we want 50 eigenvectors thats 200mb.
including all other mem usage thats likely a rough upper limit for fully dense simulation
1mm res for a 10cm cube; 0.5mm voxel thickness analyzer more often than not times out at 300s, so that does not bode well

bcs; all hodges are dual-to-primal; 3 generations of hodges are used.
modulating each of these gives perhaps more degrees of freedom than material parameters?
not really; mu is on faces, lambda on edges, and rho on cubes
seems like quite similar strategies in the end
however, modulating them in a consistent way with a single scaling might be beneficial?
no wait is it more subtle? lambda mu rho all appear  with mutliplication in momentum equation
however, laplace-beltrami contains hodges in both directions, after rewriting
with different scalings for volume, area or line elements tho; maybe this cancels out?

does it make sense to coarsen by just averaging the material parameters over their cells?
physically it seems ok; half filled cell has half the mass and stiffness;
but what about closing gaps?
it only makes sense in the tangential direction of the surface really;
there the properties act and add in parallel
in the normal direction the materials are linked in series; higher compliance in the fine cells
ought to dominate the response of the coarse cell
so the question can be phrased as: should we average compliances or stiffnesses?
most likely there isnt a correct answer
how much this matters probably also depends on the smoother and other aspects of MG
if restriction/prolongation is based on smoothing on fine level, it should compensate for this effect somewhat
also, integration coarse correction in a more nuanced way than just subtracting;
treating it as a direction on the fine grid and rescaling it there for optimum effect should help;
maybe it would even be enough to solve the topology change problem? doubtfull of that
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


def lame_elasticity(complex2):
    """Formulate elasticity over a 2-complex"""

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
    P2D0 = sparse_diag(complex2.hodge_PD[2])

    # FIXME: autogen this given system shape
    P2D2_0 = sparse_zeros((P2, D2))
    P1D1_0 = sparse_zeros((P1, D1))
    P0D0_0 = sparse_zeros((P0, D0))
    P2D0_0 = sparse_zeros((P2, D0))

    S = complex2.topology.dual.selector

    rotation    = [P0D2       , P0D2 * D2D1       , P0D0_0       ]
    momentum    = [P1P0 * P0D2, P1D1_0            , P1D1 * D1D0  ]
    compression = [P2D2_0     , P2P1 * P1D1 * S[1], P2D0         ]

    # set up boundary equations; default to zeros
    compression_bc = [sparse_zeros((B0, d)) for d in dual.n_elements[::-1]]
    momentum_bc = [sparse_zeros((B1, d)) for d in dual.n_elements[::-1]]
    # force pressure at free boundary to zero
    compression_bc[2] = d_matrix(inlet + outlet, compression_bc[2].shape, P2)
    # force shear at free boundary to zero
    momentum_bc[0] = o_matrix(inlet + outlet + closed, boundary.parent_idx[0], momentum_bc[0].shape)

    equations = [
        rotation,
        momentum,
        momentum_bc,
        compression,
        compression_bc,
    ]

    omega    = np.zeros(D2)
    displacement = np.zeros(D1)
    pressure = np.zeros(D0)
    unknowns = [
        omega,
        displacement,
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

system = lame_elasticity(mesh)
