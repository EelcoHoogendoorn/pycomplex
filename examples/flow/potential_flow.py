
# -*- coding: utf-8 -*-

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

from time import clock
import scipy.sparse

from examples.linear_system import *


def grid(shape):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()


def setup_mesh(levels=4):
    # generate a mesh
    mesh = grid(shape=(3, 3))
    # make a hole in it
    mask = np.ones((3, 3), dtype=np.int)
    mask[1, 1] = 0
    mesh = mesh.select_subset(mask.flatten())

    hierarchy = [mesh]
    # subdivide
    for i in range(levels):
        mesh = mesh.subdivide_cubical()
        hierarchy.append(mesh)
    return mesh, hierarchy


def setup_domain(mesh):
    """Construct domain

    Parameters
    ----------
    mesh : Complex

    Returns
    -------
    chains
        boundary and body description chains
    """

    # identify boundaries
    PP = mesh.primal_position
    BPP = mesh.boundary.primal_position
    left_1  = (BPP[1][:, 0] == BPP[1][:, 0].min()).astype(sign_dtype)
    right_1 = (BPP[1][:, 0] == BPP[1][:, 0].max()).astype(sign_dtype)

    bottom_1 = (BPP[1][:, 1] == BPP[1][:, 1].min()).astype(sign_dtype)

    # right = (BPP[1][:, 0] == BPP[1][:, 0].max()).astype(sign_dtype)
    # construct closed part of the boundary
    all_1 = mesh.boundary.topology.chain(1, fill=1)

    left_0  = (BPP[0][:, 0] == BPP[0][:, 0].min()).astype(sign_dtype)
    right_0 = (BPP[0][:, 0] == BPP[0][:, 0].max()).astype(sign_dtype)

    bottom_0 = (BPP[0][:, 1] == BPP[0][:, 1].min()).astype(sign_dtype)
    bottom_0 = bottom_0 * (1-left_0) * (1-right_0)

    # 0-element boundary
    interior_0 = (np.linalg.norm(BPP[0], axis=1) < 1).astype(sign_dtype)    # inside the doughnut

    all_0 = mesh.boundary.topology.chain(0, fill=1)
    top_right_0 = mesh.boundary.topology.chain(0, fill=0)
    top_right_0[np.argmin(np.linalg.norm(BPP[0]-[0.5,0.5], axis=1))] = 1

    exterior_0 = all_0 - interior_0

    return dict(
        all_1=all_1,
        left_1=left_1,
        right_1=right_1,
        bottom_0=bottom_0,
        bottom_1=bottom_1,
        interior_0=interior_0,
        exterior_0=exterior_0,
        all_0=all_0,
        top_right_0=top_right_0,

    )


def setup_potential_flow(complex, regions):
    """new style potential flow setup

    Parameters
    ----------
    complex : Complex
        complex with topological dimension > 2
    regions : dict[str, ndarray]
        dict of chains specifying regions of interest

    Returns
    -------
    system : System
        complete first order system describing potential flow problem

    Notes
    -----
    The mathematical structure of this system is wholly identical to magnetostatics/electrostatics.
    The only differences arise insofar different boundary conditions are used
    """
    assert complex.topology.n_dim >= 2  # makes no sense in lower-dimensional space
    # unknown is a dual 1-form; two equations to close from from both sides
    system = System.canonical(complex)[-3:, :][:, [-2]]
    equations = dict(rotation=0, flux=1, divergence=2)  # should these names be tracked by system?
    variables = dict(flux=0)

    system.A.block[equations['flux'], variables['flux']] *= 0   # flux is mostly unspecified

    # NOTE: boundary conditions begin here

    # impose a circulation around both boundaries
    # cannot set tangent directly; important that we specify the sum
    # FIXME: make a less hacky interface for these circulation bcs on system class
    system.set_dia_boundary(equations['flux'], variables['flux'], regions['interior_0'], rows=0)
    q = np.zeros_like(regions['all_1'])
    q[0] = -1e2
    q[1] = 0#+1e9
    system.set_rhs_boundary(equations['flux'], q)
    # no point setting a second constraint; circulation should be preserverd
    # system.set_dia_boundary(equations['flux'], variables['flux'], regions['exterior_0'], rows=1)

    # activate normal flux constraint everywhere
    system.set_off_boundary(equations['divergence'], variables['flux'], regions['all_1'])
    system.set_rhs_boundary(equations['divergence'], regions['left_1'] - regions['right_1'])

    return system


if __name__ == '__main__':
    mesh, hierarchy = setup_mesh()
    regions = setup_domain(mesh)
    # mesh.plot(plot_dual=False, plot_vertices=False)

    system = setup_potential_flow(mesh, regions)
    # system.plot()
    system = system.balance(1e-9)   # NOTE: this first-order preconditioning solves about 30 times faster than previous method
    normal = system.normal()
    # normal.plot()

    t = clock()
    solution, res = normal.solve_minres()
    print('solution time:', clock() - t)

    flux = solution.merge()

    from examples.flow.stream import setup_stream, solve_stream
    phi = solve_stream(setup_stream(mesh), flux)
    mesh.plot_primal_0_form(phi - phi.min(), cmap='jet', plot_contour=True, levels=29)
    import matplotlib.pyplot as plt
    plt.show()
    quit()


