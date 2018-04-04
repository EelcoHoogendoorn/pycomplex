
# -*- coding: utf-8 -*-

"""

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
[[I, 0], [δ, 0, 0], [0, 0]] [ωi]   [mi]
[[0, I], [δ, b, I], [0, 0]] [ωp]   [mp]

[[d, d], [0, 0, 0], [δ, 0]] [vi]   [fi]
[[0, b], [0, 0, 0], [I, I]] [vp] = [fp]
[[0, _], [0, 0, _], [0, 0]] [vd]   [_]

[[0, 0], [d, I, 0], [0, 0]] [Pi]   [si] source/sink
[[0, 0], [0, _, 0], [0, _]] [Pd]   [_]

This implies a relation between [vp, Pd] and [ωp, vd] if we seek to restore symmetry to the system

normalize bcs with potential infs on the diag
drop the infs by giving them prescribed values
we now have a symmetric well posed problem that we can feed to minres (P may still have a gauge)
this merely allows us to see a subset of boundary conditions that is provably consistent;
does not provably give us all possible consistent boundary conditions

However, rather than making the first order equation symmetrical, we can simply solve the normal equations.
This is at least conceptually even simpler

in 3d, there is such a thing as vorticity on the dual boundary
is it strange that it does not seem to matter in any way?
think this is fine; we can just set that diag to identity and it will not influence anything

[[I, 0, 0], [δ, 0, 0], [0, 0]] [ωi]   [mi]
[[0, I, 0], [δ, b, I], [0, 0]] [ωp]   [mp]
[[0, 0, 0], [0, 0, 0], [0, 0]] [ωd]   [mp]

[[d, d, 0], [0, 0, 0], [δ, 0]] [vi]   [fi]
[[0, b, 0], [0, 0, 0], [I, I]] [vp] = [fp]
[[0, _, 0], [0, 0, _], [0, 0]] [vd]   [_]

[[0, 0, 0], [d, I, 0], [0, 0]] [Pi]   [si] source/sink
[[0, 0, 0], [0, _, 0], [0, _]] [Pd]   [_]


Note that the equations discussed so far describe incompressible time-invariant stokes flow
Compressible time-variant stokes flow would use every 'slot' available on the 2d chain complex,
and has the highest complexity, as measured by the number of terms in our equations.
However, since it is mathematically more diagonally dominant,
and its physics more local, it should be numerically easier to solve, too.

Furthermore, note that by reinterpreting the flux as a displacement,
we can also use this as a model for isotropic linear elasticity

"""

import matplotlib.pyplot as plt

from examples.linear_system import *


def grid(shape=(32, 32)):
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid(shape)
    return mesh.as_22().as_regular()


def concave(levels=3):
    # take a 2x2 grid
    mesh = grid(shape=(2, 2))
    # discard a corner
    mesh = mesh.select_subset([1, 1, 1, 0])

    for i in range(levels):  # subdiv 5 is already pushing our solvers...
        mesh = mesh.subdivide_cubical()
        # left = mesh.topology.transfer_matrices[1] * left
        # right = mesh.topology.transfer_matrices[1] * right
    return mesh


def get_regions(mesh):
    """Identify (boundary) regions of the mesh

    Returns
    -------
    dict[chain]
    """
    # identify sides of the domain
    BP1 = mesh.boundary.primal_position[1]
    left_1  = (BP1[:, 0] == BP1[:, 0].min()).astype(sign_dtype)
    right_1 = (BP1[:, 0] == BP1[:, 0].max()).astype(sign_dtype)
    # construct closed part of the boundary
    all_1 = mesh.boundary.topology.chain(1, fill=1)
    all_0 = mesh.boundary.topology.chain(0, fill=1)
    closed_1 = all_1 - left_1 - right_1

    # FIXME: 0-1 nomenclature is ndim-specific
    return dict(
        left_1=left_1,
        right_1=right_1,
        closed_1=closed_1,
        all_1=all_1,
        all_0=all_0
    )


def stokes_system(complex, regions):
    """New style stokes system setup"""
    assert complex.topology.n_dim >= 2  # makes no sense in lower-dimensional space
    system = System.canonical(complex)[-3:, -3:]
    equations = dict(vorticity=0, momentum=1, continuity=2)
    variables = dict(vorticity=0, flux=1, pressure=2)

    # FIXME: encapsulate this direct access with some modifier methods
    system.A.block[equations['momentum'], variables['flux']] *= 0   # no direct force relation with flux
    system.A.block[equations['continuity'], variables['pressure']] *= 0   # no compressibility

    # NOTE: setup pipe flow with step diameter change
    # prescribe tangent flux on entire boundary
    system.set_dia_boundary(equations['momentum'], variables['flux'], regions['all_0'])
    # set normal flux to zero
    system.set_off_boundary(equations['continuity'], variables['flux'], regions['closed_1'])
    # prescribe pressure on inlet and outlet
    inlet, outlet = regions['left_1'], regions['right_1']
    system.set_dia_boundary(equations['continuity'], variables['pressure'], inlet + outlet)
    system.set_rhs_boundary(equations['continuity'], outlet.astype(np.float) - inlet.astype(np.float))

    return system


if __name__ == '__main__':
    mesh = concave(levels=2)
    regions = get_regions(mesh)
    # mesh.plot()
    system = stokes_system(mesh, regions)

    # system = system.balance(1e-9)
    system.plot()

    if True:
        # FIXME: not working yet; should be possible
        # without vorticity constraint, we have an asymmetry in the [1,0] block
        # is this a problem for elimination? yes it is; gmres instead of minres works fine
        # just need to symmetrize bcs first before elimination
        system_up = system.eliminate([0], [0])
        # system_up.plot()
        solution, residual = system_up.solve_minres()
        flux = solution[-2].merge()
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
