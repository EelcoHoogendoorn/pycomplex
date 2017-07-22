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




probably a fine start

but can we simplify this into a system involving only a streamfunction phi?
we note that expressing v as curl phi fails to prescribe vd; so this substitution is incomplete
can we augment the curl operator with an I passthrough in lower right corner?
and if we multiplied with the same op from the left, what would happen to the [vp, Pd] equations?
extra

since v in incompressible, we can write it as the curl of a potential
v = curl phi
putting curl in front of force equation too for symmetry

[I,    curl, 0   ] [O]   [0]
[curl, 0,    grad] [dphi] = [f]
[0,    div,  0   ] [P]   [0]

bottom eq drops out since div curl = 0; right column drops out since curl grad = 0





"""
import numpy as np
import scipy.sparse

def grid():
    from pycomplex import synthetic
    mesh = synthetic.n_cube_grid((32, 32))
    return mesh.as_22().as_regular()

def concave():
    from pycomplex.complex.regular import ComplexRegular2
    vertices = np.indices((3, 3))
    vertices = vertices.reshape(2, -1).T[:-1]
    quads = [
        [[0, 1],
         [3, 4]],
        [[1, 2],
         [4, 5]],
        [[3, 4],
         [6, 7]],
    ]

    mesh = ComplexRegular2(vertices=vertices, cubes=quads)
    for i in range(5):
        mesh = mesh.subdivide()
    return mesh

mesh = concave()
mesh.metric()


from examples.spherical_harmonics import get_harmonics_0, get_harmonics_2

v = get_harmonics_2(mesh)
q = mesh.to_simplicial_transfer_2(v[:, -10])
mesh.to_simplicial().as_2().plot_primal_2_form(q)



v = get_harmonics_0(mesh)
print(v[:, 5])
q = mesh.to_simplicial_transfer_0(v[:, 7])
mesh.to_simplicial().as_2().plot_primal_0_form(q)
