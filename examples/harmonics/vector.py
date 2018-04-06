"""Demonstate vectorial harmonics"""
from examples.linear_system import System

import matplotlib.pyplot as plt

from pycomplex import synthetic


def setup_mesh(levels=3):
    # generate a mesh
    mesh = synthetic.n_cube(n_dim=2).as_22().as_regular()
    hierarchy = [mesh]
    # subdivide
    for i in range(levels):
        mesh = mesh.subdivide_cubical()
        hierarchy.append(mesh)
    return mesh, hierarchy


mesh, hierarchy = setup_mesh()
# 1-form laplacian with default boundary conditions
laplace = System.laplace(mesh, k=1, dirichlet_dual=True, dirichlet_primal=True)
# this works alright for laplace with default boundaries.
# elimination only preserves symmetry though, and does not produce it
# elimination still fails for dirichlet_primal=True
laplace = laplace.eliminate([0, 2], [0, 2])
# test symmetry of the eqs
assert laplace.is_symmetric

# laplace = laplace.balance()
# there is something appealing about normal equations, but laplacian form fits neater into eigen framework
# can we create a linear operator that represents the action of the eliminated laplacian?
# laplacian maps displacements to forces
# eigenvectors of the first order system; does that even make sense? only if we can make it symmetrical,
# but then again that is also a precondition for ending with a symmetric system after elimination
# so the crux is to render the first order system symmetric
# either way, we could preconditon an eigendecomposition on the second order eliminated system,
# by inverting the first order normal equations using mg, padding the target rhs with zeros
laplace.plot()
