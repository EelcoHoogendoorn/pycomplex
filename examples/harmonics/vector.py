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
laplace = System.canonical(mesh).laplace(1)
laplace.plot()
