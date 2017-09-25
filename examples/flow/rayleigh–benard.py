
# -*- coding: utf-8 -*-

"""Example illustrating convection between plates with a temperature differential. or Rayleigh–Bénard convection

Note that this example does not really introduce any new functionality; it merely add buoyancy forces to an euler flow
"""
import numpy as np
import matplotlib.pyplot as plt

from pycomplex import synthetic
from pycomplex.math import linalg

from examples.diffusion.explicit import Diffusor
from examples.diffusion.perlin_noise import perlin_noise
from examples.flow.advection import Advector, BFECC
from examples.flow.euler_flow import VorticityAdvector
from examples.util import save_animation


# set up grid
grid = synthetic.n_cube_grid((4, 1), False)
for i in range(6):
    grid = grid.subdivide_cubical()

grid = grid.as_22().as_regular()
grid.topology.check_chain()
tris = grid.subdivide_simplicial()

if False:
    grid.plot()
# find driven edges
x, y = grid.vertices.T
bottom = y == y.min()
top = y == y.max()


temperature_diffusor = Diffusor(grid)
temperature_advector = Advector(grid)

# give some initial temperature disturbance, to speed up startup
temperature_p0 = perlin_noise(
    grid,
    [
        (.01, .01),
        (.02, .02),
        (.04, .04),
        # (.4, .4),
    ]
)


flux_d1 = grid.topology.dual.chain(n=1)


vorticity_advector = VorticityAdvector(grid, diffusion=1e-4)
edges_d1 = grid.topology.dual.matrices_2[0].T * grid.dual_position[0]


path = r'c:\development\examples\rayleigh–benard_22'

dt = 0.002
gravity = [0, -4000]
for i in save_animation(path, frames=1000, overwrite=True):

    temperature_p0[top] = 0
    temperature_p0[bottom] += 1
    temperature_p0 = temperature_diffusor.integrate_explicit(temperature_p0, dt=dt*4e-3)

    # advect temperature
    def advect(temperature_p0, dt):
        return np.clip(temperature_advector.advect_p0(flux_d1, temperature_p0, dt=dt), 0, 1)

    # temperature_p0 = advect(temperature_p0, dt=dt)
    temperature_p0 = BFECC(advect, temperature_p0, dt=dt)


    force_p0 = temperature_p0[:, None] * [gravity]
    force_edge = grid.topology.averaging_operators_0[1] * force_p0
    force_d1 = linalg.dot(edges_d1, force_edge)

    # advect vorticity
    def advect(flux_d1, dt):
        return vorticity_advector.advect_vorticity(flux_d1, dt, force=force_d1)
    # cant use BFECC or the likes if diffusion is part of the update step
    # flux_d1 = BFECC(advect, flux_d1, dt=dt)
    flux_d1 = advect(flux_d1, dt=dt)

    # plot temperature field
    form = tris.topology.transfer_operators[0] * temperature_p0
    tris.as_2().plot_primal_0_form(form, plot_contour=False, cmap='magma', shading='gouraud', vmin=0, vmax=1)
    plt.axis('off')
