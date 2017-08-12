"""Example illustrating convection between plates with a temperature differential. or Rayleigh–Bénard convection

Note that this example does not really introduce any new functionality; it merely add buoyancy forces to an euler flow
"""
# FIXME: add weighted averaging for regular grids
# FIXME: add boundary handling to dual averaging

import matplotlib.pyplot as plt

from examples.advection import Advector
from examples.diffusion.explicit import Diffusor
from examples.diffusion.perlin_noise import perlin_noise
from examples.flow.euler_flow import VorticityAdvector
from examples.util import save_animation
from pycomplex import synthetic
from pycomplex.math import linalg

# set up grid
grid = synthetic.n_cube_grid((4, 1), False)
for i in range(6):
    grid = grid.subdivide()

grid = grid.as_22().as_regular()
grid.topology.check_chain()
tris = grid.to_simplicial()

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
        (.05, .05),
        (.1, .1),
        # (.2, .2),
        # (.4, .4),
    ]
)


flux_d1 = grid.topology.dual.chain(n=1)


vorticity_advector = VorticityAdvector(grid, diffusion=2e-4)
edges_d1 = grid.topology.dual.matrices_2[0].T * grid.dual_position[0]


path = r'c:\development\examples\rayleigh–benard_8'

dt = 0.005
gravity = [0, -1000]
for i in save_animation(path, frames=1000, overwrite=True):

    temperature_p0[top] = 0
    temperature_p0[bottom] = 1
    temperature_p0 = temperature_diffusor.integrate_explicit(temperature_p0, dt=dt*5e-3)
    temperature_p0 = temperature_advector.advect_p0(flux_d1, temperature_p0, dt=dt)

    force_p0 = temperature_p0[:, None] * [gravity]
    force_edge = grid.topology.averaging_operators[1] * force_p0
    force_d1 = linalg.dot(edges_d1, force_edge)

    # advect vorticity
    def advect(flux_d1, dt):
        return vorticity_advector.advect_vorticity(flux_d1, dt, force=force_d1)

    # flux_d1 = BFECC(advect, flux_d1, dt=dt)
    flux_d1 = advect(flux_d1, dt=dt)

    # plot temperature field
    form = tris.topology.transfer_operators[0] * temperature_p0
    tris.as_2().plot_primal_0_form(form, plot_contour=False, cmap='magma')
    plt.axis('off')
