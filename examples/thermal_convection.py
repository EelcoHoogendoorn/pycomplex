"""Example illustrating convection between plates with a temperature differential. or Rayleigh–Bénard convection
"""
# FIXME: get picking working on regular grids
# FIXME: add weighted averaging for regular grids
# FIXME: add boundary handling to dual averaging

import numpy as np

from pycomplex import synthetic
from pycomplex.util import save_animation

from examples.flow.euler_flow import VorticityAdvector
from examples.harmonics import get_harmonics_0
from examples.advection import MacCormack, BFECC, Advector
from examples.diffusion.explicit import Diffusor


# set up grid
grid = synthetic.n_cube_grid((4, 1), False)
for i in range(3):
    grid = grid.subdivide()

grid = grid.as_22().as_regular()
grid.topology.check_chain()
tris = grid.to_simplicial()

if True:
    grid.plot()
# find driven edges
x, y = grid.vertices.T
bottom = y == y.min()
top = y == y.max()


temperature_diffusor = Diffusor(grid)
temperature_advector = Advector(grid)

temperature_p0 = grid.topology.chain(0, fill=0)


H = get_harmonics_0(grid)[:, 2]

T01, T12 = grid.topology.matrices
curl = T01.T
flux_p1 = curl * H
flux_d1 = grid.hodge_DP[1] * flux_p1


vorticity_advector = VorticityAdvector(grid)

# test that integrating over zero time does almost nothing
advected_0 = vorticity_advector.advect_vorticity(flux_d1, dt=0)
print(np.abs(advected_0 - flux_d1).max())
print(np.abs(flux_d1).max())
assert np.allclose(advected_0, flux_d1)


def advect(flux_d1, dt):
    return vorticity_advector.advect_vorticity(flux_d1, dt)


path = r'c:\development\examples\rayleigh–benard_1'
# path = None

dt = 0.01
for i in save_animation(path, frames=10):

    temperature_p0[top] = 0
    temperature_p0[bottom] = 1
    temperature_p0 = temperature_diffusor.integrate_explicit(temperature_p0, dt=dt)
    temperature_p0 = temperature_advector.advect_p0(flux_d1, temperature_p0, dt=dt)

    flux_d1 = vorticity_advector.advect_d0(flux_d1, dt=dt)
    # diffuse momentum, integrate body forces, and apply momentum BC's

    form = tris.topology.transfer_operators[0] * temperature_p0
    tris.as_2().plot_primal_0_form(form, plot_contour=False)
