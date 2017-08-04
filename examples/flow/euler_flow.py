"""
Euler equations:
    Dω / dt = 0     (material derivative of vorticity is zero; or constant along streamlines)
    div(u) = 0      (incompressible)
    ω = curl(u)

This example is an implementation of the excellent paper 'Stable, Circulation-Preserving, Simplicial Fluids'
We depart from it in some aspects; primarily, the use of subdomain-interpolation rather than barycentric interpolation.
This should have as advantage that

References
----------
[1] http://www.geometry.caltech.edu/pubs/ETKSD07.pdf
"""

from pycomplex import synthetic
import matplotlib.pyplot as plt
sphere = synthetic.icosphere(refinement=3)

sphere.plot()

# get initial conditions; solve for incompressible irrotational field
from examples.harmonics import get_harmonics_1, get_harmonics_0
# H = get_harmonics_1(sphere)[:, 0]
# or maybe incompressible but with rotations?
H = get_harmonics_0(sphere)[:, -2]

sphere.as_euclidian().as_3().plot_primal_0_form(H, plot_contour=False, cmap='bwr', shading='gouraud')
plt.show()


# interpolate flux to primal-2-form
# cascade down to lower forms
# pick velocity at dual 0-form
# advect the dual mesh
# integrate the tangent flux of the advected mesh
# that is; sample at the midpoints of all advected dual edges
# advected vorticity is the curl of this
# solve for a new flow field
# potentially add BFECC step
# do momentum diffusion, if desired

