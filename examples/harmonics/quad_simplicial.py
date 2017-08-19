
import matplotlib.pyplot as plt
from pycomplex import synthetic
from examples.harmonics import  *

quad = synthetic.delaunay_cube(10, 2)
quad = quad.optimize_weights()
quad = quad.as_2().as_2()


v = get_harmonics_0(quad)
f0 = v[:, 8]
quad.plot_primal_0_form(f0, plot_contour=False, shading='gouraud')
quad.plot(ax=plt.gca())
plt.show()


# v = get_harmonics_2(quad)
# f2 = v[:, 12]
# # do plotting via mapping to simplicial complex
# quad.plot_dual_0_form_interpolated(f2, weighted=False, plot_contour=False, shading='gouraud')
# quad.plot_primal_2_form(f2)
#
# plt.show()