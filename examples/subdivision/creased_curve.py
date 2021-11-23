"""Demonstrate 1d curve embedded in 2-space with a single crease point"""

from pycomplex import synthetic

quad = synthetic.n_cube(2)
curve = quad.boundary
root = curve
# make one vertex a crease vertex
crease = curve.topology.chain(0)
crease[0] = 1
# accumulate a total subdivision operator through all subdivision steps
total_operator = 1
for i in range(5):
    operator = curve.subdivide_operator(smooth=True, creases={0: crease})
    total_operator = operator * total_operator
    curve = curve.subdivide_cubical(smooth=True, creases={0: crease})
    crease = curve.topology.transfer_matrices[0] * crease


curve.as_12().plot(plot_vertices=False)
# overwrite vertices with operator based ones to demonstrate equivalence
# note that this is useful in the context of realtime editing of subdivision surfaces
curve = curve.copy(vertices=total_operator * root.vertices)
curve.as_12().plot(plot_vertices=False)
import matplotlib.pyplot as plt
plt.show()
