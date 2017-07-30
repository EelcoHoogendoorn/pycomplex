"""Demonstrate 1d curve embedded in 2-space with a crease point"""

from pycomplex import synthetic

quad = synthetic.n_cube(2)
curve = quad.boundary()
# make one vertex a crease vertex
crease = curve.topology.chain(0)
crease[0] = 1
for i in range(5):
    curve = curve.subdivide(smooth=True, creases={0: crease})
    crease = curve.topology.transfer_matrices[0] * crease

curve.as_12().plot(plot_vertices=False)
