"""Demonstrate subdivision curves"""
from pycomplex import synthetic


def test_curve():
    """Test 1d curve embedded in 2-space"""
    quad = synthetic.n_cube(2)
    curve = quad.boundary()
    crease = curve.topology.chain(0, fill=0)
    crease[0] = 1
    for i in range(5):
        curve = curve.subdivide(smooth=True, creases={0: crease})
        crease = curve.topology.transfer_matrices[0] * crease

    curve.as_12().plot(plot_vertices=False)
