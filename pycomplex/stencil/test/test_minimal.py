
import matplotlib.pyplot as plt
import scipy.ndimage
from pycomplex.stencil.minimal import StencilComplexSimple2d, SimpleHeatEquation2d
from pycomplex.stencil.multigrid import solve_full_cycle




def test_2d():
    shape = 512, 512
    complex = StencilComplexSimple2d(shape=shape)

    source = complex.form(0)
    source[0, 32:-32, 32:-32] = 1
    source[0, 128:-64, 0:-64] = 0
    source = scipy.ndimage.gaussian_filter(source, [0, 0.5, 0.5])
    constraint = 1 - source

    equation = SimpleHeatEquation2d(complex, source=source, constraint=constraint * 1e-1)


    if True:
        solution = equation.solve(source, iterations=100)
        complex.plot_0(solution)
        plt.show()


    hierarchy = [equation]

    for i in range(4):
        hierarchy.append(hierarchy[-1].coarsen())
        # source = hierarchy[-2].restrict(source)

    hierarchy = hierarchy[::-1]

    if False:
        hierarchy[0].complex.plot_0(source)
        plt.show()


    if True:
        solution = solve_full_cycle(hierarchy, source)
        complex.plot_0(solution)
        plt.show()
