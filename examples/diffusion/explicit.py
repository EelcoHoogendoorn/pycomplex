"""Time-dependent diffusion is interesting because it arguably represents
the simplest 'introduction to vector calculus' possible, while still doing something physical and useful.

"""

import numpy as np
import scipy.sparse

kind = 'letter'

if kind == 'sphere':
    from pycomplex import synthetic
    grid = synthetic.icosphere(refinement=6)
    grid.metric()
if kind == 'regular':
    from pycomplex import synthetic
    grid = synthetic.n_cube_grid((32, 32)).as_22().as_regular()
    for i in range(2):
        grid = grid.subdivide()
    grid.metric()
if kind == 'letter':
    from examples.subdivision import letter_a
    grid = letter_a.create_letter(4).to_simplicial().as_3()
    grid.vertices *= 10
    grid.metric()

print(grid.box)
assert grid.topology.is_oriented

def laplacian_0(complex):
    T01 = complex.topology.matrices[0]
    grad = T01.T
    div = T01

    D1P1 = scipy.sparse.diags(complex.D1P1)
    D2P0 = scipy.sparse.diags(complex.D2P0)
    P0D2 = scipy.sparse.diags(complex.P0D2)

    # construct our laplacian
    laplacian = div * D1P1 * grad

    largest_eigenvalue = scipy.sparse.linalg.eigsh(laplacian, M=D2P0, k=1, which='LM', tol=1e-6, return_eigenvectors=False)
    return P0D2 * laplacian, largest_eigenvalue


L, v = laplacian_0(grid)
print(L.shape, v)

field = np.random.rand(grid.topology.n_elements[0])
for i in range(3000):
    field = field - L * field / v

if kind == 'sphere':
    grid = grid.as_euclidian()
    grid.plot_primal_0_form(field)
if kind == 'regular':
    tris = grid.to_simplicial().as_2()
    field = grid.as_22().to_simplicial_transfer_0(field)
    tris.plot_primal_0_form(field)
if kind == 'letter':
    grid.plot_primal_0_form(field, plot_contour=False)
