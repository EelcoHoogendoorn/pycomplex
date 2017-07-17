import numpy as np

from pycomplex.complex.cubical import ComplexCubical
from pycomplex import synthetic
from pycomplex.math import linalg


def example_letter_a():
    """Create a letter 'a' using subdivision

    Notes
    -----
    Recreates figure 8 of [1]

    References
    ----------
    [1] https://pdfs.semanticscholar.org/475a/42a1b193d46c1c2f7a02b94f5f98e7c6098c.pdf
    """
    if False:
        # handcrafted example from paper
        vertices = [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 4],
        ]
        quads = [
            [[0, 1], [4, 5]],
            [[1, 2], [5, 6]],
            [[2, 3], [6, 7]],
            [[5, 6], [9, 10]],
            [[3, 15], [7, 11]],
            [[0+8, 1+8], [4+8, 5+8]],
            [[1+8, 2+8], [5+8, 6+8]],
            [[2+8, 3+8], [6+8, 7+8]],
        ]
        quads = ComplexCubical(vertices=vertices, cubes=quads)

    else:
        quads = synthetic.n_cube_grid((3, 4))
        s = np.ones((3, 4), dtype=np.bool)
        s[1][[0, 2]] = 0

        quads = ComplexCubical(
            vertices=quads.vertices,
            cubes=quads.topology.elements[-1][s.flatten()]
        )
    quads.as_22().plot(plot_dual=False)

    segment = synthetic.n_cube(1)

    grid = quads.product(segment)

    grid = grid.boundary()

    # get bottom face chain
    fp = grid.primal_position()[2]
    fi = np.argsort(fp[:, 1])[:2]
    fc = grid.topology.chain(2, 0)
    fc[fi] = 1
    ec = grid.topology.matrix(1, 2) * fc

    # propagate this chain along the subdivision
    grid = grid.subdivide(smooth=True, creases={1:ec})
    ec = grid.topology.transfer_matrices[1] * ec
    grid = grid.subdivide(smooth=True, creases={1:ec})
    ec = grid.topology.transfer_matrices[1] * ec
    grid = grid.subdivide(smooth=True, creases={1:ec})
    ec = grid.topology.transfer_matrices[1] * ec


    grid = grid.as_23().to_simplicial()#.smooth().smooth()

    grid.vertices = np.dot(grid.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    grid.as_3().plot_3d(plot_dual=False, plot_vertices=False)


def example_sphere():
    sphere = synthetic.icosahedron().as_euclidian()
    # map sphere complex to triangle complex
    sphere.vertices = np.dot(sphere.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

    crease1 = sphere.topology.chain(1, fill=0)
    crease1[0] = 1
    crease0 = sphere.topology.matrix(0, 1) * crease1
    print(crease0)

    for i in range(3):
        sphere = sphere.subdivide(smooth=True, creases={0: crease0, 1: crease1})
        crease1 = sphere.topology.transfer_matrices[1] * crease1
        crease0 = sphere.topology.transfer_matrices[0] * crease0
        print(crease0)

    sphere.plot_3d(backface_culling=True)

# example_letter_a()
example_sphere()