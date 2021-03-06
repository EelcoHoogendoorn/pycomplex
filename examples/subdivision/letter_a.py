"""Create a letter 'a' using subdivision, illustrating the concept of 'creases' [1]

Two approaches are taken; both direct subdivision of the mesh is employed,
as well as constructing a series of sparse matrices mapping between the vertices at each level.
Concatenating these sparse matrices allows for super efficient animations of the subdivision surface,
as the original control points are manipulated.

Notes
-----
Recreates figure 8 of [1]

References
----------
[1] https://pdfs.semanticscholar.org/475a/42a1b193d46c1c2f7a02b94f5f98e7c6098c.pdf
"""

import numpy as np

from pycomplex.complex.cubical import ComplexCubical
from pycomplex import synthetic
from pycomplex.math import linalg


def create_letter_2d(handcrafted=True):
    if handcrafted:
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
            [[5, 6], [9, 10]],      # mid piece
            [[3, 15], [7, 11]],     # top piece
            [[0+8, 1+8], [4+8, 5+8]],
            [[1+8, 2+8], [5+8, 6+8]],
            [[2+8, 3+8], [6+8, 7+8]],
        ]
        quads = ComplexCubical(vertices=vertices, cubes=quads)

    else:
        quads = synthetic.n_cube_grid((3, 4))
        s = np.ones((3, 4), dtype=np.bool)
        s[1][[0, 2]] = 0
        quads = quads.select_subset(s.flatten())

    print(quads.topology.is_oriented)
    return quads.copy(vertices=quads.vertices - quads.vertices.mean(axis=0))


def create_letter_3d(quads, subdivisions=2):

    # extract quads to cubes
    segment = synthetic.n_cube(1)
    # get boundary of cubes
    grid = quads.product(segment).boundary
    # product preserves orientation
    print(grid.topology.is_oriented)

    # get bottom face chain
    fp = grid.primal_position[2]
    fi = np.argsort(fp[:, 1])[:2]
    fc = grid.topology.chain(2, 0)
    fc[fi] = 1
    # get faces around hole
    fi = np.argsort(np.linalg.norm(fp - grid.vertices.mean(axis=0) - [0, 1, 0], axis=1))[:4]
    fc[fi] = 1
    c1 = grid.topology.matrix(1, 2) * fc
    c0 = grid.topology.chain(0)


    if False:
        # test interaction between crease-types
        c0[0] = 1

    total_operator = 1
    original = grid
    # subdivide; propagate crease along the subdivision
    for i in range(subdivisions):
        operator = grid.subdivide_operator(smooth=True, creases={0: c0, 1: c1})
        total_operator = operator * total_operator

        grid = grid.subdivide_cubical(smooth=True, creases={0: c0, 1: c1})
        c1 = grid.topology.transfer_matrices[1] * c1
        c0 = grid.topology.transfer_matrices[0] * c0

    if False:
        import matplotlib.pyplot as plt
        t = total_operator.tocoo()
        plt.scatter(t.row, t.col, c=t.data)
        # plt.axis('equal')
        plt.show()

    # test if operator approach gives identical results
    # grid = grid.copy(vertices=total_operator * original.vertices)

    return grid.as_23()


def create_letter(subdivisions):
    return create_letter_3d(create_letter_2d(handcrafted=True), subdivisions=subdivisions)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    letter = create_letter_2d()
    if False:
        letter.plot(plot_dual=True)

    # add random rotation
    np.random.seed(7)
    rotation = linalg.power(linalg.orthonormalize(np.random.randn(3, 3)), 0.2)

    if True:
        from examples.util import save_animation
        path = r'../output/letter_a'
        plt.close('all')
        for i in save_animation(path, frames=4, overwrite=True):
            fix, ax = plt.subplots(1, 1, figsize=(4,6))
            letter = create_letter(subdivisions=i)
            print(letter.box)
            letter = letter.transform(rotation)
            letter.plot(plot_dual=False, plot_vertices=False, ax=ax)
            plt.gca().set_adjustable("box")
            ax.set_ylim(-3, 3)
            ax.set_xlim(-2, 2)
            plt.axis('off')
        quit()


    letter = create_letter(subdivisions=2)

    letter = letter.transform(rotation)

    letter.plot(plot_dual=False, plot_vertices=False)

    # turns out fundamental-simplicial subdivision is the surest method for getting positive duals
    letter = letter.subdivide_fundamental()#.smooth()#.smooth()

    # letter = letter.optimize_weights_metric()
    # letter = letter.as_3().optimize_weights()
    letter = letter.optimize_weights_fundamental()
    print(letter.weights)

    letter.as_2().as_3().plot_3d(plot_dual=True, plot_vertices=False)

    PM, DM = letter.metric
    plt.scatter(*letter.vertices[DM[2]<0][:, :2].T, c='k')
    for i, m in enumerate(PM):
        print(i)
        print(m.min(), m.max())
    for i, m in enumerate(DM):
        print(i)
        print(m.min(), m.max())
    plt.figure()
    plt.hist(DM[2], bins=100)
    plt.show()
