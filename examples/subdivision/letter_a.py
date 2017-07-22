"""Create a letter 'a' using subdivision

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
    # FIXME: construction isnt perfect; currently we fail to plot the dual on this

else:
    quads = synthetic.n_cube_grid((3, 4))
    s = np.ones((3, 4), dtype=np.bool)
    s[1][[0, 2]] = 0
    quads = quads.select_subset(s.flatten())


quads.as_22().plot(plot_dual=True)

segment = synthetic.n_cube(1)

grid = quads.product(segment)
print(grid.topology.is_oriented)

grid = grid.boundary()

# fix the orientation of the resulting cubes
print(grid.topology.is_oriented)
# grid.topology = grid.topology.fix_orientation()
print(grid.topology.is_oriented)

# get bottom face chain
fp = grid.primal_position()[2]
fi = np.argsort(fp[:, 1])[:2]
fc = grid.topology.chain(2, 0)
fc[fi] = 1
c1 = grid.topology.matrix(1, 2) * fc

# add random rotation
grid.vertices = np.dot(grid.vertices, linalg.orthonormalize(np.random.randn(3, 3)))

# subdivide; propagate crease along the subdivision
grid = grid.subdivide(smooth=True, creases={1:c1})
c1 = grid.topology.transfer_matrices[1] * c1
grid = grid.subdivide(smooth=True, creases={1:c1})
c1 = grid.topology.transfer_matrices[1] * c1
grid = grid.subdivide(smooth=True, creases={1:c1})
c1 = grid.topology.transfer_matrices[1] * c1


grid.as_23().plot(plot_dual=False, plot_vertices=False)

grid = grid.as_23().to_simplicial()#.smooth().smooth()
grid.topology = grid.topology.fix_orientation()


grid.as_3().plot_3d(plot_dual=False, plot_vertices=False)

