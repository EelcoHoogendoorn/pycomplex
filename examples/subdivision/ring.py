"""
This example creates a complex suitable for modelling decorative rings.
It gives flexible control over the exact cross section used for the ring,
and constructs stretch-free uv coordinates, so that a displacement map can be applied,
to engrave a relief onto the ring.
"""

import numpy as np
import scipy.ndimage
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from pycomplex.math import linalg
from pycomplex.complex.cubical import ComplexCubical1Euclidian2, ComplexCubical2Euclidian3
from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
from pycomplex.synthetic import n_cube, n_cube_grid


def add_coordinates(complex: ComplexCubical1Euclidian2):
    """Assign continuous coordinates to a closed contour, encoded as complex numbers"""
    topology = complex.topology[0].T
    edge_length = np.linalg.norm(topology * complex.vertices, axis=1)
    cut_idx = np.argmin(complex.dual_position[0][:, 1])
    # rather than deleting last edge, better to split connections of target vertex
    cols = np.delete(np.arange(len(complex.vertices)), cut_idx)
    x = scipy.sparse.linalg.lsqr(topology[cols], edge_length[cols])[0]
    # scale correctly
    x = x - x.min()
    x = x / edge_length.sum()
    x = x * 2 * np.pi - np.pi
    complex.coords = np.exp(1j * x)[:, None]


def section(n: int) -> ComplexCubical1Euclidian2:
    """Generate the cross sectional profile of the ring"""
    ring = n_cube_grid((1, 2)).boundary
    ring = ring.transform(np.diag([2, 5]))
    # ring = ring.subdivide_cubical(smooth=False)
    for i in range(n):
        ring = ring.subdivide_cubical(smooth=True)
    ring = ring.as_12()

    add_coordinates(ring)
    return ring


def circle(n: int, radius: float) -> ComplexCubical1Euclidian2:
    """Construct a plain circle"""
    complex = n_cube(2, centering=True).boundary
    for i in range(n):
        complex = complex.subdivide_cubical(smooth=True)
        complex = complex.copy(vertices=linalg.normalized(complex.vertices))
    complex = complex.as_12().copy(vertices=complex.vertices * radius)

    add_coordinates(complex)
    return complex


def sweep(section: ComplexCubical1Euclidian2, path: ComplexCubical1Euclidian2) -> ComplexCubical2Euclidian3:
    """Sweep the section around the path to form a torus"""
    # FIXME: demote to sweep instead? alternatively, get angles from path tangent
    # FIXME: do product in terms of uv coords; then construct spatial verts from that in subsequent transform?

    # rotation around the z axis
    angle = path.coords[:, 0]
    c, s, z = angle.real, angle.imag, np.zeros_like(angle.real)
    rotations = np.array([[c, s, z], [-s, c, z], [z, z, np.ones_like(angle.real)]])

    # centers, in the xy plane
    centers = np.concatenate([path.vertices, path.vertices[:, 0:1] * 0], axis=1)
    # all points of the section, in the yz plane
    points = np.concatenate([section.vertices[:, 0:1] * 0, section.vertices[:, 0:2]], axis=1)
    vertices = np.einsum('ijr, pj -> rpi', rotations, points) + centers[:, None, :]

    torus = ComplexCubical2Euclidian3(
        topology=section.topology.product(path.topology),
        vertices=vertices.reshape(-1, 3)
    )
    # broadcast the coords along the same convention as used in topological product
    j, i = np.indices((len(path.vertices), len(section.vertices)))
    coords = np.concatenate(
        [
            section.coords[i.flatten()],
            path.coords[j.flatten()]
        ], axis=1
    )
    torus.coords = coords
    return torus


def displacement_map(torus: ComplexTriangularEuclidian3, texture: np.ndarray, depth: float):
    """Apply a displacement map to the torus"""
    texture = np.fliplr(texture)
    coords = (np.angle(torus.coords) / (np.pi * 2) + 0.5) * texture.shape
    height = scipy.ndimage.map_coordinates(
        texture, coords.T,
        mode='constant', order=1, prefilter=False
    )
    return torus.copy(vertices=torus.vertices - linalg.normalized(torus.vertex_normals()) * height[:, None] * depth)


def load_texture():
    """Load a grayscale relief map"""
    texture = plt.imread('texture2.png').sum(axis=-1)
    texture = texture / texture.max()
    texture = scipy.ndimage.gaussian_filter(texture, 1)
    return texture


if False:
    # visualise the section
    complex = section(4)
    complex.plot()
    plt.scatter(*complex.vertices.T, c=complex.coords[:, 0])
    plt.show()
    quit()


texture = load_texture()

if False:
    # visualise the texture
    plt.imshow(texture, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.show()

if True:
    complex = sweep(section(6), circle(8))
    print('swept the ring')

    # map the product 2-cubes to a simplicial representation
    tricomplex = complex.subdivide_simplicial().as_3()#.smooth()
    # map the uv coords in complex form onto the derived triangular complex; complex representation is key here
    tricomplex.coords = tricomplex.topology.transfer_operators[0] * complex.coords
    print('to simplicial')

    tricomplex = displacement_map(tricomplex, texture, depth=0.3)
    print('added displacement')

    tricomplex.save_STL('ring.stl')
    print('saved')

    if False:
        tricomplex = tricomplex.transform(linalg.orthonormalize(np.random.normal(size=(3, 3))))
        tricomplex.plot(plot_dual=False)
        plt.show()
