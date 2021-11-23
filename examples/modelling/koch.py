"""
Script to generate a 3d printable lampshade based on a twisted extrusion of a koch snowflake
"""

from pycomplex.synthetic import n_simplex, n_cube_grid
import matplotlib.pyplot as plt
import numpy as np
from pycomplex.complex.simplicial.euclidian import ComplexSimplicialEuclidian

# cube = n_cube_grid((1, 1, 1))
# cube.boundary.as_23().subdivide_simplicial().as_3().save_STL('cube.stl')
# quit()
def subdivide_koch(coarse, f):
    assert coarse.topology.is_oriented
    assert coarse.topology.is_manifold
    # assert coarse.topology.is_closed

    cv = coarse.vertices
    cev = coarse.topology.incidence[1, 0]
    q = cv[cev]
    cv = np.einsum('vpc,p->vc', q, [6/6, 0/6])
    lv = np.einsum('vpc,p->vc', q, [4/6, 2/6])
    mv = np.einsum('vpc,p->vc', q, [3/6, 3/6])
    rv = np.einsum('vpc,p->vc', q, [2/6, 4/6])
    qv = np.einsum('vpc,p->vc', q, [0/6, 6/6])

    # displace middle vertex
    r = [[0, 1], [-1, 0]]
    n = (q[:, 1] - q[:, 0]).dot(r)
    mv = mv + n / 3 * np.sqrt(3) / 2 * f

    N = len(cev)
    fev = np.zeros((N, 4, 2), dtype=cev.dtype)

    fev[:, 0, 0] = cev[:, 0] + [0*N]
    fev[:, 0, 1] = cev[:, 0] + [1*N]
    fev[:, 1, 0] = cev[:, 0] + [1*N]
    fev[:, 1, 1] = cev[:, 0] + [2*N]
    fev[:, 2, 0] = cev[:, 0] + [2*N]
    fev[:, 2, 1] = cev[:, 0] + [3*N]
    fev[:, 3, 0] = cev[:, 0] + [3*N]
    fev[:, 3, 1] = cev[:, 0] + [4*N]

    fv = np.array([cv, lv, mv, rv, qv])

    return ComplexSimplicialEuclidian(fv.reshape(-1, 2), fev.reshape(-1, 2))


import numpy_indexed as npi

def merge(complex):
    """stitch contour closed; ugly but too stupid or lazy to get it right in one go"""
    vertices, idx, inverse = npi.unique(complex.vertices, return_index=True, return_inverse=True)

    return idx, inverse


def do_merge(complex, idx, inverse):
    vertices = complex.vertices[idx]
    e = complex.topology.elements[-1]
    elements = npi.remap(e.flatten(), np.arange(len(inverse), dtype=e.dtype), inverse.astype(e.dtype)).reshape(e.shape)
    return type(complex)(vertices, elements)


def recurse(n, e):
    koch = n_simplex(2).boundary
    for i in range(max(n, int(np.ceil(e)))):
        koch = subdivide_koch(koch, np.clip(e-i, 0, 1))
    return koch.as_cube().as_12()

def rotation(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


def twist(complex, factor=0.2):
    # from examples.harmonics import get_harmonics_0
    x, y, z = complex.vertices.T
    L = z.max()
    zn = z / L * np.pi / 2 + np.pi / 4
    S = np.power(np.cos(zn), 6)
    # plt.figure()
    # plt.scatter(z, S)
    # plt.show()
    r = z / L * 2 * np.pi * factor
    R = rotation(r)
    a = np.arctan2(y, x) + r
    G = np.sin(a * 1.5) ** 6
    # plt.figure()
    # plt.scatter(a, G)
    # plt.show()
    T = R
    vertices = -complex.vertices.copy()
    vertices[:, :2] = np.einsum('ijv, vi->vj', T, vertices[:, :2])
    return complex.copy(vertices=vertices)



divisions = np.arange(1, 6)
length = 1 / 3**divisions
print(length)
kochs = [recurse(divisions.max(), i) for i in divisions]
merge_params = merge(kochs[0])
kochs = [do_merge(k, *merge_params) for k in kochs]
vertices = np.array([k.vertices for k in kochs])
from scipy import interpolate
interp = interpolate.interp1d(length, vertices, axis=0)

koch = kochs[-1]

z = np.linspace(0, 250, 60)
r = 10 + np.sqrt(z + 5) * 7

print(r)
r_interp = interpolate.interp1d(z, r)
l = 1 / r * 1.7
print(l)
# quit()

from pycomplex import synthetic
line = synthetic.n_cube_grid((len(z)-1,), centering=False)
line = line.scale(z.max() / len(z))

lamp = koch.product(line).as_23()
# print(line.vertices[:, 0].max())
# quit()

V = lamp.vertices * 1

V[:, :2] = interp(l).reshape(-1, 2)

lamp = lamp.copy(vertices=V)


# time to go to triangular domain
lamp = lamp.subdivide_simplicial()
rscale = np.repeat(r_interp(lamp.vertices[:, 2])[:, None], 3, axis=1)
rscale[:, 2] = 1
lamp = lamp.copy(vertices=lamp.vertices * rscale).as_3()

lamp = twist(lamp)

lamp.save_STL('lamp_alt.stl')

if False:
    fix, ax = plt.subplots(1, 1)
    # lamp = lamp.transform(linalg.orthonormalize(np.random.randn(3, 3)))

    lamp.plot(ax=ax, plot_dual=False, plot_vertices=False)
plt.show()
# root.plot(ax=ax, plot_dual=False, primal_color='r')

# koch = koch.copy(vertices=interp(0.2))
# koch.plot(ax=ax, plot_dual=False, plot_vertices=False)

