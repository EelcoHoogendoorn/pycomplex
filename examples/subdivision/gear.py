import numpy as np
import matplotlib.pyplot as plt


def rotation(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


def trochoid(R, r, s, res):
    # r = R / N
    N = R / r
    a = np.linspace(0, 2*np.pi/N, int(r * res), endpoint=False)
    b = (R+r*s) / r * a * s

    q = [[np.cos(a), np.cos(b)], [np.sin(a), np.sin(b)]]
    return np.dot([(R+r*s), -r*s], q).T


def gear(R, N, f, res):
    r = R / N
    t = 2*np.pi/N
    p = trochoid(R, r * f, +1, res=res)
    n = trochoid(R, r * (1-f), -1, res=res)
    n = np.dot(n, rotation(t*f))
    u = np.concatenate([p, n], axis=0)
    c = np.concatenate([np.dot(u, rotation(t*i)) for i in range(N)], axis=0)
    from pycomplex.complex.cubical import ComplexCubical1Euclidian2
    v = np.arange(len(c))
    cubes = np.array([v, (v+1)%len(c)]).T
    return ComplexCubical1Euclidian2(vertices=c, cubes=cubes)


def extrude_twist(profile, L, offset, factor=1.0):

    from pycomplex.synthetic import n_cube_grid
    # assuming a 1/1 twist rate, similar number of points in both directions of the surface makes sense
    N = profile.topology.n_elements[0]
    line = n_cube_grid((N,), centering=False)
    line = line.copy(vertices=line.vertices * L / N)

    cylinder = profile.product(line)
    from pycomplex.math import linalg
    cylinder = cylinder.as_23().subdivide_simplicial().as_3()

    R = rotation(cylinder.vertices[:, -1] / L * 2 * np.pi * factor)
    vertices = -cylinder.vertices.copy()
    vertices[:, :2] = np.einsum('ijv, vi->vj', R, vertices[:, :2])
    cylinder = cylinder.copy(vertices=vertices)

    offsets = linalg.normalized(cylinder.vertex_normals() * [1, 1, 0]) * offset
    print(cylinder.box)
    cylinder = cylinder.copy(vertices=cylinder.vertices + offsets)
    print(cylinder.box)

    # cylinder = cylinder.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    # cylinder.plot_3d(plot_dual=False, plot_vertices=False)
    # plt.show()
    return cylinder


# fraction of epicyloid vs cycloid
f = 0.3
# note; classical progressive cavity consists of the 0-case, offset by some radius
# at constant radius, doubling N almost halves displaced area
N = 4
target_radius = 12
L = 50

offset = 0.3    # printer line thickness

rotor = gear(N, N, f, res=100)
stator = gear(N+1, N+1, f, res=100)


def quantify_error(res):
    a = gear(N, N, f, res=res).subdivide_cubical(smooth=False)
    b = gear(N, N, f, res=res*2)
    import scipy.spatial
    tree = scipy.spatial.cKDTree(a.vertices)
    d, i = tree.query(b.vertices, k=1)
    max_d = d.max()
    print(f'Discretization error estimate {max_d}')
quantify_error(res=100)


max_radius = np.linalg.norm(stator.vertices, axis=1).max()
scale = target_radius / max_radius
rotor = rotor.transform(np.eye(2) * scale)
stator = stator.transform(np.eye(2) * scale)

extrude_twist(rotor, L=L * 1.5, offset=-offset*0.5, factor=1.5).save_STL('gear_inner.stl')
extrude_twist(stator, L=L * 1.5, offset=+offset*0.5, factor=1.5 * N / (N+1)).save_STL('gear_outer.stl')
# quit()
print(max_radius)
print(rotor.volume())
print(stator.volume())
print('cc / rev')
print((stator.volume() - rotor.volume()) * L / 1000)


path = r'../output/gear4'
from examples.util import save_animation
frames = 60
for i in save_animation(path, frames=frames, overwrite=True):
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    if False:
        rotor.\
            transform(rotation(i / frames * 2 * np.pi / N * (N+1))).\
            translate([scale, 0]).\
            transform(rotation(-i / frames * 2 * np.pi / (N+1) * (N+1))).\
            plot(ax=ax, plot_vertices=False, color='b')
        stator.plot(ax=ax, plot_vertices=False, color='r')

    else:
        rotor.\
            transform(rotation(i / frames * 2 * np.pi / N)).\
            translate([scale, 0]).\
            plot(ax=ax, plot_vertices=False, color='b')
        stator.\
            transform(rotation(i / frames * 2 * np.pi / (N+1))).\
            plot(ax=ax, plot_vertices=False, color='r')

    ax = plt.gca()
    ax.set_xlim(*stator.box[:, 0]*1.1)
    ax.set_ylim(*stator.box[:, 1]*1.1)
    plt.axis('off')

    # plt.show()