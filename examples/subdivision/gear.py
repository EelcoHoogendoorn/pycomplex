import numpy as np
import matplotlib.pyplot as plt


def ring(c):
    from pycomplex.complex.cubical import ComplexCubical1Euclidian2
    v = np.arange(len(c))
    cubes = np.array([np.roll(v, 1), v]).T
    return ComplexCubical1Euclidian2(vertices=c, cubes=cubes)


def rotation(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


def trochoid_part(R, r, s, res):
    # r = R / N
    N = R / r
    a = np.linspace(0, 2*np.pi/N, int(r * res), endpoint=False)
    b = (R+r*s) / r * a * s

    q = [[np.cos(a), np.cos(b)], [np.sin(a), np.sin(b)]]
    return np.dot([(R+r*s), -r*s], q).T


def epitrochoid(a: float, q: int, d: float):
    b = a / q
    k = d / b
    t = np.linspace(0, np.pi*2, 1000)
    x = b * ((q + 1) * np.cos(t) - k * np.cos((q+1)*t))
    y = b * ((q + 1) * np.sin(t) - k * np.sin((q+1)*t))
    return np.array([x, y]).T


def hypotrochoid(a: float, q: int, d: float):
    """
    where a is the radius of the base circle,
    b = a / q that of the rolling circle,
    and d = k b the distance between the point and the centre of the moving circle
    """
    # a = a - 0.6
    b = a / q
    print(b)
    k = d / b
    # k = 0.6
    t = np.linspace(0, np.pi*2, 1000)
    x = b * ((q - 1) * np.cos(t) + k * np.cos((q-1)*t))
    y = b * ((q - 1) * np.sin(t) - k * np.sin((q-1)*t))
    return np.array([x, y]).T


def buffer(complex, r):
    from shapely.geometry import Polygon
    poly = Polygon(complex.vertices).buffer(r)
    coords = np.array(poly.exterior.coords)
    # print(len(coords))
    return ring(coords[::-1])


def test_epi():
    curve = hypotrochoid(2, 6, 2/7)
    complex = buffer(ring(curve), 0.3)

    print(len(complex.vertices))
    plt.plot(*complex.vertices.T)
    plt.axis('equal')
    plt.show()
    quit()
# test_epi()


def gear(R, N, f, res):
    """compound gear of alternating epi and hypo curves

    Parameters
    ----------
    R: float
        radius of fixed circle
    N: int
        number of teeth
    f: float
        fraction of epi-vs-hypo
    res: int
        number of vertices per curve-section
    """
    r = R / N
    t = 2*np.pi/N
    p = trochoid_part(R, r * f, +1, res=res)
    n = trochoid_part(R, r * (1 - f), -1, res=res)
    n = np.dot(n, rotation(t*f))
    u = np.concatenate([p, n], axis=0)
    c = np.concatenate([np.dot(u, rotation(t*i)) for i in range(N)], axis=0)
    return ring(c)


def hypo_gear(R, N, b, f=1):
    # FIXME: only the f=1 gears mesh properly currently. not sure yet how to solve. correction factor to base radius seems called for
    complex = ring(hypotrochoid(R, N, f))
    return buffer(complex, b)


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

    # FIXME: use buffer here?
    offsets = linalg.normalized(cylinder.vertex_normals() * [1, 1, 0]) * offset
    print(cylinder.box)
    cylinder = cylinder.copy(vertices=cylinder.vertices + offsets)
    print(cylinder.box)

    # cylinder = cylinder.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    # cylinder.plot_3d(plot_dual=False, plot_vertices=False)
    # plt.show()
    return cylinder


def pcp():
    # classical progressive cavity consists of the N=1 case, offset by some radius
    rotor = hypo_gear(1, 1, 1)
    stator = hypo_gear(1, 2, 1)
    fig, ax = plt.subplots(1)
    rotor.plot(ax=ax, plot_vertices=False)
    stator.plot(ax=ax, plot_vertices=False)
    plt.show()


# fraction of epicyloid vs cycloid
f = 0.26
N = 5
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

rotor= buffer(rotor, 1.6)
stator = buffer(stator, 1.6)

rotor = buffer(ring(epitrochoid(N, N, 0.999)), -1.1)
stator = buffer(ring(epitrochoid(N + 1, N + 1, 0.999)), -1.1)

# rotor = hypo_gear(N, N, 0.9, f=0.9)
# stator = hypo_gear(N+1, N+1, 0.9, f=0.9 * (N / (N + 1)))
# f = 0.7
# scale = N / (N - (1-f))
# rotor = hypo_gear(N*scale, N, 1.2*0, f=f)#.transform(np.eye(2) * scale)
# stator = hypo_gear((N+1), N+1, 1.2*0, f=f)

if True:
    fig, ax = plt.subplots(1)
    rotor.plot(ax=ax, plot_vertices=False)
    stator.plot(ax=ax, plot_vertices=False)
    plt.show()


max_radius = np.linalg.norm(stator.vertices, axis=1).max()
scale = target_radius / max_radius
rotor = rotor.transform(np.eye(2) * scale)
stator = stator.transform(np.eye(2) * scale)

translation = np.linalg.norm(stator.vertices, axis=1).max() - np.linalg.norm(rotor.vertices, axis=1).max()

extrude_twist(rotor, L=L * 1.5, offset=-offset*0.5, factor=1.5).save_STL('gear_inner.stl')
extrude_twist(stator, L=L * 1.5, offset=+offset*0.5, factor=1.5 * N / (N+1)).save_STL('gear_outer.stl')

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
            translate([translation, 0]).\
            plot(ax=ax, plot_vertices=False, color='b')
        stator.\
            transform(rotation(i / frames * 2 * np.pi / (N+1))).\
            plot(ax=ax, plot_vertices=False, color='r')

    ax = plt.gca()
    ax.set_xlim(*stator.box[:, 0]*1.1)
    ax.set_ylim(*stator.box[:, 1]*1.1)
    plt.axis('off')

    # plt.show()