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


def trochoid(R, r, s):
    # r = R / N
    N = R / r
    a = np.linspace(0, 2*np.pi/N, int(r * 100), endpoint=False)
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
    b = a / q
    k = d / b
    t = np.linspace(0, np.pi*2, 1000)
    x = b * ((q - 1) * np.cos(t) + k * np.cos((q-1)*t))
    y = b * ((q - 1) * np.sin(t) - k * np.sin((q-1)*t))
    return np.array([x, y]).T


def buffer(complex, r):
    from shapely.geometry import Polygon
    poly = Polygon(complex.vertices).buffer(r)
    coords = np.array(poly.exterior.coords)
    # print(len(coords))
    return ring(coords)


def test_epi():
    curve = hypotrochoid(2, 6, 2/7)
    complex = buffer(ring(curve), 0.3)

    print(len(complex.vertices))
    plt.plot(*complex.vertices.T)
    plt.axis('equal')
    plt.show()
    quit()
# test_epi()


def gear(R, N, f):
    r = R / N
    t = 2*np.pi/N
    p = trochoid(R, r * f, +1)
    n = trochoid(R, r * (1-f), -1)
    n = np.dot(n, rotation(t*f))
    u = np.concatenate([p, n], axis=0)
    c = np.concatenate([np.dot(u, rotation(t*i)) for i in range(N)], axis=0)
    return ring(c)


def hypo_gear(R, N, b, f=1):
    complex = ring(hypotrochoid(R * f, N, R / N * f))
    return buffer(complex, b)


def extrude_twist(profile, L):
    from pycomplex.synthetic import n_cube_grid
    line = n_cube_grid((L,), centering=False).subdivide_cubical().subdivide_cubical()

    cylinder = profile.product(line)
    from pycomplex.math import linalg
    cylinder = cylinder.as_23().subdivide_simplicial().as_3()

    R = rotation(cylinder.vertices[:, -1] / L * 2 * np.pi)
    vertices = -cylinder.vertices.copy()
    vertices[:, :2] = np.einsum('ijv, vi->vj', R, vertices[:, :2])
    cylinder = cylinder.copy(vertices=vertices)
    cylinder.save_STL('gear.stl')

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
f = 0.66
N = 4
target_radius = 12
L = 50

rotor = gear(N, N, f)
stator = gear(N+1, N+1, f)

# rotor = hypo_gear(N, N, 0.9, f=0.9)
# stator = hypo_gear(N+1, N+1, 0.9, f=0.9 * (N / (N + 1)))
rotor = hypo_gear(N, N, 1.2, f=0.7)
stator = hypo_gear(N+1, N+1, 1.2, f=0.7)

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

extrude_twist(rotor, L=L)

print(max_radius)
print(rotor.volume())
print(stator.volume())
print('cc / rev')
print((stator.volume() - rotor.volume()) * L / 1000)


path = r'../output/gear2'
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