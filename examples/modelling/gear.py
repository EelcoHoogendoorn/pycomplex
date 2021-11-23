"""
Hacky utilities for generating gear shapes and manipulating them


https://core.ac.uk/download/pdf/74220401.pdf
throwing involutes into the mix might be quite nice

another exposition
https://www.researchgate.net/publication/326346299_Generation_Method_for_a_Novel_Roots_Rotor_Profile_to_Improve_Performance_of_Dry_Multi-stage_Vacuum_Pumps

classic gerotor accounted for
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.826.5852&rep=rep1&type=pdf

good recent paper
https://www.mdpi.com/1996-1073/12/6/1126

other good recent paper
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.859.470&rep=rep1&type=pdf

how to do sdf simulation?
perform on cylindrical grid? more efficient use of voxels and makes rotation easy;
but translation operation would be a strange one; could use geometric_transform

have these constraints
    outer sdf is inner sdf 'unrolled' and negated
    tooth depth equals twice the origin offset

    soft constraints; want smooth curve

    what is the incentive to touch all along the curve?
    not sure there is one natively.
"""


import numpy as np
import matplotlib.pyplot as plt


def make_sdf(complex, spacing):
    import scipy.spatial
    tree = scipy.spatial.cKDTree(complex.vertices)
    box = complex.box * np.sqrt(2)
    size = box[1] - box[0]
    pixels = np.ceil(size / spacing).astype(np.int)
    grid = np.meshgrid(*[np.arange(d) for d in pixels], indexing='ij')
    coords = np.moveaxis(np.array(grid), 0, -1) * spacing + box[0]
    d, i = tree.query(coords.reshape(-1, 2))
    print()


# import cv2 as cv
def make_sdf(complex, spacing):
    box = complex.box * np.sqrt(2)
    size = box[1] - box[0]
    pixels = np.ceil(size / spacing).astype(np.int)
    grid = np.meshgrid(*[np.arange(d) for d in pixels], indexing='ij')
    coords = np.moveaxis(np.array(grid), 0, -1) * spacing + box[0]

    raw_dist = np.empty(pixels, dtype=np.float32)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            raw_dist[i, j] = cv.pointPolygonTest(complex.vertices.astype(np.float32), tuple(coords[i, j]), True)
    return raw_dist


def hull_sdf(sdf):
    """convex hull of sdf? smooth the sdf, while keeping it at minimum equal to original"""
    from scipy import ndimage
    q = sdf
    for i in range(10):
        q = ndimage.gaussian_filter(q, 10)
        q = np.maximum(sdf, q)
    return q


def from_sdf(sdf):
    import skimage.measure
    contours = skimage.measure.find_contours(sdf, level=0.0)
    return [ring(c) for c in contours][0]


class SDF(object):
    def __init__(self, image, spacing, origin=None, rotation=None):
        # maps pixels to global coordinates
        self.image = image
        self.spacing = spacing

        self.origin = origin or -np.array(self.image.shape) / 2 * spacing
        self.rotation = rotation or np.eye(2)

    def copy(self, **kwargs):
        return type(self)(**self.to_dict(), **kwargs)

    def to_dict(self):
        return {
            'image': self.image,
            'spacing':self.spacing,
            'rotation':self.rotation,
            'offset': self.offset
        }

    def translate(self, translation):
        return self.copy(origin = self.origin + translation)
    def rotate(self, rotation):
        """rotate around coordinate origin"""
        # new_coords = delta_rotation . old_coords
        #nr . (np - no) = dr . or . (op - oo)
        return self.copy(
            rotation=np.dot(rotation, self.rotation),
        )

    def pixels_to_coords(self, pixels):
        return np.dot(self.rotation, (pixels - self.offset) * self.spacing)
    def coords_to_pixels(self, coords):
        return np.dot(self.rotation.T, coords / self.spacing) + self.offset
    @property
    def transform(self):
        """multiplication maps pixels to world coordinates"""
        return

    def map(self, other):
        assert self.spacing == other.spacing
        dr = np.dot(self.rotation, other.rotation.T)
        offset = self.offset - other.offset.dot(dr)

    def marching_squares(self):
        import skimage.measure
        contours = skimage.measure.find_contours(self.image, level=0.0)
        return [ring(c) for c in contours][0]

    def sample(self, points):
        return

    def sample_grad(self, points):
        return

    def lasso(self, N, res=100):
        """fit single contour with iterative lasso"""
        a = np.linspace(0, np.pi*2, res)
        complex = ring(np.array([np.cos(a), np.sin(a)]).T)
        for i in range(10):
            sd = self.sample(complex.vertices)

        return complex


def sweep_sdf(sdf, radius, N, spacing, offset):
    """sweep sdf to compute complimentary profile

    Parameters
    ----------
    sdf
        inner rotor

    radius
    N: int
        number of teeth on inner rotor
    spacing: float
        size of each pixel


    """
    from scipy import ndimage
    inner = radius / (N+1) * N
    out = -np.inf
    c_in = c_out = np.array(sdf.shape) / 2
    c_in = c_out + [offset / spacing, 0]
    for a in np.linspace(0, 2*np.pi, 20):
        transform = rotation(a)
        offset = c_in-c_out.dot(transform)
        temp = ndimage.interpolation.affine_transform(
            sdf,
            transform.T,
            order=2,
            offset=offset,
            output_shape=sdf.shape,
            cval=sdf.min(),
            output=np.float32
        )
        out = np.maximum(out, temp)

    return out


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


def epitrochoid(a: float, q: int, d: float, N=2000):
    b = a / q
    k = d / b
    t = np.linspace(0, np.pi*2, N)
    x = b * ((q + 1) * np.cos(t) - k * np.cos((q+1)*t))
    y = b * ((q + 1) * np.sin(t) - k * np.sin((q+1)*t))

    # q = (a + b) / b
    x = (a + b) * np.cos(t) - d * np.cos(q*t)
    y = (a + b) * np.sin(t) - d * np.sin(q*t)
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
    t = np.linspace(0, np.pi*2, 500)
    x = b * ((q - 1) * np.cos(t) + k * np.cos((q-1)*t))
    y = b * ((q - 1) * np.sin(t) - k * np.sin((q-1)*t))

    q = (a - b) / b
    x = (a + b) * np.cos(t) + d * np.cos(q*t)
    y = (a + b) * np.sin(t) - d * np.sin(q*t)

    return np.array([x, y]).T


def sinusoid(T: int, s:float=0.01, r: float=1.0, N=2000):
    t = np.linspace(0, np.pi*2, N)
    s = s
    x = (r + s * np.sin(T * t)) * np.cos(t)
    y = (r + s * np.sin(T * t)) * np.sin(t)
    return np.array([x, y]).T


def buffer(complex, r):
    from shapely.geometry import Polygon
    poly = Polygon(complex.vertices).buffer(r)
    coords = np.array(poly.exterior.coords)
    # print(len(coords))
    return ring(coords[::-1])


def test_epi():
    fix, ax = plt.subplots()

    circle = ring(epitrochoid(2, 1e9, 0))
    # epi = ring(epitrochoid(2, 12, 2./12 * 0.9))
    ep2 = ring(epitrochoid(2, 90, 2./90 * 0.5))
    # hypo = ring(hypotrochoid(2, 11, 2./12 * 0.9))

    ep2 = ring(sinusoid(T=55))

    # complex = buffer((curve), 0.0)

    circle.plot(plot_vertices=False, ax=ax, color='gray')
    # epi.plot(plot_vertices=False, ax=ax)
    ep2.plot(plot_vertices=False, ax=ax, color='r')
    ep2.translate([2, 0]).plot(plot_vertices=False, ax=ax, color='b')
    plt.show()
    quit()

    sdf = make_sdf(complex, 0.02)
    hsdf = hull_sdf(sdf)
    plt.figure()
    plt.imshow(sdf, cmap='seismic')
    plt.colorbar()

    fix, ax = plt.subplots()
    contour = from_sdf(sdf)
    contour.plot(ax=ax, plot_vertices=False)
    contour = from_sdf(hsdf)
    contour.plot(ax=ax, plot_vertices=False)
    plt.show()

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
    """
    References
    ----------
    https://www.researchgate.net/publication/303053954_Specific_Sliding_of_Trochoidal_Gearing_Profile_in_the_Gerotor_Pumps
    """
    # FIXME: only the f=1 gears mesh properly currently. not sure yet how to solve. correction factor to base radius seems called for
    complex = ring(hypotrochoid(R, N, f))
    return buffer(complex, b)


def extrude_twist(profile, L, offset, N, sleeve=0, twist=lambda z: z):

    from pycomplex.synthetic import n_cube_grid
    # assuming a 1/1 twist rate, similar number of points in both directions of the surface makes sense
    # N = int(profile.topology.n_elements[0] * np.abs(factor))
    line = n_cube_grid((N,), centering=False)
    line = line.copy(vertices=line.vertices * L / N)

    cylinder = profile.product(line)
    from pycomplex.math import linalg
    cylinder = cylinder.as_23().subdivide_simplicial().as_3()

    # from examples.harmonics import get_harmonics_0
    x, y, z = cylinder.vertices.T
    zn = z / z.max() * np.pi / 2 + np.pi / 4
    S = np.power(np.cos(zn), 6)
    # plt.figure()
    # plt.scatter(z, S)
    # plt.show()
    r = z / L
    R = rotation(twist(r))
    # a = np.arctan2(y, x) + r
    # G = np.sin(a * 1.5) ** 6
    # plt.figure()
    # plt.scatter(a, G)
    # plt.show()
    # S = (1 + S * 0.1 * sleeve) * (1 + G*0.03*sleeve)
    S = 1
    T = R * S
    vertices = cylinder.vertices.copy()
    vertices[:, :2] = np.einsum('ijv, vi->vj', T, vertices[:, :2])


    cylinder = cylinder.copy(vertices=vertices)

    if offset:
        # FIXME: use buffer here? more accurate but for small steps should be fine
        offsets = linalg.normalized(cylinder.vertex_normals() * [1, 1, 0]) * offset
        print(cylinder.box)
        cylinder = cylinder.copy(vertices=cylinder.vertices + offsets)
        print(cylinder.box)

    # cylinder = cylinder.transform(linalg.orthonormalize(np.random.randn(3, 3)))
    # cylinder.plot_3d(plot_dual=False, plot_vertices=False)
    # plt.show()
    return cylinder


def quantify_error(res):
    a = gear(N, N, f, res=res).subdivide_cubical(smooth=False)
    b = gear(N, N, f, res=res*2)
    import scipy.spatial
    tree = scipy.spatial.cKDTree(a.vertices)
    d, i = tree.query(b.vertices, k=1)
    max_d = d.max()
    print(f'Discretization error estimate {max_d}')
# quantify_error(res=100)


def pcp():
    # classical progressive cavity consists of the N=1 case, offset by some radius
    rotor = hypo_gear(1, 1, 1)
    stator = hypo_gear(1, 2, 1)
    fig, ax = plt.subplots(1)
    rotor.plot(ax=ax, plot_vertices=False)
    stator.plot(ax=ax, plot_vertices=False)
    plt.show()


def generate_gcode(profile, height, layer_height, filament_factor=0.05, twist=1):
    """generate spiralized outer contour.
    ideally, profile is generated by a snake, or some other quality controlled method"""
    # how an angle around the profile relates to an angle step around the extruded part
    angle_ratio = 1 + layer_height / height * twist
    vertices = profile.vertices
    angles = np.arctan2(*vertices.T)
    angles = - angles + angles[0]

    g1 = 'G1 F{f:.3f} X{x:.3f} Y{y:.3f} Z{z:.3f} E{e:.5f}\n'
    gcode = []

    # generate points along contour
    # transform them to world space coordinates
    # e equal to edge length in world space (plus optional curvature correction)
    # f normalized for constant extrusion

    z = 0
    e = 0
    b = 0
    _a, _b, _x, _y, _z, _e = 0, 0, 0, 0, 0, 0
    while True:
        print(z)
        for i, (v, a) in enumerate(zip(vertices, angles)):
            da = (a - _a) % (2 * np.pi)
            db = da * angle_ratio
            b = b + db
            x, y = np.dot(rotation(b - a), v)
            dx = x - _x
            dy = y - _y
            db = b - _b
            dz = db / (2 * np.pi) * layer_height
            z = z + dz
            de = np.linalg.norm([dx, dy, dz]) * filament_factor
            e = e + de
            f = 600
            gcode.append(g1.format(f=f, x=x, y=y, z=z, e=e))
            _a, _b, _x, _y, _z, _e = a, b, x, y, z, e
        if z > height:
            break

    with open('../subdivision/foo.gcode', 'w') as fh:
        fh.writelines(gcode)




