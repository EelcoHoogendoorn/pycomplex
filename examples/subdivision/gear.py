"""
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


def check_compound_compat(R, P, S, n):
    # this formula just follows from geometry of neutral lines; two planets plus a sun must fit in ring
    assert S + (P * 2) == R
    # this ensures possibility of equal spacing of the planets
    assert ((S + R) % n) == 0


def solve_compound(G1, G2):
    # numerical solution for compound gear ratios
    print(G1, G2)
    M = np.zeros((8,8))
    def planet_relations(S, P, R):
        return [
            [S, +P, -S - P, 0],
            [0, -P, -R + P, R],
        ]
    t = np.zeros(8)
    M[0:2, 0:4] = planet_relations(*G1)
    M[2:4, 4:8] = planet_relations(*G2)
    M[4, 1::4] = [1, -1]    # tie planet variables
    M[5, 2::4] = [1, -1]    # tie carrier variables

    # got some interesting bcs here. could elect to use suns as outputs too? seems like theyd always spin fast

    results = []
    # BC's
    M[6:, :] = 0
    M[6, [0, 3]] = [1, 1]; t[6] = 1  # spin S1 relative to R1
    # M[6, 0] = 1; t[6] = 1   # spin S1
    M[7, 3] = 1             # pin R1
    r = np.linalg.solve(M, t)
    results.append(1 / (r[3] - r[7]))

    M[6:, :] = 0
    M[6, [4, 7]] = [1, 1]; t[6] = 1  # spin S2 relative to R2
    # M[6, 0+4] = 1; t[6] = 1   # spin S2
    M[7, 3+4] = 1             # pin R2
    r = np.linalg.solve(M, t)
    results.append(1 / (r[3] - r[7]))

    M[6:, :] = 0
    M[6, [0, 7]] = [1, 1]; t[6] = 1  # spin S1 relative to R2
    # M[6, 0] = 1; t[6] = 1     # spin S1
    M[7, 3+4] = 1             # pin R2
    r = np.linalg.solve(M, t)
    results.append(1 / (r[3] - r[7]))

    M[6:, :] = 0
    M[6, [4, 3]] = [1, 1]; t[6] = 1  # spin S2 relative to R1
    # M[6, 0+4] = 1; t[6] = 1     # spin S2
    M[7, 3] = 1                   # pin R1
    r = np.linalg.solve(M, t)
    results.append(1 / (r[3] - r[7]))

    # this is a cute idea; but where do the stator wires go?
    # M[6:, :] = 0
    # M[6, [0, 4]] = [1, 1]; t[6] = 1  # spin S1 relative to S2
    # M[7, 3] = 1                     # pin R1
    # r = np.linalg.solve(M, t)
    # results.append(1 / (r[3] - r[7]))
    print(results)
    return results

from functools import lru_cache

@lru_cache()
def solve_compound_symbolic():

    from sympy import symbols, linsolve
    s1, r1, s2, r2, p, c, d = symbols('s1 r1 s2 r2 p c d')
    S1, P1, R1, S2, P2, R2 = symbols('S1 P1 R1 S2 P2 R2')
    eqs = [
        S1 * s1 + P1 * p - (S1 + P1) * c,
        R1 * r1 - P1 * p - (R1 - P1) * c,
        S2 * s2 + P2 * p - (S2 + P2) * c,
        R2 * r2 - P2 * p - (R2 - P2) * c,
    ]
    # first item is the 'ground'; what the motor stator is connected to, mechanically and electrically
    # second item is what the motor rotor is connected to
    # third item is what we consider to be the mechanical output, relative to the ground
    configs = [
        # [r, s, r]
        # primary config; drive s relative to r, and rings as mechanical outputs
        [r1, s1, r2],
        [r1, s2, r2],
        [r2, s1, r1],
        [r2, s2, r1],

        # [s, r, s]
        # use both suns as mechanical connections
        # these seem like viable configs;
        # generally lower ratio than primary ones; 3-20x lower, but certainly nontrivial ones available
        # however, mechanically less favorable; another seal to deal with?
        # or just add housing attached to fixed shaft, with single seal on other shaft?
        # nice from sealing complexity pov, but now housing is additional component.
        [s1, r1, s2],
        [s2, r1, s1],
        [s1, r2, s2],
        [s2, r2, s1],

        # [r, s, s]
        # using sun as output shaft; getting ratios rather close to one
        [r1, s1, s2],
        [r1, s2, s1],
        [r2, s1, s2],
        [r2, s2, s1],
        # [s, r, r]
        # similar story
        [s1, r2, r1],
        [s1, r1, r2],
        [s2, r2, r1],
        [s2, r1, r2],
        #
        # # [s, s, r]
        # # these configs perform upgearing; similar to r-r-s config; r backdriving of the primary
        # [s1, s2, r1],
        # [s1, s2, r2],
        # [s2, s1, r1],
        # [s2, s1, r2],
        # # [r, r, s]
        # # backdriving of the primary config; upgearing like above
        # [r1, r2, s1],
        # [r1, r2, s2],
        # [r2, r1, s1],
        # [r2, r1, s2],
    ]
    # generate bcs
    conditions = {
        str(c): [
            # fix ground
            c[0] - 0,
            # apply one unit of rotation across motor rotor/stator
            c[1] - c[0] - 1,
            # define mechanical output
            c[2] - c[0] - d
        ]
        for c in configs
    }
    r = {
        k: linsolve(eqs + bcs, (s1, p, c, r1, s2, r2, d))
        for k, bcs in conditions.items()
    }
    # d-term is of interest
    r = {k: str(v.args[0][-1]) for k, v in r.items()}

    print(r)
    return r

def eval_compound_symbolic(G1, G2):
    """Evaluate a computed symbolic ratio relationship"""
    S1, P1, R1 = G1
    S2, P2, R2 = G2

    eqs = solve_compound_symbolic()
    # R = {k: eval(eq) for k, eq in eqs.items()}
    R = {}
    # print(G1, G2)
    for k, eq in eqs.items():
        R[k] = eval(eq)
        # print(k, 1 / R[k])
    return R


def brute_compound_gear(target_ratio=50):
    S, P = np.meshgrid(np.arange(5, 20), np.arange(5, 20))
    S, P = S.flatten(), P.flatten()
    R = S + P * 2
    # FIXME: add ratio calcs for all drive combinations in output
    #  can we hold ring1 fixed wile driving sun1?
    #  or can we drive s1 relative to s2?
    def calc_ratio(s, p, r):
        """

        S * Ts + P * Tp - (S + P) * Tc
        R * Tr - P * Tp - (R - P) * Tc

        S1 * Ts1 + P1 * Tp1 - (S1 + P1) * Tc1
        R1 * Tr1 - P1 * Tp1 - (R1 - P1) * Tc1

        S2 * Ts2 + P2 * Tp2 - (S2 + P2) * Tc2
        R2 * Tr2 - P2 * Tp2 - (R2 - P2) * Tc2

        Tc1 = Tc2
        Ts1 = 1
        Tr1 = 0

        now 7 relations, for 6 variables
        actually a linear system
        [S, P, C, R] variable ordering

        [0] = [S1, +P1, -S1-P1, 0,  0,  0,   0, 0]
        [0] = [0 , -P1, -R1+P1, R1, 0,  0,   0, 0]
        [0] = [0,  0,   0,      0,  S2, +P2, -S2-P2, 0]
        [0] = [0,  0,   0,      0,  0,  -P2, -R2+P2, R2]
        [0] = [0,  1,   0,      0,  0,  -1,   0, 0]
        [0] = [0,  0,   1,      0,  0,  0,   -1, 0]
        [1] = [1,  0,   0,      0,  0,  0,   0, 0]
        [0] = [0,  0,   0,      1,  0,  0,   0, 0]
        """

        # calc ratios with r1 locked and s1 driven
        s1, p1, r1 = s[:, None], p[:, None], r[:, None]
        s2, p2, r2 = s[None, :], p[None, :], r[None, :]
        q = eval_compound_symbolic((s1, p1, r1), (s2, p2, r2))['[r1, s1, r2]']
        # q = ((r2 - ((r1 / p1) * p2)) / r2) * (s1 / (r1 + s1))
        q = 1 / q
        return np.where(q > 1e9, 0, q)

    ratio = {}
    print()
    for n in range(3, 25):

        # 0.25 is minimum fraction of outward pointing gear we consider
        AIR = (S + R) / 2 * np.pi - n * P * (1 + 0.25 * 2 / P)
        AIR = AIR / P  # overlap as fraction of planet diameter
        # FIXME: makes no sense; air is high like ==1 for high n but good fit
        # but low for low n but still overlap

        f = np.logical_and(
            # need uniform spacing
            (S + R) % n == 0,
            # dont want overlap between gears
            AIR > 0
        )
        s = S[f]
        p = P[f]
        r = R[f]
        air = AIR[f]
        scale = s + p
        module = 1 / scale

        torque = np.minimum(module[:, None], module[None, :]) * n
        ratios = np.abs(calc_ratio(s, p, r))

        # score = np.abs(ratios - target_ratio) - torque * 10 # FIXME: add torque rating into score
        # score = -ratios * torque # power density score
        locked = (ratios == 0)
        ratio_error = np.abs(ratios - target_ratio)
        p_diff = np.abs(p[:, None] - p[None, :])
        air_error = np.maximum(np.abs(air[None, :] - 0.15), np.abs(air[:, None] - 0.15))
        score = locked * 1e9 - torque * 0 + ratio_error * 1e-1 + p_diff * 1e1 + air_error * 1e1
        i1, i2 = np.indices(score.shape)
        i = np.argsort(score.flatten())[:10]

        i1 = i1.flatten()[i]
        g1 = s[i1], p[i1], r[i1]
        i2 = i2.flatten()[i]
        g2 = s[i2], p[i2], r[i2]

        # do we want mod-ratio? or is smallest tooth in absolute sense most interesting?
        mod_ratio = ((g1[0] + g1[1]) / (g2[0] + g2[1]))

        print(n)
        # tooth size * n should be a measure of torque rating
        for g in zip(zip(*g1), zip(*g2), ratios.flatten()[i], np.minimum(air[i1], air[i2]), np.minimum(module[i1], module[i2]) * n):
            print(g)
        print()

        # ratio[n] = [s, p, r, ratios]
        # print(n, ratio[n][-1].max())

    print()


def generate_gearset(s, p, r, b=0.5, cycloid=True, N=2000):
    """
    as generated, this triplet of gears are constructed to mesh,
    after translation of the planet by [1, 0]
    """
    # normalize sizes so that planets lie on unit radius
    scale = s + p
    # just about 0.6mm bottom to top depth of tooth if scaling box to 10cm
    depth = 0.01

    if cycloid:
        sg = gear(s / scale, s, b, N)
    else:
        sg = ring(sinusoid(T=s, r=s / scale, s=depth, N=n))
    # if planet is uneven, rotate sun by half a tooth, to ensure meshing
    sg = sg.transform(rotation(2 * np.pi / s * ((p % 2) / 2)))
    if cycloid:
        pg = gear(p / scale, p, 1 - b, N)
    else:
        pg = ring(sinusoid(T=p, r=p / scale, s=depth, N=N))
    if cycloid:
        rg = gear(r / scale, r, 1 - b, N)
    else:
        rg = ring(sinusoid(T=r, r=r / scale, s=depth, N=N))

    return sg, pg, rg


def rotate_gearset(s, p, r, sg, pg, rg, phase):
    """animate a set of gears by a given phase angle, where one phase is a whole tooth"""
    sg = sg.transform(rotation(2 * np.pi / s * -phase))
    pg = pg.transform(rotation(2 * np.pi / p * phase))
    rg = rg.transform(rotation(2 * np.pi / r * phase))
    return sg, pg, rg


def generate_planetary(s, p, r, n, sg, pg, rg, phase=None):
    # expand set of gears into properly aligned ring config, with n meshing planets
    check_compound_compat(r, p, s, n)

    if phase is None:
        phase = float(np.random.rand(1)) # phase offset as a fractional tooth
    sg, pg, rg = rotate_gearset(s, p, r, sg, pg, rg, phase)

    # expand single planet into full ring of n
    pgs = []
    # get min and max radius of gear; plot/return those to optimize gear fitting logic?
    for i in range(n):
        sa = 2 * np.pi * i / n
        pa = -sa / p * r
        pgs.append(pg.transform(rotation(pa)).translate([1, 0]).transform(rotation(sa)))

    return sg, pgs, rg


def plot_planetary(s, p, r, n, b=0.5, phase=None, ax=None, col='b'):
    """plot the specified planetary gearset"""
    sg, pg, rg = generate_gearset(s, p, r, cycloid=True, b=b)
    sg, pgs, rg = generate_planetary(s, p, r, n, sg, pg, rg, phase)

    sg.plot(plot_vertices=False, ax=ax, color=col)

    for pg in pgs:
        pg.plot(plot_vertices=False, ax=ax, color=col)

    rg.plot(plot_vertices=False, ax=ax, color=col)


def animate_planetary(s, p, r, n, b=0.33, col='b'):

    from examples.util import save_animation
    N = 40
    path = f'../output/gear_{s}_{p}_{r}_{n}'
    for i in save_animation(path, frames=N, overwrite=True):
        fig, ax = plt.subplots(1)
        phase = i / N
        sg, pgs, rg = generate_planetary(s, p, r, n, b, phase, cycloid=True)

        sg.plot(plot_vertices=False, ax=ax, color=col)
        for pg in pgs:
            pg.plot(plot_vertices=False, ax=ax, color=col)

        rg.plot(plot_vertices=False, ax=ax, color=col)
        plt.axis('off')
        ax.set_xlim(-1.6, +1.6)
        ax.set_ylim(-1.6, +1.6)


def compound(g1, g2, n=None, b1=0.5, b2=0.5):
    """add extrusions using correct helical logic
    how to do transition regions? modulate amplitude of teeth?
    for sls print really doesnt matter
    """
    s1, p1, r1 = g1
    s2, p2, r2 = g2

    # ratios = solve_compound(g1, g2)
    keys = [
        '[r1, s1, r2]',
        '[r1, s2, r2]',
        '[r2, s1, r1]',
        '[r2, s2, r1]',
    ]
    ratios_ = eval_compound_symbolic(g1, g2)
    ratios = [1. / ratios_[k] for k in keys]

    # ratio = (((r2 - ((r1 / p1) * p2)) / r2) * (s1 / (r1 + s1)))
    # print(ratio)
    # mod_ratio = ((s1 + p1) / (s2 + p2))
    # print(mod_ratio)

    fig, ax = plt.subplots(1, 1)
    ax.set_title('{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*ratios))
    plot_planetary(s1, p1, r1, n, b1, ax=ax, col='r', phase=0.25)
    plot_planetary(s2, p2, r2, n, b2, ax=ax, col='b', phase=0)
    plt.show()


def calc_slippage(N1, N2):
    """ calculate an estimate of slippage by mean of arc length diff.
    ideally do so in incremental manner, as the ideal contact point moves along

    for cycloidal section at least we know two ideal contact points
    """
    # N = 60
    r = 0.1
    # R = 1.0
    R1 = r * N1
    R2 = r * N2
    # r = R / N
    res = 1000

    f = 0.50
    p = trochoid_part(R1, r * f, +1, res=res)
    n = trochoid_part(R2, r * (1 - f), -1, res=res)
    if False:
        plt.plot(*p.T)
        plt.plot(*n.T)
        plt.axis('equal')
        plt.show()
    def length(c):
        d = np.diff(c, axis=0)
        a = np.linalg.norm(d, axis=1)
        return a.sum()
    return (length(p) / length(n))


def calc_slippage_sinusoid(N1, N2):
    # this is about a 0.6mm tooth
    p = sinusoid(N1, r=N1 / 10, N=N1*1000, s=0.03)
    n = sinusoid(N2, r=N2 / 10, N=N2*1000, s=0.03)

    p = p[750:1250]
    n = n[250: 750]

    if True:
        plt.plot(*p.T)
        plt.plot(*n.T)
        plt.axis('equal')
        plt.show()

    def length(c):
        d = np.diff(c, axis=0)
        a = np.linalg.norm(d, axis=1)
        return a.sum()

    return (length(p) / length(n))


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


def extrude_planetary(s, p, r, n, b=0.5):
    """extrude a full set of gears based on the given specs"""
    def flatten(x, offset):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / n * (n.mean() + offset)
    # single planetary set, single herringbone
    # matching into compound set left to other function
    sg, pg, rg = generate_gearset(s, p, r, b, cycloid=True)

    def do_stuff(g, teeth, offset, L, suffix):
        print(suffix)
        gf = g.copy(vertices=flatten(g.vertices, offset=offset))
        gfe = extrude_twist(gf, L=L, N=100, offset=0, twist=lambda z: np.abs(z - 0.5) * 2 * np.pi * 2 / teeth)
        ge = extrude_twist(g, L=L, N=100, offset=0, twist=lambda z: np.abs(z - 0.5) * 2 * np.pi * 2 / teeth)
        b = np.clip(((1 - np.abs(L/2 - ge.vertices[:, 2:]) / (L/2))) * 20, 0, 1)
        ge = ge.copy(vertices=ge.vertices * (b) + gfe.vertices * (1 - b))
        ge.save_STL(f'gear_{suffix}.stl')

    do_stuff(sg, -s, -0.005, 0.5, 'sun')
    do_stuff(pg, p, -0.005, 0.5, f'planet_{n}')
    do_stuff(rg, r, 0.005, 0.5, 'ring')

    # sgf = sg.copy(vertices=flatten(sg.vertices, offset=-0.005))
    # sgfe = extrude_twist(sgf, L=0.5, N=100, offset=0, twist=lambda z: np.abs(z - 0.5) * 2 * np.pi * 2 / -s)
    # sge = extrude_twist(sg, L=0.5, N=100, offset=0, twist=lambda z: np.abs(z - 0.5) * 2 * np.pi * 2 / -s)
    # b = np.clip(((1 -np.abs(0.25 - sge.vertices[:, 2:]) / 0.25)) * 20, 0, 1)
    # sge = sge.copy(vertices=sge.vertices * (b) + sgfe.vertices * (1 - b))
    # sge.save_STL('sun_gear.stl')
    #
    # pge = extrude_twist(pg, L=0.5, offset=0, factor=2/p)
    # pge.save_STL('planet_gear.stl')
    #
    # rge = extrude_twist(rg, L=0.5, offset=0, factor=2/r)
    # rge.save_STL('ring_gear.stl')


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

    import itertools

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

    with open('foo.gcode', 'w') as fh:
        fh.writelines(gcode)




