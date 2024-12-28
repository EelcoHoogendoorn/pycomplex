import matplotlib.pyplot as plt
import numpy as np

from examples.modelling.compound import *


def test_solve_compound_symbolic():
    r = solve_compound_symbolic()
    print(r)


def test_4point_num():
    pressure_concentration = 2.36 #@ 25 deg pressure angle
    # pressure_concentration = 4 #@ 15 deg pressure angle
    D = 160e-3
    # does not matter for transmission ratios or axial/torque ratio
    # 40x4g = 160g; starting to add up
    d = 8e-3        # 3000N
    # d = 10e-3       # 5000N; x40=200k; only 2x of tapered roller!
    d = 12e-3       # 8000N
    V = 4/3*np.pi*(d/2)**3
    gap = 1e-3
    n_max = int(np.pi*D / (d + gap))
    print(n_max)
    n = n_max
    mass = V * 7.8e3 * n_max
    print('mass', mass)
    normal = 400 # 2gpa for r=4 in 4.6 race
    normal = 400
    axial = normal / pressure_concentration * n
    print('required axial preload', axial)
    # seen values up to 0.11 mentioned
    # 0.09 at only 1.1gpa, not bad
    traction_coef = 0.09  # https://www.idemitsu.com/en/business/lube/tdf/traction.html
    torque = n * (D + d) / 2 * normal * traction_coef
    print('torque:', torque)
    print(axial / torque)


def test_4point():
    """test tilted 4-point contact ball bearing reducer"""
    # FIXME: add both in-plane and tangential force balance calcs
    Dr = 150 / 12   # rolling vs ball diameter

    # conicity; major driver of ratios
    cone = 5
    # tilt of contacts. does not seem hugely influential
    tilt = -5
    # flattness of contacts.
    # flatter means more radial, less axial capacity.
    # more squat, also means less contact point rotation
    # low squat also lowers ratios
    # more squat, also means lower axial preload requirements?
    # high squat also eats into moment arm to transfer torque over the ball
    squat = 20
    # assymmetry; diff in spacing between contat points, inner/outer
    asym = -2
    asym = 0

    a = [
        [-135 - cone + tilt - squat + asym, +135 - cone + tilt + squat - asym],
        [-45 + cone + tilt + squat + asym, 45 + cone + tilt - squat - asym],
    ]
    a = np.array(a)
    print(a)
    ar = a / 180 * np.pi
    P = np.dstack([np.cos(ar), np.sin(ar)])
    print(P[..., 1])
    print('preload concentration')
    print(np.sqrt(1 + (P[..., 0] / P[..., 1])**2))
    # return
    res = solve_4point_symbolic(Dr, P)['[rob, rib, rot]']
    print(res)
    # print('ratio')
    ratio = (1 / res['r'])
    axis = np.array([res['py'], -res['px']])
    axis = axis / np.linalg.norm(axis)


    import matplotlib.pyplot as plt
    plt.scatter([0], [0])
    plt.scatter(P[..., 0].flatten(), P[..., 1].flatten())
    plt.plot(*(axis[:, None] * [-1, +1]))

    text = lambda t, p, a: plt.text(*p, t, horizontalalignment='center', verticalalignment='center', rotation=a)
    for i in range(2):
        for j in range(2):
            text(a[i][j], P[i, j] * 0.8, 0)

    text('ground', P[1, 0] * 1.1, a[1, 0] - 90)
    text('output', P[1, 1] * 1.1, a[1, 1] - 90)
    text('input', P[0, :].mean(axis=0) * 1.2, a[0].mean(axis=0) + 90)


    a = np.linspace(0, 2*np.pi, 200)
    xy = np.array([np.cos(a), np.sin(a)]).T
    # plot cup shapes
    delta = xy.reshape(-1, 1, 2) - P.reshape(1, 4, 2)
    dist = np.linalg.norm(delta, axis=2)
    mindist = np.min(dist, axis=1)
    curve_ratio = 0.5   # 0=unit circle, 1=flat
    r = mindist ** 2 * curve_ratio / 2 + 1
    ds = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    r[np.logical_and(xy[:, 0] < 0, mindist > ds[0] / 1.9)] = np.nan
    r[np.logical_and(xy[:, 0] > 0, mindist > ds[1] / 2.1)] = np.nan
    # r[[0, 1, -1, -2]] = np.nan # split ring
    plt.plot(*(xy * r[:, None]).T)

    plt.plot(*xy.T)

    plt.axis('equal')
    plt.title(f'input/output =  {ratio:.2f}')
    plt.xlabel('inner <-> outer')
    plt.ylabel('bottom <-> top')
    plt.show()



def test_brute_compound_gear():
    brute_compound_gear()


def test_flex():
    n = 3
    d = 3
    b = 20
    n = 2
    d = 0
    b = 30
    r1, s1 = b, b
    r2, s2 = b + d + n, b + d
    # r1, s1 = 60, 60
    # r2, s2 = 62, 60
    m1 = r1 - s1
    m2 = r2 - s2
    r = (m2 / r2 - m1 / r1)
    print(1/r)
    return



def test_sun_io():

    # finer teeth
    compound((13*2, 8*2, 29*2), (14*2, 7*2, 28*2), 7)#10.666666666666666, 0.8091807156732074, 0.3333333333333333)
    # love this one too
    compound((19, 9, 37), (18, 10, 38), 8)#8.756756756756758, 0.3964594300514207, 0.2857142857142857)
    compound((13, 17, 47), (18, 17, 52), 5) #10.4
    compound((16, 14, 44), (19, 14, 47), 6) #15.666


    # big bore sun attached
    compound((19, 6, 31), (18, 7, 32), 10)#7.680000000000004, 0.505688048534976, 0.4) # mixed planets
    compound((18, 7, 32), (15, 5, 25), 10)# 6.999999999999997, 0.505688048534976, 0.4) # two plaets
    compound((13, 7, 27), (14, 6, 26), 8  )#4.8999999999999995, 0.4045504388279803, 0.4) # single planet?


    # nice and low ring gear tooth count
    # note; single tooth diff on all accounts. kinda cool?
    compound((13, 8, 29), (14, 7, 28), 7)#10.666666666666666, 0.8091807156732074, 0.3333333333333333)


    compound((9, 6, 21), (5, 5, 15), 5)#, 5.999999999999997, 0.7831853071795856, 0.3333333333333333)
    # this one is very magical somehow; especially if leaving out 22 ring
    # tiny tooth count if that your thing and great ratio. solid bore but stiff planets
    compound((8, 7, 22), (7, 5, 17), 6)# 13.22222222222223, 0.3034128291209847, 0.4)


def test_save_json():
    fig, ax = plt.subplots()
    plot_planetary(*(4, 2, 8), 6, ax=ax) # nice bearing config for p=8
    plot_planetary(*(4, 3, 8), 6, ax=ax) # nice bearing config for p=8

    # n = 2
    # plot_planetary(*(n, n, n*3), 4, ax=ax)
    # n = 7
    # plot_planetary(*(n, n+1, n*3+2), 5, ax=ax)
    # n = 5
    # plot_planetary(*(n, n, n*3), 5, ax=ax)
    # save_json_planetary(*(n, n+1, n*3+2), 5)       # trochoidal
    plt.show()
    # save_json_planetary(*(7, 5, 17), 6)       # trochoidal


def test_save_json_cycloid():
    fig, ax = plt.subplots()
    import json
    n = 7
    R = 80
    r = R / n

    gear(R, n, f=1)
    # p = trochoid_part(R, r, +1, res=10, endpoint=True)
    # ax.plot(*p.T)
    # plt.show()
    # with open(f'../output/cycloid_{R}_{n}.json', 'w') as fh:
    #     json.dump([g.tolist() for g in [p]], fh)


def test_extrude():
    extrude_planetary_STL(*(7, 5, 17), 6)       # trochoidal

    # extrude_planetary_STL(*(21, 24, 69), 5)

    # two equal-planet meidum tooth count solutions
    # compound((15, 15, 45), (10, 15, 40), 5)#32.0, 0.06932108931632115, 0.16666666666666666)
    # compound((23, 22, 67), (18, 22, 62), 5)#48.52173913043479, 4.66370614359171, 0.11111111111111112)


def test_slippage():
    # about 3x slippage reduction in sinus pattern. might be worth it, but still substantial
    print(calc_slippage_sinusoid(20, 20))
    print(calc_slippage(20, 20))

    N = np.arange(1, 10000)
    s = [calc_slippage(n, n * 1) for n in N]
    plt.plot(np.log(N), np.log(np.log(s)))
    plt.show()


def test_overlap():
    # some configs with overlap
    compound((40, 24, 88), (44, 24, 92), 8)#73.60000000000001, 0.21091374290611498, 0.11764705882352941)

    compound((11, 14, 39), (10, 15, 40), 5)#101.81818181818183, 0.06932108931632115, 0.2)
    compound((21, 24, 69), (16, 24, 64), 5)#54.857142857142854, 0.13182108931632244, 0.11111111111111112)
    # about half-tooth clearance
    compound((22, 23, 68), (17, 23, 63), 5)# 51.54545454545455, 0.3549437453735526, 0.11111111111111112)


def test_animate():
    # n = 5
    # animate_planetary(*(n, n, n*3), 5)
    # n = 7
    # animate_planetary(*(n, n+1, n*3+2), 5)
    animate_planetary(*(4, 2, 8), 6) # nice bearing config for p=8

    # animate_planetary(*(1, 1, 3), 4)
    # animate_planetary(*(18, 10, 38), 8)
    # animate_planetary(7, 5, 17, 6)
    return

    animate_planetary(1, 2, 5, 3)
    animate_planetary(5, 19, 43, 3)
    animate_planetary(34, 22, 78, 7)
    animate_planetary(4, 2, 8, 3)


def test_min_p_diff():
    # targetting small p-tooth count diff; zero infact
    # compound((43, 7, 57), (35, 5, 45), 20)#24.418604651162756, 1.011376097069952, 0.4)
    # compound((35, 5, 45), (43, 7, 57), 20)#21.714285714285715, 1.011376097069952, 0.4)
    # quit()
    # first one nice; small clearance
    compound((23, 22, 67), (18, 22, 62), 5)#48.52173913043479, 4.66370614359171, 0.11111111111111112)
    compound((30, 20, 70), (35, 20, 75), 5)#50.0, 47.07963267948965, 0.09090909090909091)

    # some low gear ratios; low gear ratios and equal p give high diameter ratios it seems.
    compound((35, 14, 63), (21, 14, 49), 7)#9.799999999999999, 4.9557428756427555, 0.14285714285714285)
    compound((41, 19, 79), (23, 19, 61), 6)#9.918699186991871, 8.446891450771332, 0.1)
    compound((34, 16, 66), (19, 16, 51), 5)# 10.0, 21.955742875642756, 0.1)

    # these have interchangeable planets
    compound((5, 19, 43), (8, 19, 46), 3)#147.20000000000002, 8.898223686155035, 0.1111111111111111)
    # NOTE: very nice one for 1800rpm/2nm motor
    compound((21, 24, 69), (26, 24, 74), 5)#63.42857142857142, 9.371669411540694, 0.1)
    # ring tooth diff equals planets; is that good, bad or neutral?
    # wait very common feature of these zerop-p-tooth diff setups
    compound((34, 22, 78), (41, 22, 85), 7, 0.7, 0.3)#40.0, 10.929188601028414, 0.1111111111111111)
    compound((33, 12, 57), (42, 12, 66), 9)#20.0, 27.371669411540694, 0.16666666666666666)
    compound((33, 11, 55), (44, 11, 66), 11)#16.0, 11.730076757950883, 0.19999999999999998)
    # these somehow do not have identical planets; two types
    # with 25% phase shift and flipping mirror pairs,
    # can still print only single exchangable planet though
    # kinda nice for assembly for also especially if urethane molding or the like
    # also easy to model lofts
    compound((11, 23, 57), (13, 23, 59), 4)  # 182.36363636363635, 3.314150222052973, 0.1111111111111111)
    compound((25, 23, 71), (28, 23, 74), 6)#94.71999999999998, 1.2964473723100696, 0.11764705882352941)
    # this might be one of my current faves.
    compound((38, 22, 82), (42, 22, 86), 8)  # 67.89473684210526, 1.4955592153875727, 0.125)
    compound((39, 16, 71), (44, 16, 76), 10)#42.87179487179487, 4.787595947438632, 0.16666666666666666)


def test_max_p_diff():
    # targetting large tooth count diff
    compound((9, 5, 19), (40, 23, 86), 7)#191.11111111111228, 6.482297150257104, 0.1111111111111111)
    compound((5, 5, 15), (26, 24, 74), 5)#148.0, 3.9159265358979276, 0.1)
    compound((27, 15, 57), (42, 24, 90), 6)#233.3333333333355, 79.44689145077132, 0.045454545454545456)

def test_max_teeth():
    # attempt at going to higher tooth counts
    # depth of these teeth top to bottom is about 1% of total diameter;
    # 1mm on 10cm; tip to tip distance about 4x that, or 4mm
    # pretty tricky to fdm with 0.6mm nozzle
    compound((17, 18, 53), (15, 15, 45), 5) #222.35294117647183, 10.955742875642741, 0.14285714285714285)
    compound((22, 20, 62), (21, 18, 57), 6, 0.6, 0.5)#181.363636363637, 1.946891450771318, 0.14285714285714285)
    compound((21, 15, 51), (19, 17, 53), 6, 0.4, 0.6)#37.857142857142875, 2.5973355292325664, 0.16666666666666666)
    compound((31, 18, 67), (35, 21, 77), 7)#208.64516129032174, 18.429188601028414, 0.125)
    compound((37, 19, 75), (34, 18, 70), 8)#201.297297297297, 10.362817986669256, 0.14285714285714285)
    compound((37, 17, 71), (38, 16, 70), 9)#64.32432432432441, 8.146003293848821, 0.16666666666666666)
    compound((39, 16, 71), (43, 17, 77), 10, 0.6)#138.99487179487178, 4.787595947438632, 0.16666666666666666)

def test_min_teeth():
    # low total tooth count

    # # this one doesnt actually work; 6 rollers in 4 lobes?
    # compound((5, 1, 7), (2, 1, 4), 6, 0.3, 0.5)#3.2, 2.9247779607693793, 1.0)
    # # same here; 16 rollers in 9 lobes
    # compound((7, 1, 9), (15, 1, 17), 16, 0.8)#4.857142857142858, 8.632741228718345, 1.0)
    # # 4 in 3 lobes
    # compound((1, 1, 3), (3, 1, 5), 4, 0.4, 0.3)#10.0, 1.7831853071795862, 1.0)
    # lowest tooth crazy found so far
    compound((1, 2, 5), (2, 1, 4), 3, 0.95, 0.25)#, 16.0, 2.4247779607693793, 1.0)
    # very low sum of teeth box
    compound((3, 3, 9), (4, 2, 8), 3)#16.0, 8.349555921538759, 0.5)

    compound((8, 4, 16), (9, 3, 15), 8, 0.6, 0.4)#15.0, 3.6991118430775174, 0.6666666666666666)
    # super low ratio
    compound((7, 2, 11), (4, 2, 8), 6, 0.3, 0.6) # 6.857142857142857, 5.849555921538759, 0.6666666666666666)
    # actually looking kinda useful. seems to have all identical planets too
    compound((6, 4, 14), (5, 5, 15), 5, 0.4, 0.6)#20.0, 3.9159265358979276, 0.5)

def test_various():
    # visually pleasing boxes
    compound((16, 8, 32), (14, 6, 26), 8, 0.5, 0.4)#39.0, 3.398223686155035, 0.3333333333333333)
    compound((15, 6, 27), (14, 7, 28), 7, 0.4, 0.5)#22.4, 9.973445725385659, 0.3333333333333333)

    # getting a lower ratio in there
    compound((12, 6, 24), (10, 8, 26), 6, 0.4, 0.6)#13.0, 0.5486677646162761, 0.3333333333333333)
    compound((10, 8, 26), (11, 7, 25), 6, 0.6, 0.4)# 40.0, 0.5486677646162761, 0.3333333333333333)

    compound((9, 6, 21), (8, 7, 22), 5)# 29.333333333333336, 5.123889803846893, 0.3333333333333333)
    # like below, but with bigger hole
    compound((11, 7, 25), (12, 6, 24), 6)# 30.545454545454568, 7.548667764616276, 0.3333333333333333)
    # feels very balanced visually somehow
    compound((8, 7, 22), (9, 6, 21), 5 ) #36.750000000000014, 5.123889803846893, 0.3333333333333333)
    # crazy gear ratio
    compound((13, 21, 55), (10, 16, 42), 4)#2306.769230769239, 1.6814089933346281, 0.11764705882352941)

def test_50():
    # stuff in 50 range
    # really big teeth; viable with pointy planets
    compound((15, 5, 25), (11, 4, 19), 10, 0.4, 0.6) #50.66666666666667, 4.483889803846893, 0.5)
    compound((14, 4, 22), (19, 5, 29), 12, 0.6, 0.4)#49.714285714285715, 4.548667764616276, 0.5)
    compound((17, 4, 25), (23, 5, 33), 14, 0.6, 0.4)#46.588235294117645, 5.973445725385659, 0.5)
    compound((20, 4, 28), (27, 5, 37), 16, 0.6, 0.4)#44.4, 8.758223686155034, 0.5)
    compound((23, 4, 31), (31, 5, 41), 18, 0.6, 0.4)# 42.78260869565218, 8.82300164692441, 0.5)
    compound((29, 4, 37), (39, 5, 49), 22, 0.6, 0.4) #40.55172413793103, 11.672557568463176, 0.5)

    compound((5, 4, 13), (9, 6, 21), 6, 0.66, 0.33)#50.400000000000006, 0.27433388230813804, 0.4)
    compound((9, 3, 15), (11, 4, 19), 6, 0.4, 0.4)#50.66666666666667, 16.699111843077517, 0.4)


def test_power():
    # power-density optimized ones
    # double-odd planets; quite rare. tiny sun like this might be good since little torque requirement there anyway
    compound((2, 7, 16), (3, 9, 21), 3, 0.6, 0.4)# 440.99999999999795, 0.27433388230813804, 0.25)
    compound((23, 13, 49), (21, 12, 45), 6, 0.3, 0.3)#610.4347826087073, 19.672557568463176, 0.16666666666666666)
    compound((34, 8, 50), (39, 9, 57), 12, 0.3, 0.3)# 187.76470588235296, 27.946891450771318, 0.25)
    compound((25, 11, 47), (19, 8, 35), 9)#123.19999999999969, 3.097335529232552, 0.25)
    # decent ratio/torque products. also allows internal motor? goes well with b=0.4
    compound((31, 5, 41), (39, 6, 51), 18, 0.4, 0.3)# 65.80645161290307, 18.097335529232552, 0.4)
    compound((27, 5, 37), (34, 6, 46), 16)#68.1481481481484, 15.53096491487338, 0.4)
    compound((23, 5, 33), (29, 6, 41), 14)#71.30434782608667, 12.964594300514207, 0.39999999999999997)
    compound((29, 6, 41), (35, 7, 49), 14, 0.3, 0.3)#101.37931034482716, 19.955742875642756, 0.3333333333333333)


def test_min_s():
    # some smaller suns;
    compound((5, 4, 13), (7, 5, 17), 6, 0.7, 0.5)# 81.6, 0.27433388230813804, 0.5)
    compound((5, 5, 15), (3, 2, 7), 5, 0.5, 0.6)#28.0, 1.4159265358979312, 0.5)

    # true torque monster
    compound((5, 4, 13), (6, 3, 12), 6, 0.75, 0.3)#19.2, 0.27433388230813804, 0.6666666666666666)
    # impressive torque for n=3 still
    compound((5, 4, 13), (6, 3, 12), 3, 0.4, 0.3)#, 19.2, 12.274333882308138, 0.3333333333333333)
    # these super low sun gear configs are pretty cool too
    compound((3, 3, 9), (2, 4, 10), 3, 0.3, 0.6)#20.0, 2.8495559215387587, 0.5)
    # also good torque; interesting for 3-3 config
    compound((5, 3, 11), (7, 3, 13), 4, 0.4, 0.3)#20.799999999999997, 10.132741228718345, 0.4)


# also good torque rating
# compound((7, 5, 17), (8, 4, 16), 6)#22.85714285714285, 2.6991118430775174, 0.5)
# compound((35, 5, 45), (26, 4, 34), 20)#38.85714285714286, 10.247779607693786, 0.5)

def test_gdfw():
    # pretty decent torque rating in this category
    # compound((39, 10, 59), (28, 7, 42), 14)#150.76923076923168, 3.938040025899852, 0.2857142857142857)
    # one of the highest ratios with decent torque rating
    compound((10, 12, 34), (11, 13, 37), 4) # 976.800000000014, 9.115038378975441, 0.16666666666666666)
    # geardownforwhat cf record holder; looking rather good now that upping the tooth count
    compound((30, 10, 50), (34, 11, 56), 10, 0.4, 0.4)  # 149.33333333333334, 15.66370614359171, 0.22222222222222224)
    # this one seems strictly superior; if ignoring slippage friction
    compound((27, 8, 43), (23, 7, 37), 10, 0.4, 0.4)#153.48148148148147, 17.247779607693786, 0.2857142857142857)


def test_10():
    # nice ones in the 10 gear range. simpler gearboxes may exist in this range but not with such nice self bearing props
    # 10 range should be quite easy to backdrive?
    # compound((12, 4, 20), (8, 4, 16), 8)#10.666666666666666, 1.6991118430775174, 0.5)
    # compound((20, 4, 28), (14, 4, 22), 12)#8.8, 4.548667764616276, 0.5)
    # kinda cool; seems all planets are idential
    compound((22, 4, 30), (21, 5, 31), 13, 0.3, 0.6)#11.272727272727272, 11.681408993334628, 0.5) # noice; usually the primes dont do this well
    # compound((34, 4, 42), (33, 5, 43), 19)#10.117647058823529, 19.38052083641213, 0.5)
    # 80 range; seem to max out around 0.4 torque
    compound((19, 6, 31), (15, 5, 25), 10, 0.4, 0.5)#, 78.9473684210524, 7.831853071795855, 0.4)
    compound((11, 5, 21), (14, 6, 26), 8, 0.6, 0.5)#94.54545454545489, 5.26548245743669, 0.4)
    # at fairly lonely heights in the 160 range; better then gdfw, except less interior space
    compound((11, 7, 25), (9, 6, 21), 6)#160.3636363636356, 5.123889803846893, 0.3333333333333333)


def test_eval_symbolic():
    # gdfw; large sun. s-r-s config viable
    eval_compound_symbolic((30, 10, 50), (34, 11, 56))#, 10, 0.4, 0.4)  # 149.33333333333334, 15.66370614359171, 0.22222222222222224)
    eval_compound_symbolic((13, 21, 55), (10, 16, 42))
    # small sun example; s-r-s config useless
    eval_compound_symbolic((5, 19, 43), (8, 19, 46))#, 3)  # 147.20000000000002, 8.898223686155035, 0.1111111111111111)


    eval_compound_symbolic((35, 14, 63), (21, 14, 49))#, 7)  # 9.799999999999999, 4.9557428756427555, 0.14285714285714285)
    eval_compound_symbolic((41, 19, 79), (23, 19, 61))#, 6)  # 9.918699186991871, 8.446891450771332, 0.1)
    eval_compound_symbolic((34, 16, 66), (19, 16, 51))#, 5)  # 10.0, 21.955742875642756, 0.1)


def test_solve_compound():
    # 2k gear ratio
    solve_compound((13, 21, 55), (10, 16, 42))
    solve_compound((23, 22, 67), (18, 22, 62))

    solve_compound((23, 22, 67), (18, 22, 62))#, 5)  # 48.52173913043479, 4.66370614359171, 0.11111111111111112)
    solve_compound((30, 20, 70), (35, 20, 75))#, 5)  # 50.0, 47.07963267948965, 0.09090909090909091)

    # some low gear ratios; low gear ratios and equal p give high diameter ratios it seems.
    solve_compound((35, 14, 63), (21, 14, 49))#, 7)  # 9.799999999999999, 4.9557428756427555, 0.14285714285714285)
    solve_compound((41, 19, 79), (23, 19, 61))#, 6)  # 9.918699186991871, 8.446891450771332, 0.1)
    solve_compound((34, 16, 66), (19, 16, 51))#, 5)  # 10.0, 21.955742875642756, 0.1)
    # these have interchangeable planets
    solve_compound((5, 19, 43), (8, 19, 46))#, 3)  # 147.20000000000002, 8.898223686155035, 0.1111111111111111)
    solve_compound((21, 24, 69), (26, 24, 74))#, 5)  # 63.42857142857142, 9.371669411540694, 0.1)
    solve_compound((34, 22, 78), (41, 22, 85))#, 7, 0.7, 0.3)  # 40.0, 10.929188601028414, 0.1111111111111111)
    solve_compound((33, 12, 57), (42, 12, 66))#, 9)  # 20.0, 27.371669411540694, 0.16666666666666666)
    solve_compound((33, 11, 55), (44, 11, 66))#, 11)  # 16.0, 11.730076757950883, 0.19999999999999998)
    # these somehow not; two types
    # with 25% phase shift and flipping mirror pairs,
    # can still print only single exchangable planet though
    # kinda nice for assembly for also especially if urethane molding or the like
    # also easy to model lofts
    solve_compound((11, 23, 57), (13, 23, 59))#, 4)  # 182.36363636363635, 3.314150222052973, 0.1111111111111111)
    solve_compound((25, 23, 71), (28, 23, 74))#, 6)  # 94.71999999999998, 1.2964473723100696, 0.11764705882352941)
    # this might be one of my current faves.
    solve_compound((38, 22, 82), (42, 22, 86))#, 8)  # 67.89473684210526, 1.4955592153875727, 0.125)
    solve_compound((39, 16, 71), (44, 16, 76))#, 10)  # 42.87179487179487, 4.787595947438632, 0.16666666666666666)
