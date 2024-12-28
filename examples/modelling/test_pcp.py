"""
Some hacky scripts to generate 'Progressive Cavity Pump' geometries
"""
from examples.modelling.gear import *

# extrude_twist(ring(epitrochoid(15/2, 1, 0)), 60, 0, 1).save_STL('cylinder.stl')
# quit()

# fraction of epicyloid vs hypocycloid
f = 0.999
N = 5       # number of inner gear teeth
if False:
    M = 6
    ratio = (N * M) / (N - M)
    print(ratio)
    quit()
target_radius = 8
L = 50
stages = 2
# printer line thickness; or conservative estimate thereof
# should also contain layer thickness factor? thick lines key together more at overhangs;
# want controllable keying. that said, in areas without overhang, logic is different
# controllable roughness regardless of overhang would be great.
offset = 0.0

rotor = gear(N, N, f, res=295)
stator = gear(N+1, N+1, f, res=295)

# expand-contract cycle doesnt work;
# insofar as offset smaller than curvature radius, operation is reversible;
# and insofar not reversible, it messes things up
# offsets that dont violate curvature still viable tho
# but shrinking offsets increase discontinuity
# whereas growing leads to poorer pressure angles and more unbalanced curvature
#
# b = 2.3
# rotor = buffer(rotor, b)
# stator = buffer(stator, b)

# rotor = buffer(ring(hypotrochoid(N, N, 0.5)), 1.1)
# stator = buffer(ring(hypotrochoid(N + 1, N + 1, 0.5)), 1.1)

# rotor = hypo_gear(N, N, 0.9, f=0.9)
# stator = hypo_gear(N+1, N+1, 0.9, f=0.9 * (N / (N + 1)))
# f = 0.7
# scale = N / (N - (1-f))
# rotor = hypo_gear(N*scale, N, 1.2*0, f=f)#.transform(np.eye(2) * scale)
# stator = hypo_gear((N+1), N+1, 1.2*0, f=f)


# scaling that achieves the target radius
max_radius = np.linalg.norm(stator.vertices, axis=1).max()
scale = target_radius / max_radius
rotor = rotor.scale(scale)
stator = stator.scale(scale)

# FIXME: for some stupid reason this buffer does not appear to correspond to absolute coords
b = -1
rotor = buffer(rotor, b)
stator = buffer(stator, b)



if False:
    layer_height = 0.21
    wall_thickness = 0.4
    filament_diameter = 1.75
    ff = layer_height * wall_thickness / (filament_diameter ** 2 / 4 * np.pi)
    generate_gcode(rotor, height=100, layer_height=layer_height, filament_factor=ff, twist=1)
    quit()




# translation which aligns the tips
translation = np.linalg.norm(stator.vertices, axis=1).max() - np.linalg.norm(rotor.vertices, axis=1).max()

if True:
    fig, ax = plt.subplots(1)
    rotor.translate([translation, 0]).plot(ax=ax, plot_vertices=False)
    stator.plot(ax=ax, plot_vertices=False)

    circle = ring(sinusoid(1, amplitude=0, pitch_radius=0.5))
    circle.scale(2).plot(ax=ax, plot_vertices=False)
    circle.scale(5).translate([translation, 0]).plot(ax=ax, plot_vertices=False)
    circle.scale(6).translate([translation, 0]).plot(ax=ax, plot_vertices=False)
    plt.show()
    quit()



print(max_radius)
print(rotor.volume())
print(stator.volume())
print('cc / rev')
print((stator.volume() - rotor.volume()) * L / 1000)

# generate 3d extrusions
if False:
    sleeve_clearance = 0.04 # 0.05 quite tight; 0.10 a bit on the loose side
    backing_clearance = 0.4
    wall_thickness = 0.4
    interference = 0.05  # interference between inner and outer, as modelled
    # note that with filler additive, positive sleeve clearance can translate into positive interference too
    # perhaps best to keep interference low
    offset = wall_thickness - interference
    sleeve_offset = wall_thickness + sleeve_clearance
    backing_offset = wall_thickness + backing_clearance

    # extrude_twist(
    #     buffer(rotor, -offset * 0.5),
    #     L=L * stages,
    #     offset=-offset*0.5*0,
    #     factor=stages).save_STL('gear_inner.stl')
    extrude_twist(
        buffer(rotor, -offset * 0.5 + sleeve_offset),
        L=L * stages,
        offset=-offset*0.5*0,
        sleeve=+1,
        factor=stages).save_STL('gear_inner_sleeve.stl')
    # extrude_twist(
    #     buffer(rotor, -offset * 0.5 - backing_offset),
    #     L=L * stages,
    #     offset=-offset*0.5*0,
    #     factor=stages).save_STL('gear_inner_backing.stl')
    # extrude_twist(
    #     buffer(rotor, -offset * 0.5 + sleeve_offset + backing_offset +0.3),
    #     L=L * stages,
    #     offset=-offset*0.5*0,
    #     sleeve=+1,
    #     factor=stages).save_STL('gear_inner_sleeve_backing.stl')

    # note; outer
    # extrude_twist(
    #     buffer(stator, +offset * 0.5),
    #     L=L * stages,
    #     offset=+offset*0.5*0,
    #     factor=stages * N / (N+1)).save_STL('gear_outer.stl')
    extrude_twist(
        buffer(stator, +offset * 0.5 - sleeve_offset),
        L=L * stages,
        offset=+offset*0.5*0,
        sleeve = -1,
        factor=stages * N / (N+1)).save_STL('gear_outer_sleeve.stl')
    # extrude_twist(
    #     buffer(stator, +offset * 0.5 + backing_offset + 0.3),
    #     L=L * stages,
    #     offset=+offset*0.5*0,
    #     factor=stages * N / (N+1)).save_STL('gear_outer_backing.stl')
    # extrude_twist(
    #     buffer(stator, +offset * 0.5 - sleeve_offset - backing_offset),
    #     L=L * stages,
    #     offset=+offset*0.5*0,
    #     sleeve = -1,
    #     factor=stages * N / (N+1)).save_STL('gear_outer_sleeve_backing.stl')

    quit()

path = r'../output/gear5'
from examples.util import save_animation
frames = 60
for i in save_animation(path, frames=frames, overwrite=True):
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    a = i / frames * 2 * np.pi
    if False:
        rotor.\
            transform(rotation(a / N * (N+1))).\
            translate([translation, 0]).\
            transform(rotation(-a / (N+1) * (N+1))).\
            plot(ax=ax, plot_vertices=False, color='b')
        stator.plot(ax=ax, plot_vertices=False, color='r')

    else:
        rotor.\
            transform(rotation(a / N)).\
            translate([translation, 0]).\
            plot(ax=ax, plot_vertices=False, color='b')
        stator.\
            transform(rotation(a / (N+1))).\
            plot(ax=ax, plot_vertices=False, color='r')

    ax = plt.gca()
    ax.set_xlim(*stator.box[:, 0]*1.1)
    ax.set_ylim(*stator.box[:, 1]*1.1)
    plt.axis('off')

    # plt.show()