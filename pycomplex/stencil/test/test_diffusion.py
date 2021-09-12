"""Dump things here to make self contained diffusion example



"""

import numpy as np

from pycomplex.stencil.complex import StencilComplex2D
from pycomplex.stencil.diffusion import Diffusion
from pycomplex.stencil.block import BlockArray


def rect(complex, pos, size):
    r = complex.topology.form(0)
    r[0, pos[0]-size[0]:pos[0]+size[0], pos[1]-size[1]:pos[1]+size[1]] = 1
    return r

def circle(complex, pos, radius):
    p = complex.primal_position[0]
    d = np.linalg.norm(p - pos, axis=-1) - radius
    return 1 / (1 + np.exp(d * 8))


def test_diffusion(show_plot):
    """setup and solve 2d diffusion problem

    solve by means of dec-multigrid; without variable elimination
    this is likely to be less efficient but generalizes better to more complex problems;
    that is zero-resistance term precludes elimination
    and even when it is not precluding elimination,
    dividing by small values leads to poor conditioning.
    remains to be seen how this works out in first order system exactly,
    but the suspicion is that it will work much better

    """

    """    
    pin value at boundary to zero? infact not obvious how to set these constraints
    0 * T + d flux = Q
    just introduce some nonzeros in diagonal of T?
    or need we model auxiliary constraint equations?
    no not really have room for hard constraints though;
    either redundant or need to be satisfied in least square sense
    seems like we can add relation between divergence and temperature
    cannot relate flux and temp directly
    but divergence and temp relation over distance can have same effect
    the higher weight we set on temp, the more the equation becomes dominated to just being a temp constraint
    """

    """
    when does it make sense to eliminate a variable, versus solving it in first order normal form?
    sometimes there is no choice; can only eliminate if there is an invertable diagonal.
    examples of impossible elimination are stokes-pressure and EM-field

    elimination seems more efficient a priori. but in the limit to incompressibility probably a bad idea
    should we eliminate pressure in near-incompressible, or rubber-like material?
    or should we eliminate flux in reservoir simulation?
    of course with block diagonal it is still trivial to solve these variables with one jacobi step,
    but simultanious updates in other vars mean making residual zero still isnt trivial,
    meaning there is something to move to the coarse grid.
    
    which makes an interesting point; we can do per-variable gauss-seidel
    that way, the elimination candidate would have zero residual
    is this equivalent to variable elimination, if done in the right order?
    
    not really. note that variable elimination fails for diffusion in areas where diagonal goes to zero;
    that is, where gradients in the 0-form are constrained to be 0. using normal equation method, 
    like other zero-diagonal terms, this is not an issue, and we can solve for this flat region,
    which is effectively a super-conductor.
    
    does it go both ways; does normal equation enable regions of zero-conductivity too?
    here, diagonal goes to infinity; or off-diag to zero. no matter the gradient, there is no flux
    this would have been possible with var-eliminated second order laplace already.
    
    but so even for a simple diffusion problem, first order form has objective benefits, at least in terms of generality 
    
    would there by any benefit in having both a conductivity and resistance field,
    and coarsening both independently?
    """

    complex = StencilComplex2D.from_shape((128, 128))

    # setup simple source and sink
    source = complex.topology.form(0)

    mid = 8
    ext = 3
    sep = 4




    # source = circle(complex, [8, 8], 4)
    source = rect(complex, [64, 64], [32, 16])
    source -= rect(complex, [32, 64], [32, 8])
    source = np.clip(source, 0, 1)
    constraint = 1 - source

    # complex.plot_0(source)
    # complex.plot_2(complex.topology.averaging_operators_0[2] * source)
    # show_plot()

    conductance = complex.topology.form(1, init='ones')
    conductance[:, :, 64:] *= 1e-1
    resistance = complex.topology.form(1, init='ones')
    # resistance[:, :, 64:] *= 1e-1
    # FIXME: ah; the reason we cannot lower conductance in this manner is diagonal dominance of jacobi of course!
    # FIXME: what can be done about this? absorb scaling into unknown using (left?)right-preconditioner?
    # FIXME: also there is the question of overall balance of jacobi equations;
    # just because we tune a coefficient does not mean we intend to give the equation more weight
    # FIXME: would like to have sum of abolute coefficients to judge diagonal dominance; but not sure if easy to get from operator
    # NOTE: gauss-seidel does not require diagonal dominance, but SPD will do
    # seems like block-gauss-seidel may address both dominance and balance concerns?
    # seems like we have working block SG;
    # but it seems to have the same behavior as jacobi in this respect...
    # so is it not a problem with dominance, or do with still have dominance problems inside the diagonal block that we do jacobi on?
    # also; would the transpose algorithm have better results? should be promising for under-determined system;
    # which is what we are moving towards with the problematic changes
    # transpose algo seems worse; we have a zero-diagonal for coeffs that go to zero


    fields = {
        'constraint': constraint * 2e-1,
        'conductance': conductance,
        'resistance': resistance,
    }

    system = Diffusion.formulate(complex, fields)

    D = system.inverse_sg_normal_diagonal
    print()
    rhs = BlockArray([-source, complex.topology.form(1)])

    hierarchy = system.hierarchy(levels=5)
    from pycomplex.stencil.multigrid import solve_full_cycle
    from time import time
    t = time()
    # x = system.solve_minres_normal(rhs, x0=rhs * 0)
    x = solve_full_cycle(hierarchy, rhs, iterations=10)
    print()
    print(time() - t)

    complex.plot_0(x[0])

    show_plot()


def test_isolator(show_plot):
    """Test to drive conductance to zero in linear gradient field; cylinder in potential flow if you will

    With minres solver, it is working for both resistance and conductance = 0

    mg is still struggeling however; needs a thorough debugging. prone to kinda-working

    """
    complex = StencilComplex2D.from_shape((128, 128))

    # source = circle(complex, [8, 8], 4)
    source = rect(complex, [3, 64], [1, 64])
    source -= rect(complex, [128-3, 64], [1, 64])
    # constraint = 1 - source

    complex.plot_0(source)
    complex.plot_2(complex.topology.averaging_operators_0[2] * source)
    show_plot()

    quit()
    # conductance = complex.topology.form(1, init='ones')
    conductance = complex.topology.form(0, init='zeros') + circle(complex, [64, 64], 32)

    if True:
        import scipy.ndimage
        conductance[0] = scipy.ndimage.gaussian_filter(conductance[0], 2)
        complex.plot_0(conductance)
        show_plot()

    conductance = complex.topology.averaging_operators_0[1] * conductance

    resistance = complex.topology.form(1, init='ones')

    constraint = complex.topology.form(0, init='zeros')


    fields = {
        'constraint': constraint,
        'conductance': resistance,
        'resistance': conductance,
    }

    system = Diffusion.formulate(complex, fields)

    rhs = BlockArray([-source, complex.topology.form(1, init='zeros')])

    hierarchy = system.hierarchy(levels=5)
    from time import time
    t = time()
    x = system.solve(rhs, x0=rhs * 0, tol=1e-8)
    # x = system.solve_qmr(rhs, x0=rhs * 0, atol=1e-12)
    # x = solve_full_cycle(hierarchy, rhs, iterations=2)
    # x = system.solve(rhs)
    print()
    print(time() - t)

    complex.plot_0(x[0])
    complex.plot_1(x[1])

    show_plot()

    # show surface charges. we may multiply flux with mask, and then take the divergence,
    # to compute the induced charge density. this may be due to movement of free electrons,
    # or polarization-displacement currents; any gradient in permitivvity will result in an induced charge
    complex.plot_0(complex.topology.dual[0](conductance * x[1]) + source)

    show_plot()


def test_ray_mapping(show_plot):
    """Test mapping from domain boundary to object boundary.

    place source distribution on object boundary,
    matched by source distribution on domain boundary.

    We want to see flow lines connecting both of these, in a uniform manner

    looking pretty good.

    note that setting conductance to zero in interior leads to zero entries in system matrix.
    seems problematic for jacobi based smoother method; no diagonal to divide by.
    can we use another smoother? or simply dampen smoothing, with epsilon added to diag?

    periodic bcs really dont make sense here; would like to replace with prescribed-flux bcs.
    or just isolating ones with sink/source field; need that anyway for surface bcs.

    would this problem lend itself well to MC-integration? not sure noise is very welcome.
    also not sure about prescribed flux bcs.

    what options do we have for fused stencil? numba.stencil may be alright
    set cval to inf, and do conditional to replace grads with central value?

    loopy; cupy; numba.cuda.jit?

    """
    import scipy.ndimage

    complex = StencilComplex2D.from_shape((128, 128))

    # source = circle(complex, [8, 8], 4)
    source = rect(complex, [1, 64], [1, 64])
    source += rect(complex, [128-1, 64], [1, 64])
    source += rect(complex, [64, 1], [64, 1])
    source += rect(complex, [64, 128-1], [64, 1])

    outer = 1 - rect(complex, [64, 64], [63, 63])
    source = 1 - rect(complex, [64, 64], [62, 62]) - outer
    # source = rect(complex)

    # outer = source
    # constraint = 1 - source

    # conductance = complex.topology.form(1, init='ones')
    solid = complex.topology.form(0, init='zeros') + circle(complex, [64, 64], 32)
    solid = rect(complex, [64, 64], [40, 40]) - rect(complex, [64, 44], [20, 20])
    solid = scipy.ndimage.gaussian_filter(solid, 1)

    # solid = rect(complex, [64, 64], [5, 40])

    boundary = scipy.ndimage.gaussian_filter(np.abs(np.clip(scipy.ndimage.laplace(solid), -1e9, 1e9)), 0.5)

    source = source - boundary / boundary.sum() * source.sum()


    solid = complex.coarsen[0] * solid
    boundary = complex.coarsen[0] * boundary
    source = complex.coarsen[0] * source
    outer = complex.coarsen[0] * outer

    complex = complex.coarse


    if True:
        complex.plot_0(outer)
        complex.plot_0(source)
        show_plot()

    if False:
        complex.plot_0(source)
        complex.plot_2(complex.topology.averaging_operators_0[2] * source)
        show_plot()



    # inverse of amount of flow generated for a given potential difference
    solid_1 = complex.topology.averaging_operators_0[1] * solid
    resistance = 0.1 + solid_1 * 10
    resistance = complex.topology.form(1, init='ones')

    # should be a zero-form; literally irrelevant here
    constraint = complex.topology.form(0, init='zeros')
    # 1-form. def want this at 1 in this context?
    conductance = complex.topology.form(1, init='ones')
    # conductance = 1.0 - solid_1

    # foo zero inside domain, means interior grad of potential does not produce a flux
    # effective results seems constant potential in interior; any value should do but appears not to matter
    # going to zero appears to form effective barrier to flow, whic his as intended.
    # somehow streamlines are screwed up tho
    foo = 1.001 - solid_1
    # foo = complex.topology.form(1, init='ones')

    fields = {
        'constraint': constraint,
        'conductance': conductance,
        'resistance': resistance,
        'foo': foo
    }

    system = Diffusion.formulate(complex, fields)

    rhs = BlockArray([+source, complex.topology.form(1, init='zeros')])

    hierarchy = system.hierarchy(levels=5)
    from time import time
    t = time()
    from pycomplex.stencil.multigrid import solve_full_cycle
    x = system.solve(rhs, x0=rhs * 0, tol=1e-12)
    # x = system.solve_qmr(rhs, x0=rhs * 0, atol=1e-12)
    # x = solve_full_cycle(hierarchy, rhs, iterations=5)
    # x = system.solve(rhs)
    print()
    print(time() - t)

    complex.plot_0(x[0] * (1-solid))
    # x[1] = system.complex.topology.primal[0](x[0])
    complex.plot_1(x[1] * (1-solid))

    show_plot()
    quit()
