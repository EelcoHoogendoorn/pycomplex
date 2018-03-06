"""Attempt at geometric multigrid solver
Start with primal 0-form laplacian; but aim for more general framework
Start by cleaning up escheresque solver

MG code should be entirely agnostic about what complex or forms it is working with;
this should be encapsulated by the Equation object

how to transfer boundary conditions in mg-setting is an interesting problem that i havnt given a lot of attention yet.
Equation re-discretizes on each level; maybe need to make BC-setup scale-invariant callable as well?
alternative is to apply petrov-galerkin coarening to the bcs instead
"""


import numpy as np


def v_cycle(hierarchy, rhs, x):
    """Recursive solver V cycle using residual correction

    Parameters
    ----------
    hierarchy: List[Equation]
        discrete equations, from root to finest
    rhs : ndarray, float
        right hand side of equation on finest level
    x : ndarray, float
        current best guess at a solution on finest level

    Returns
    -------
    ndarray, float
        solution with improved error

    Notes
    -----
    currently uses jacobi relaxation
    can also create a variant where each layer uses minres,
    preconditioned by a recursion to the coarse level

    """

    fine = hierarchy[-1]

    # root level recursion break
    if len(hierarchy) == 1:
        return fine.solve(rhs)

    coarse = hierarchy[-2]

    # def profile(func):
    #     """debug output for intermediate steps"""
    #     def inner(ix):
    #         ox = func(ix)
    #         err = np.linalg.norm(fine.residual(ix, rhs).ravel()) - \
    #               np.linalg.norm(fine.residual(ox, rhs).ravel())
    #         print('improvement', func.__name__)
    #         print(err)
    #         return ox
    #     return inner

    # reduce error on the fine equation by overrelaxation
    knots = np.linspace(1, 4, 8, True)  # we need to zero out eigenvalues from largest to factor 4 smaller
##    knots = np.sqrt( (knots-1)) + 1
    def solve_overrelax(x):
        return fine.overrelax(x, rhs, knots)

    def solve_iterate(x, iterations):
        for i in range(iterations):
##            x = fine.jacobi(x, rhs)
            x = fine.descent(x, rhs)
        return x

    def coarsesmooth(x):
        fine_res = fine.residual(x, rhs)
        coarse_res = coarse.restrict(fine_res)
        coarse_error = v_cycle(
            hierarchy[:-1],
            rhs=coarse_res,
            x=np.zeros_like(coarse_res),
            )
        fine_error = coarse.interpolate(coarse_error)
        return x - fine_error      # apply residual correction scheme

    # presmooth    = (solve_iterate)
    # postsmooth   = (solve_iterate)
    # coarsesmooth = (coarsesmooth)

##    x = presmooth(x, 5)
    x = solve_overrelax(x)
    x = coarsesmooth(x)
##    x = postsmooth(x, 5)
    x = solve_overrelax(x)

    return x


def solve_v_cycle(hierarchy, rhs, x0=None, iterations=10):
    """Repeated v-cycle multigrid solver.

    Not as efficient as full cycle but conceptually simpler
    """
    if x0 is None:
        x = np.zeros_like(rhs)

    for i in range(iterations):
        # print(i, np.linalg.norm(equation.residual(x, rhs)))
        x = v_cycle(hierarchy, rhs, x)
    return x


def solve_full_cycle(hierarchy, rhs):
    """Full multigrid schema

    first restrict towards and solve on coarsest level
    then prolongate and do a single v-cycle on that level

    Parameters
    ----------
    hierarchy : List[Equation]
    rhs : ndarray, float

    Returns
    -------
    ndarray, float
    """
    fine = hierarchy[-1]

    # root level break
    if len(hierarchy) == 1:
        return fine.solve(rhs)

    coarse = hierarchy[-2]

    # get solution on coarser level first
    x = coarse.interpolate(solve_full_cycle(hierarchy[:-1], coarse.restrict(rhs)))

    # do some V-cycles
    for i in range(2):
        x = v_cycle(hierarchy, rhs, x)

    # print residual
    print(np.linalg.norm(fine.residual(x, rhs)))

    return x
