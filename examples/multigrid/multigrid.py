"""Attempt at geometric multigrid solver
Start with primal 0-form laplacian; but aim for more general framework
Start by cleaning up escheresque solver

MG code should be entirely agnostic about what complex or forms it is working with;
this should be encapsulated by the Equation object

currently only a single `-`, zeros_like and linalg.norm left;
could easily be wrapped on equation object so we can work with block vectors and such



how to transfer boundary conditions in mg-setting is an interesting problem that i havnt given a lot of attention yet.
Equation re-discretizes on each level; maybe need to make BC-setup scale-invariant callable as well?
alternative is to apply petrov-galerkin coarening to the bcs instead
"""


import numpy as np


def v_cycle(hierarchy, y, x=None):
    """Recursive solver V cycle using residual correction

    Parameters
    ----------
    hierarchy: List[Equation]
        discrete equations, from root to finest
    y : ndarray, float
        right hand side of equation on finest level
    x : ndarray, float
        current best guess at a solution on finest level

    Returns
    -------
    x : ndarray, float
        solution with improved error

    Notes
    -----
    currently uses jacobi relaxation
    can also create a variant where each layer uses minres,
    preconditioned by a recursion to the coarse level

    """
    if x is None:
        x = np.zeros_like(y)

    fine = hierarchy[-1]

    # root level recursion break
    if len(hierarchy) == 1:
        return fine.solve(y)

    # reduce error on the fine equation by overrelaxation
    knots = np.linspace(1, 4, 8, True)  # we need to zero out eigenvalues from largest to factor 4 smaller
##    knots = np.sqrt( (knots-1)) + 1
    def solve_overrelax(x):
        return fine.overrelax(x, y, knots)

    def solve_iterate(x, iterations):
        for i in range(iterations):
##            x = fine.jacobi(x, rhs)
            x = fine.descent(x, y)
        return x

    def coarsesmooth(x):
        fine_res = fine.residual(x, y)
        coarse_res = fine.restrict(fine_res)
        coarse_error = v_cycle(hierarchy[:-1], y=coarse_res)
        fine_error = fine.interpolate(coarse_error)
        return x - fine_error      # apply residual correction scheme

    # presmooth    = (solve_iterate)
    # postsmooth   = (solve_iterate)
    # coarsesmooth = (coarsesmooth)

    # x = presmooth(x, 5)
    x = solve_overrelax(x)
    x = coarsesmooth(x)
    # x = postsmooth(x, 5)
    x = solve_overrelax(x)

    return x


def solve_v_cycle(hierarchy, y, x=None, iterations=10):
    """Repeated v-cycle multigrid solver.

    Not as efficient as full cycle but conceptually simpler
    """
    for i in range(iterations):
        x = v_cycle(hierarchy, y, x)
    return x


def solve_full_cycle(hierarchy, y, iterations=2):
    """Full recursive multigrid schema

    First restrict towards and solve on coarsest level
    then prolongate and do a single v-cycle on that level

    Parameters
    ----------
    hierarchy : List[Equation]
    y : ndarray, float
        right hand side
    iterations : int
        number of v-cycles to correct coarse result

    Returns
    -------
    x : ndarray, float
    """

    fine = hierarchy[-1]

    # root level break
    if len(hierarchy) == 1:
        return fine.solve(y)

    # get solution on coarser level first
    x = fine.interpolate(solve_full_cycle(hierarchy[:-1], y=fine.restrict(y)))

    # do some V-cycles to correct residual error
    x = solve_v_cycle(hierarchy, y=y, x=x, iterations=iterations)

    return x
