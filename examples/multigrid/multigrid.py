"""Geometric multigrid solver

MG code should be entirely agnostic about what complex or forms it is working with;
this should be encapsulated by the Equation object

currently only a single `-`, zeros_like and linalg.norm left;
could easily be wrapped on equation object so we can work with block vectors and such


how to transfer boundary conditions in mg-setting is an interesting problem that i havnt given a lot of attention yet.
Equation re-discretizes on each level; maybe need to make BC-setup scale-invariant callable as well?
alternative is to apply petrov-galerkin coarening to the bcs instead

would like to implement a recursive-krylov-cycle as well, where we solve using a krylov method at each level,
and precondition with the coarse grid operator at each step. This should be a much stronger solver than gradient descent
or jacobi iteration, and hopefully cope better with anisotropy
"""


import numpy as np


def v_cycle(hierarchy, y, x=None):
    """Recursive V cycle using residual correction

    Parameters
    ----------
    hierarchy: List[Equation]
        discrete equations, from root to finest
    y : ndarray, float
        right hand side of equation on finest level
    x : ndarray, float, optional
        current best guess at a solution on finest level

    Returns
    -------
    x : ndarray, float
        solution with improved error

    Notes
    -----
    currently uses smoothing iteration
    can also create a variant where each layer uses minres,
    preconditioned by a recursion to the coarse level
    basically; replace the simple descent we have now with a krylov-solver
    this might have quite some benefit when lower layers start shear locking for instance;
    instead of applying the coarse refinement directly; treat them as suggested directions
    and apply them more nuanced in the context of the fine problem

    """
    if x is None:
        x = np.zeros_like(y)

    fine = hierarchy[-1]

    # root level recursion break
    if len(hierarchy) == 1:
        return fine.solve(y)

    def coarsesmooth(x):
        fine_res = fine.residual(x, y)
        coarse_res = fine.restrict(fine_res)
        coarse_error = v_cycle(hierarchy[:-1], y=coarse_res)
        fine_error = fine.interpolate(coarse_error)
        return x - fine_error      # apply residual correction scheme

    x = fine.smooth(x, y)       # presmooth
    x = coarsesmooth(x)
    x = fine.smooth(x, y)       # postsmooth

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
    x = fine.interpolate(solve_full_cycle(hierarchy[:-1], y=fine.restrict(y), iterations=iterations))

    # do some V-cycles to correct residual error
    x = solve_v_cycle(hierarchy, y=y, x=x, iterations=iterations)

    return x


# FIXME: this should be a method of the hierarchy?
def as_preconditioner(hierarchy):
    """Runtime performance very non-deterministic still

    Parameters
    ----------
    hierarchy : List[Equation]

    Returns
    -------
    LinearOperator
        Linear operator that approximately inverts the linear relation Ax=By,
        returning an approximate x given a y

    """
    import scipy.sparse
    A, B, Bi = hierarchy[-1].operators
    def inner(y):
        # FIXME: expose inner iteration parameters and restrict them; single smooth v-cycle best precondition found so far
        # return solve_full_cycle(hierarchy, Bi * y, iterations=1)
        return v_cycle(hierarchy, Bi * y)
    return scipy.sparse.linalg.LinearOperator(
        shape=A.shape,
        matvec=inner
    )
