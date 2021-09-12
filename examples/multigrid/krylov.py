import numpy as np
import scipy.sparse


def krylov_cycle(hierarchy, y, x=None, maxiter=4):
    """Solve using recursive krylov method

    Wait; how dumb is this? if every iter does a precondition call to lower level,
    number of times we operate on each level grows to the power of depth.
    so every level requires the same amount of work; or n*depth performance, or nlogn,
    given that the number of iters per level is a small constant number
    if number of iters is larger than reduction factor, work actually grows exponentially per level.
    but perhaps 4 fully preconditioned krylov iterations is just a great solver

    So in theory this isnt great; but in practice we shall see.
    expectation is improved resistance to anisotropy due to more powerfull solver than gradient descent,
    closing the gap with amg
    """
    if x is None:
        x = np.zeros_like(y)

    *tail, fine = hierarchy

    if len(hierarchy) == 1:
        return fine.solve(y)
    else:
        return fine.solve_minres(
            y=y,
            x0=x,
            preconditioner=as_krylov_preconditioner(hierarchy),
            tol=1e-9,
            maxiter=maxiter,
        )


def as_krylov_preconditioner(hierarchy):
    """Wrap krylov solver as a preconditioner

    Parameters
    ----------
    hierarchy : List[Equation]

    Returns
    -------
    LinearOperator
        Linear operator that approximately inverts the linear relation Ax=By,
        returning an approximate x given a y

    """
    *tail, fine = hierarchy

    return scipy.sparse.linalg.LinearOperator(
        shape=fine.A.shape,
        matvec=lambda y: fine.interpolate(krylov_cycle(tail, fine.restrict(fine.BI * y)))
    )
