"""Some sparse linear solvers useful in this context"""

import numpy as np


def solve_cg(deflate, operator, rhs, x0=None):
    """
    solve operator(x) = rhs using conjugate gradient iteration
    operator must be symmetric

    The deflate argument allows specification of constraints
    
    Parameters
    ----------
    deflate : callable that will project its argument to a subspace
    operator : callable that will apply the operator to be inverted
    rhs : ndarray, [n], float
    x0 : ndarray, [n], float, optional

    Returns
    -------
    x : ndarray, [n], float
        solution to the constrained system

    TODO: make a minres version of the same

    """
    if x0 is None:
        x0 = np.zeros_like(rhs)

    def dot(x,y):
        return np.dot(np.ravel(x), np.ravel(y))
##    def ortho(x,d):
##        x -= d * dot(x,d)
##    def project(x):
##        for d in deflation:
##            ortho(x, d)


    eps = 1e-9
    max_iter = 3000

    x = np.copy(x0)
    deflate(x)
    r = rhs - operator(x)
    deflate(r)

    d = np.copy(r)
    delta_new = dot(r,r)
    delta_0 = delta_new

    for i in range(max_iter):
        deflate(d)
        q = operator(d)
        deflate(q)

        dq = dot(d, q)
        if np.abs(dq) < eps: break

        alpha = delta_new / dq

        x += alpha * d
        deflate(x)
        if i%50==49:
            r = rhs - operator(x)
        else:
            r -= alpha * q
        deflate(r)

        if delta_new/delta_0 < eps:
            break
        if dot(x0,x) / dot(x0,x0) > 1e30:        #guard against runaway
            break

        delta_old = delta_new
        delta_new = dot(r,r)
        beta = delta_new / delta_old
        d = r + beta * d
    return x


def solve_minres(A, b, x0=None, maxiter=None, shift=0.0, tol=1e-5):
    """Vectorized variant of minres to solve many small least-squares problems efficiently"""

    n = A.shape[-1]

    matvec = lambda x: np.dot(A, x)
    psolve = lambda x: x        # leave out preconditioner for now

    if maxiter is None:
        maxiter = 5 * n

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = x0.dtype

    eps = np.finfo(xtype).eps

    x = np.zeros(n, dtype=xtype)

    # Set up y and v for the first Lanczos vector v1.
    # y  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.

    y = b
    r1 = b

    y = psolve(b)

    beta1 = np.inner(b,y)

    if beta1 < 0:
        raise ValueError('indefinite preconditioner')

    beta1 = np.sqrt(beta1)


    # Initialize other quantities
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    ynorm2 = 0
    cs = -1
    sn = 0
    w = np.zeros(n, dtype=xtype)
    w2 = np.zeros(n, dtype=xtype)
    r2 = r1


    while itn < maxiter:
        itn += 1

        s = 1.0/beta
        v = s*y

        y = matvec(v)
        y = y - shift * v

        if itn >= 2:
            y = y - (beta/oldb)*r1

        alfa = np.inner(v,y)
        y = y - (alfa/beta)*r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = np.inner(r2,y)
        if beta < 0:
            raise ValueError('non-symmetric matrix')
        beta = np.sqrt(beta)
        tnorm2 += alfa**2 + oldb**2 + beta**2

        if itn == 1:
            if beta/beta1 <= 10*eps:
                istop = -1  # Terminate later
            # tnorm2 = alfa**2 ??
            gmax = abs(alfa)
            gmin = gmax

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
        epsln = sn * beta     # epsln2 = 0         epslnk+1
        dbar = - cs * beta   # dbar 2 = beta2     dbar k+1
        root = np.linalg.norm([gbar, dbar])
        Arnorm = phibar * root

        # Compute the next plane rotation Qk

        gamma = np.linalg.norm([gbar, beta])       # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma             # ck
        sn = beta / gamma             # sk
        phi = cs * phibar              # phik
        phibar = sn * phibar              # phibark+1

        # Update  x.

        denom = 1.0/gamma
        w1 = w2
        w2 = w
        w = (v - oldeps*w1 - delta*w2) * denom
        x = x + phi*w

        # Go round again.

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        ynorm2 = z**2 + ynorm2
        rhs1 = rhs2 - delta*z
        rhs2 = - epsln*z

        # Estimate various norms and test for convergence.

        Anorm = np.sqrt(tnorm2)
        ynorm = np.sqrt(ynorm2)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * tol
        diag = gbar

        if diag == 0:
            diag = epsa

        qrnorm = phibar
        rnorm = qrnorm
        test1 = rnorm / (Anorm*ynorm)    # ||r||  / (||A|| ||x||)
        test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q * H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax/gmin

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        if istop == 0:
            t1 = 1 + test1      # These tests work if tol < eps
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1/eps:
                istop = 4
            if epsx >= beta:
                istop = 3
            # if rnorm <= epsx   : istop = 2
            # if rnorm <= epsr   : istop = 1
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1

        if istop != 0:
            break  # TODO check this

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return x
