"""
Implement convolution stencil based exterior derivatives and grid transfer operators

having these operators present would allow easy testing of vector-laplace friendly mg solvers,
through first order representation and grid transfer with least-squared jacobi-relaxation

should work for both elasticity and stokes; but start with 2d heat transfer
all boundary conditions are of an immersive type for these applications


applications:
    heat solver for wall thickness analysis.
    model mold as higher temperature condition cells, where heat 'leaks away' in proportion to temp?
    with source a func of temp we get a 'shielded poisson' behavior away from surface

    eigen mode solver for stiffness
    use air cell method outside as bc.

    hard to remove support / tight cavities
    stokes with uniform divergece outside the part; check resulting pressure of 'expanding foam'

    heat solver outside the part, to detect tiny hard to cut regions?

    can we do some type of flow analysis inside the part? can we identify
    the point of easiest mold fill, for instance? akin to finding to point most distant from everything,
    in some flow-ease weighted manner
    viewed more simply; we can eigensolve compressible stokes problem, with hard boundary conditions
    interestingly, this is identical to elasticity approach, with the exception of boundary conditions

"""
