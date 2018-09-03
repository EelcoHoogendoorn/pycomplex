
# -*- coding: utf-8 -*-

"""Electrostatic equations

Particularly interested here if we can see something about surface charge and its interactions,
by looking carefully at the boundary terms

Physics:
    div E = rho     divergence is charge density
    curl E = 0      static electric field is irrotational

It is tempting to model this as a potential problem straight away;
E = grad V
But we will hold off with that for a little, for the sake of generality

Or in DEC:
    dE = rho
    δE = 0

With E a 1-form on a 2d manifold, and a 1-form on a 3d manifold

In the 2d case, the equations thus formed are truly indistinguishable from the magnetostatic case!
In practice, we have different solution strategies available to us however,
considering that the absence of source terms in δE = 0 allows us to write this in terms of a scalar potential,
which isnt the case for magnetostatics in higher dimensions.
The only difference between 2d electric and magnetic fields is in their source term;
aside form this they are both irrotational and incompressible

In the 3d case, electric is a primal 1-flux and magnetic a primal 2-flux
static solutions to both in 3d are still very similar though,
in that we have a non scalar form that closed as both primal and dual form

can we solve a static problem with mobile charges?
we can say that total charge must be conserved in some area,
and that E field must vanish inside conductors.
this requires us to make rho part of unknowns somehow
can we place conducting cylinder in field, for instance?
for E to zero here; will violate closedness of form without explicit
modelling of surface current though, I think
lines of electric field always at right angles to the surface

can use the following system:

with V dual-0
[δ, 0, 0] [E]   [0]     curl(E)=0
[I, δ, 0] [V] = [0]     grad(V)=E
[d, 0, I] [r]   [r0]    div(E)=r+r0

or without V; is V really necessary in this? nice for bcs I guess; but otherwise?
note that we should only permit nonzero r0 inside conductor
yet conductor is defined on grad(V) equation... is V equation required here after all?
perhaps it is; we zero out some aspects of the system;
though would mostly expect to need it to replace zero terms in curl equation?
[δ, 0] [E]   [0]
[d, I] [r] = [r0]

note that top and right row/col accomplish the same. if i had to choose would rather
keep V in this scenario as it can be useful in bcs; rotation of E is never relevant though
note the absence of the curl of rho term; makes sense physically, but curious mathematically

additional constraints are E zero inside the conductor,
and rho zero outside the conductor
these can both be set on the middle equation
rho and V are both forms of the same kind, so we cannot slice the system from the canonical one;
need to concat the rho-related section

still havnt fully figured this out yet... in grad(V)=E equation, how to represent a conductor?
if diagonal goes to zero, grad(V) needs to be zero, and any E is fine, subject to curl and div
seems like this is fine in first-order equation solver
does it necessitate a mobile surface charge, however?

"""