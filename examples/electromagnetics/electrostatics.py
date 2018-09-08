
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

note that V is continuous but E is discontinuous over a surface charge.
just as temperature flux is discontinuous over a line of sources
for a superconductor, solution does not involve discontinuous flux though;
or does it? flux conservation demands continuous flux; but with zero gradient there is none to be determined
might be best modelled as an effective source/sink at the surface?

so can we obtain surface charge by means of limiting process? reduce resistance until gradient
in conductor approaches zero; and then consider divergence-violation when set to zero

wait; in superconducting matter flux is not zero; only temp gradient is.
makes flux underconstrained by potential; but still solvable using relaxation;
and could throw a zero-curl in there for good measure? anyway we need no source/sink modelling

by contrast, in electrostatic conductor, it is the vector field explicitly that must become zero,
in order to reach equilibrium. dont think there is a difference as viewed from the outside tho?
potential looks the same in any case no? is this strange? not really; charge redistribution
within a conductor is never felt outside the conductor

so for conductor in electric field, we define a (soft) conductivity mask,
which is 1 in dielectric and zero in conductor, we move inverse permittivity to zero accordingly,
then solve for the E-field, which may have indeterminate components within the conductor.
then, we multiply the E-field with our mask, zeroing it out inside the conductor,
and then we may find the bound surface charged by simply taking the div of this modified E-field

when working in terms of curl and div, without potential, is solving superconductor equally easy?
C * grad(P) = flux, or grad(P) = flux * R, so curl(flux * R) = 0 has the same information content.
or similar...

in curl form, R going to zero means we only have a div constraint left, but otherwise anything goes
with potential, grad(P) is forced to zero. flux however is left unconstrained;
but we lose the ability to derive flux from the potential, which really calls into question its use

what about C going to zero? does not happen in electro case, but anyway, still want to simulate it.
this is  trivial in the second order potential formulation; just zero flux for any grad(P)

[δ, 0] [E]   [0]     curl(E)=0
[I, δ] [V] = [0]     E = -grad(V) * mu
[d, 0]       [r0]    div(E)=r0

what happens if we leave it to curl equation?
resistance going to infinity will also push corresponding flux to zero,
but literally plugging in np.inf most likely does not fly?
what if flux * R is our unknown to solve for? would need multiply with C in div,
which would lead to identical trouble?
perhaps a decoupling is needed; a seperate equation to capture this divide

[C, R, 0]       [0]     C * G = R * F
[δ, 0, 0] [G]   [0]     curl(G)=0
[I, 0, δ] [F] = [0]     G = -grad(P)
[0, d, 0] [P]   [r0]    div(F)=r0

would potential-based system be able to handle both zero-C and zero-R? I think so.
since it is a two-term equation we can move the zero to either side
in magnetics we need the curl relations though;
need to consider how zeros in material properties work out there too
yeah seems like it would work; both for flow blocking obdtacle in potential flow
as well as superconductor in magnetic field. taking the curl of the masked field
should give surface current or vorticity distribution required to zero out field under the mask
in terms of streamfunction, this would simply be an area of constant streamfunction,
thus excluding flowlines

what about zero heat conduction? this is the same as streamfunction,
except we interpret isolator with flat potential as area of zero normal flux, instead of zero tangent flux
so to simulate perfect diffusion isolator we also constrain the divergence under the mask

seems like splitting the constitutive relations into a seperate equation solves the problem in a general way;
but at the same time it seems like it would also increase computational load,
since information will take longer to diffuse over the grid.

also, the immerse boundaries again show the strength of considering the equations
in first-order form, just as in a complex with a boundary
"""