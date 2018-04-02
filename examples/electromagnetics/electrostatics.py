
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


"""