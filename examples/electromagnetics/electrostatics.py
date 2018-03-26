
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
which isnt the case for magnetostatics

In the 3d case



"""