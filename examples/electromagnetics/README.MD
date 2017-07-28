
TODO
----
- 2d magnetostatics example
- 2d electrostatics example
- 2d electrodynamics using 'staggered grids'
    note that either E or B must revert to a scalar; and current or charge must disappear
- 2-1 spacetime magnetodynamics example with lorentzian metric


Thought of the day
------------------
The often contentious vector vs scalar potential approach in EM field problems is
every bit as much as artifact of mathematical tools as the
streamfunction/flow potential distinction is in fluid mechanics

Physical governing equations of EM field theory are:
    dF = 0
    δF = J
With F a 2-form on a 3d spacetime manifold

Or looking at the 2d magnetostatic component of that for simplicity
    dB = 0      magnetic flux is divergence-free
    δB = J      magnetic flux is irrotational, where no curl is present
With B a 1-form on a 2d manifold

“When a flow is both frictionless and irrotational, pleasant things happen.” –F.M.
White, Fluid Mechanics 4th ed.

The structure of these equations, and the question of how to best solve them,
is entirely analogous to the potential flow case. Yes, we could right-multiply
with either δ or d to trivialize one equation and turn the other into a laplacian.
But it is important to realize that both are a mathematical trick that may (or may not)
be a convenient method for constructing solutions to your problem, and neither has any
physical necessity for describing the behavior of our system!

 [d]   = [0]
 [δ] B = [J]

Rather than multiplying these equations from the right, or writing B as the exterior derivative of something,
we can also multiply from the left; with the system of equations itself, thus forming the normal equations.

If we used B=dA wed get a system of equations in terms of 0-forms, and if we used B=δψ we get a square system of 2-forms
But forming the normal equations results in a laplace-beltrami over 1-forms, or the space of B itself.
This results in a system with about the same number of unknowns as both potentials combined,
but it remains to be seen how that pays off in terms of condition number.


What about boundary conditions in this formulation?
Zero normal flux is a symmetry line
tangent flux should be meaningless; or is it?
how to specify anti-symmetry of solution? zero flux would do perhaps?

[[δ, 0, 0]]         [Ji]     curl; nothing to see here
[[δ, δ, I]]  [Bi]   [Jp]
             [Bp]
[[d, d, 0]]  [Bd]   [0]     divergence; no dependence on dual boundary tangent flux; makes sense
[[0, I, b]]         [bc]    implies normal flux constraint; ok. implied tangent flux change; antisymmetry if 0?


to what extent can we trick minres into helping us?

how different is this in 3d?

[[0, 0], [δ, 0, 0], [0, 0, 0], [0, 0]] [0i]   [0i]
[[0, 0], [δ, a, I], [0, 0, 0], [0, 0]] [0p]   [0p] a I1 = I2 a

[[d, d], [0, 0, 0], [δ, 0, 0], [0, 0]] [1i]   [1i]
[[0, a], [0, 0, 0], [δ, b, I], [0, 0]] [1p] = [1p] b I0 = I1 b
[[0, _], [0, 0, _], [0, 0, a], [0, 0]] [1d]   [1d]

[[0, 0], [d, d, 0], [0, 0, 0], [δ, 0]] [2i]   [2i]
[[0, 0], [0, b, 0], [0, 0, 0], [I, I]] [2p] = [2p]  I0 = I
[[0, 0], [0, _, _], [0, 0, _], [0, b]] [2d]   [2d]

[[0, 0], [0, 0, 0], [d, I, 0], [0, 0]] [3i]   [3i]
[[0, 0], [0, 0, 0], [0, _, _], [0, _]] [3d]   [3d]

if magnetic flux is a primal 2-form:

[[δ, 0, 0]]         [Ji]     curl; nothing to see here. rhs is
[[δ, δ, I]]         [Jp]
[[0, 0, b]]  [Bi]   [Jd]    does this row have physical meaning? jump in tangent flux over boundary is surface current?
             [Bp]
[[d, d, 0]]  [Bd]   [0]     divergence; no dependence on dual boundary tangent flux; makes sense
[[0, I, b]]         [bc]    implies normal flux constraint; ok. implied tangent flux change; antisymmetry if 0?

It is hardly directly evident to me what necessary or sufficient boundary conditions are from looking at this system.
Do two rows of equations suggest two independent rows of boundary equations? Or does dual operator not count?
Specifying normal everywhere does generally not suffice; leaves one open constraint per boundary contour.
