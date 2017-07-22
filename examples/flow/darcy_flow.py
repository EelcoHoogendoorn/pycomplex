"""
darcy flow:
    divergence free
    follows darcy's law; proportionality between gradient of pressure and flow velocity and permeability mu

[mu,  grad] [v] = [f]
[div, 0   ] [P]   [0]

if we model P as a primal-n-form, we get
[I, δ] [v] = [f]
[d, 0] [P]   [0]

[[I, 0, 0],  [δ, 0]]  [vi]   [fi]
[[0, I, 0],  [δ, I]]  [vp] = [fp]
[[0, 0, J],  [0, b]]  [vd]   [fd]

[[d, d, 0],  [0, 0]]  [Pi]   [0]
[[0, I, b],  [0, J]]  [Pd]   [0]

interesting question; should vd always be zero?
if b is zero, eq is quite simple

[[I, 0],  [δ, 0]]  [vi]   [fi]
[[0, I],  [δ, I]]  [vp] = [fp]

[[d, d],  [0, 0]]  [Pi]   [0]
[[0, I],  [0, J]]  [Pd]   [0]

either way, this lends itself perfectly to either solving as second order normal equation,
or directly using minres if we both to make it symmetrical

"""