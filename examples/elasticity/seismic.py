"""
Set up scenario showing s and p waves in isotropic elastic material,
as a function of position and velocity fields p and v
and material parameters ρ, μ and λ

(μ dδ + λ δd) p = d/dt (v * ρ)
              v = d/dt p

Do simple explicit timestepping of this equation

We can let material properties go to small values to simulate free boundary,
or to large values to simulate fixed boundary

https://authors.library.caltech.edu/62269/2/LMHTD15.pdf
In this paper fixed boundaries are simulated by hodge going to zero
"""