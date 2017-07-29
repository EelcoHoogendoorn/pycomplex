"""Time-dependent diffusion is interesting because it arguably represents
the simplest 'introduction to vector calculus' possible, while still doing something physical and useful.

"""


from pycomplex import synthetic
grid = synthetic.n_cube_grid((32, 32))

grid.plot()